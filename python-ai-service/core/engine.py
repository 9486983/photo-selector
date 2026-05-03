from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import pkgutil
from pathlib import Path

import cv2
import numpy as np

from core.config import EngineConfig, load_config
from core.plugin_base import BasePlugin, PluginOutput
from core.scheduler import GpuCpuScheduler

logger = logging.getLogger(__name__)


class AIEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or load_config()
        self.plugins: list[BasePlugin] = []

        self.scheduler = GpuCpuScheduler(
            gpu_workers=self.config.gpu_workers if self.config.gpu_enabled else 0,
            cpu_workers=self.config.cpu_workers,
        )
        self.batch_parallelism = max(1, self.config.batch_parallelism)

        self.score_weights = dict(self.config.score_weights)
        self.waste_thresholds = dict(self.config.waste_thresholds)

        self.state_path = Path(__file__).resolve().parent.parent / "data" / "learning_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.feedback_stats: dict[str, int] = {
            "good_overruled_waste": 0,
            "waste_overruled_good": 0,
            "portrait_feedback": 0,
        }
        self.face_identities: dict[str, list[float]] = {}
        self.image_person_map: dict[str, str] = {}
        self.confirmed_mappings: dict[str, str] = {}
        self.person_gallery: dict[str, list[list[float]]] = {}
        self.next_person_id = 1
        self._load_learning_state()
        self._ensure_weight_defaults()

    @property
    def plugin_names(self) -> list[str]:
        return [plugin.name for plugin in self.plugins]

    def load_plugins(self) -> None:
        self.plugins.clear()
        package = "plugins"
        package_path = Path(__file__).parent.parent / package

        for module_info in pkgutil.iter_modules([str(package_path)]):
            if module_info.name.startswith("_"):
                continue
            module = importlib.import_module(f"{package}.{module_info.name}")
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                    plugin = obj()
                    pc = self.config.plugins.get(plugin.name)
                    if pc is not None and not pc.enabled:
                        logger.info("Plugin %s disabled by config", plugin.name)
                        continue
                    self.plugins.append(plugin)

        self.plugins.sort(key=lambda p: p.priority)
        logger.info("Loaded %d plugins: %s", len(self.plugins), [p.name for p in self.plugins])

    async def startup(self) -> None:
        await self.scheduler.start()
        asyncio.create_task(self._preload_plugins())

    async def _preload_plugins(self) -> None:
        for plugin in self.plugins:
            if not plugin.is_loaded:
                try:
                    await asyncio.to_thread(plugin.load)
                    logger.info("Plugin %s loaded successfully", plugin.name)
                except Exception as ex:
                    logger.warning("Plugin %s preload failed: %s", plugin.name, ex)

    async def shutdown(self) -> None:
        await self.scheduler.stop()
        for plugin in self.plugins:
            try:
                plugin.unload()
            except Exception as ex:
                logger.warning("Plugin %s unload failed: %s", plugin.name, ex)
        self._save_learning_state()

    async def analyze(self, image_path: str) -> dict:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unsupported image or decode failed: {image_path}")

        return await self._analyze_with_image(image_path, image)

    async def _analyze_with_image(self, image_path: str, image: np.ndarray) -> dict:
        tasks = []
        for plugin in self.plugins:
            if not plugin.is_loaded:
                await asyncio.to_thread(plugin.ensure_loaded)
            tasks.append(self.scheduler.submit(plugin.requires_gpu, plugin.analyze, image_path, image))

        plugin_outputs = await asyncio.gather(*tasks)
        return self._compute_result(image_path, image, plugin_outputs)

    def _compute_result(self, image_path: str, image: np.ndarray, plugin_outputs: list[PluginOutput]) -> dict:
        sharpness_score = self._estimate_sharpness(image)
        exposure_score = self._estimate_exposure(image)
        yolo_score = self._find_plugin_score(plugin_outputs, "yolo")
        style_score = self._find_plugin_score(plugin_outputs, "curation")
        person_count = self._extract_feature_int(plugin_outputs, "yolo", "person_count", default=0)
        face_signature = self._extract_feature_list(plugin_outputs, "yolo", "face_signature", default=[])
        style_label = self._extract_feature_str(plugin_outputs, "curation", "style_label", default="unknown")
        color_label = self._extract_feature_str(plugin_outputs, "curation", "color_label", default="unknown")
        dominant_colors = self._extract_feature_list(plugin_outputs, "curation", "dominant_colors", default=[])
        phash = self._perceptual_hash(image)
        person_label = self._resolve_person_label(face_signature, person_count, phash, image_path)

        person_score = min(1.0, person_count / 2.0)
        overall_score = (
            sharpness_score * self.score_weights.get("sharpness", 0.32)
            + exposure_score * self.score_weights.get("exposure", 0.24)
            + yolo_score * self.score_weights.get("object", 0.20)
            + person_score * self.score_weights.get("person", 0.18)
            + style_score * self.score_weights.get("style", 0.06)
        )
        overall_score = self._clamp(overall_score, 0.0, 1.0)

        is_waste, waste_reason = self._is_waste(
            sharpness_score=sharpness_score,
            exposure_score=exposure_score,
            person_count=person_count,
            overall_score=overall_score,
        )

        auto_class = self._auto_classify(person_count, style_label, sharpness_score, is_waste)

        composition_score = self._find_plugin_score(plugin_outputs, "composition")
        exposure_quality = self._find_plugin_score(plugin_outputs, "exposure")

        return {
            "image_path": image_path,
            "overall_score": round(float(overall_score), 4),
            "sharpness_score": round(float(sharpness_score), 4),
            "exposure_score": round(float(exposure_score), 4),
            "eyes_closed": False,
            "is_duplicate": False,
            "face_count": person_count,
            "person_label": person_label,
            "style_label": style_label,
            "color_label": color_label,
            "dominant_colors": dominant_colors,
            "auto_class": auto_class,
            "is_waste": is_waste,
            "waste_reason": waste_reason,
            "composition_score": round(float(composition_score), 4),
            "exposure_quality": round(float(exposure_quality), 4),
            "plugins": [
                {
                    "plugin_name": out.plugin_name,
                    "score": out.score,
                    "objects": out.objects,
                    "features": out.features,
                }
                for out in plugin_outputs
            ],
        }

    async def analyze_many(self, image_paths: list[str]) -> list[dict]:
        images: list[np.ndarray | None] = []
        for path in image_paths:
            try:
                images.append(cv2.imread(path))
            except Exception:
                images.append(None)

        valid = [(p, img) for p, img in zip(image_paths, images) if img is not None]
        if not valid:
            return []

        valid_paths, valid_images = zip(*valid)

        per_image_outputs: dict[str, list[PluginOutput]] = {p: [] for p in valid_paths}

        for plugin in self.plugins:
            if not plugin.is_loaded:
                try:
                    await asyncio.to_thread(plugin.ensure_loaded)
                except Exception as ex:
                    logger.warning("Plugin %s failed to load: %s", plugin.name, ex)
                    continue

            if hasattr(plugin, "analyze_batch") and callable(plugin.analyze_batch):
                try:
                    items = list(zip(valid_paths, valid_images))
                    outputs = await asyncio.to_thread(plugin.analyze_batch, items)
                    for path, out in zip(valid_paths, outputs):
                        per_image_outputs[path].append(out)
                except Exception as ex:
                    logger.warning("Plugin %s batch failed, fallback to per-image: %s", plugin.name, ex)
                    for path, img in zip(valid_paths, valid_images):
                        try:
                            out = await self.scheduler.submit(plugin.requires_gpu, plugin.analyze, path, img)
                            per_image_outputs[path].append(out)
                        except Exception:
                            pass
            else:
                semaphore = asyncio.Semaphore(self.batch_parallelism)

                async def _run_plugin(p: str, img: np.ndarray, pl: BasePlugin) -> PluginOutput:
                    async with semaphore:
                        return await self.scheduler.submit(pl.requires_gpu, pl.analyze, p, img)

                tasks = [_run_plugin(p, img, plugin) for p, img in zip(valid_paths, valid_images)]
                outputs = await asyncio.gather(*tasks)
                for path, out in zip(valid_paths, outputs):
                    per_image_outputs[path].append(out)

        return [self._compute_result(p, img, per_image_outputs[p]) for p, img in zip(valid_paths, valid_images)]

    def apply_feedback(self, manual_tag: str, predicted_is_waste: bool, predicted_face_count: int) -> dict:
        tag = manual_tag.lower().strip()
        if tag == "good" and predicted_is_waste:
            self.feedback_stats["good_overruled_waste"] += 1
            self.waste_thresholds["soft_blur"] = self._clamp(self.waste_thresholds["soft_blur"] - 0.004, 0.035, 0.09)
            self.waste_thresholds["low_score_no_person"] = self._clamp(
                self.waste_thresholds["low_score_no_person"] - 0.006, 0.12, 0.25
            )
        elif tag == "waste" and not predicted_is_waste:
            self.feedback_stats["waste_overruled_good"] += 1
            self.waste_thresholds["soft_blur"] = self._clamp(self.waste_thresholds["soft_blur"] + 0.004, 0.035, 0.09)
            self.waste_thresholds["low_score_no_person"] = self._clamp(
                self.waste_thresholds["low_score_no_person"] + 0.006, 0.12, 0.25
            )
        elif tag == "portrait":
            self.feedback_stats["portrait_feedback"] += 1
            if predicted_face_count == 0:
                self.score_weights["person"] = self._clamp(self.score_weights["person"] + 0.01, 0.10, 0.30)
                self.score_weights["object"] = self._clamp(self.score_weights["object"] - 0.01, 0.10, 0.28)

        self._save_learning_state()
        return {
            "score_weights": self.score_weights,
            "waste_thresholds": self.waste_thresholds,
            "feedback_stats": self.feedback_stats,
        }

    @staticmethod
    def _estimate_sharpness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(max(score / 1200.0, 0.0), 1.0)

    @staticmethod
    def _estimate_exposure(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        diff = abs(brightness - 128.0)
        return max(0.0, 1.0 - diff / 128.0)

    @staticmethod
    def _find_plugin_score(outputs: list[PluginOutput], plugin_name: str) -> float:
        for out in outputs:
            if out.plugin_name == plugin_name:
                return float(out.score)
        return 0.0

    @staticmethod
    def _extract_feature_int(outputs: list[PluginOutput], plugin_name: str, key: str, default: int = 0) -> int:
        for out in outputs:
            if out.plugin_name == plugin_name:
                val = out.features.get(key, default)
                try:
                    return int(val)
                except Exception:
                    return default
        return default

    @staticmethod
    def _extract_feature_str(outputs: list[PluginOutput], plugin_name: str, key: str, default: str = "") -> str:
        for out in outputs:
            if out.plugin_name == plugin_name:
                val = out.features.get(key, default)
                return str(val)
        return default

    @staticmethod
    def _extract_feature_list(outputs: list[PluginOutput], plugin_name: str, key: str, default: list | None = None) -> list:
        fallback = default or []
        for out in outputs:
            if out.plugin_name == plugin_name:
                val = out.features.get(key, fallback)
                if isinstance(val, list):
                    normalized = []
                    for x in val:
                        try:
                            normalized.append(float(x))
                        except Exception:
                            normalized.append(str(x))
                    return normalized
                return fallback
        return fallback

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    def _is_waste(self, sharpness_score: float, exposure_score: float, person_count: int, overall_score: float) -> tuple[bool, str]:
        reasons: list[str] = []
        risk = 0.0

        if sharpness_score < self.waste_thresholds["hard_blur"]:
            reasons.append("hard_blur")
            risk += 0.55
        elif sharpness_score < self.waste_thresholds["soft_blur"]:
            reasons.append("soft_blur")
            risk += 0.25

        if exposure_score < self.waste_thresholds["bad_exposure"]:
            reasons.append("bad_exposure")
            risk += 0.30

        if overall_score < self.waste_thresholds["very_low_score"]:
            reasons.append("very_low_score")
            risk += 0.50
        elif overall_score < self.waste_thresholds["mid_low_score"]:
            reasons.append("mid_low_score")
            risk += 0.20

        if person_count == 0 and overall_score < self.waste_thresholds["low_score_no_person"]:
            reasons.append("no_person_low_score")
            risk += 0.25

        if person_count > 0:
            risk -= 0.15

        if risk >= 0.80:
            return True, ",".join(reasons)
        return False, ",".join(reasons) if reasons else ""

    @staticmethod
    def _auto_classify(person_count: int, style_label: str, sharpness_score: float, is_waste: bool) -> str:
        if is_waste:
            return "waste"
        if person_count >= 2:
            return "portrait_group"
        if person_count == 1:
            return "portrait_single"
        if sharpness_score > 0.65:
            return f"detail_{style_label}"
        return f"scene_{style_label}"

    def _load_learning_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.score_weights.update(state.get("score_weights", {}))
            self.waste_thresholds.update(state.get("waste_thresholds", {}))
            self.feedback_stats.update(state.get("feedback_stats", {}))
            self.face_identities.update(state.get("face_identities", {}))
            self.image_person_map.update(state.get("image_person_map", {}))
            self.confirmed_mappings.update(state.get("confirmed_mappings", {}))
            raw_gallery = state.get("person_gallery", {})
            for k, v in raw_gallery.items():
                if isinstance(v, list):
                    self.person_gallery[k] = [list(e) for e in v if isinstance(e, list)]
            self.next_person_id = int(state.get("next_person_id", self.next_person_id))
        except Exception as ex:
            logger.warning("Failed to load learning state: %s", ex)

    def _ensure_weight_defaults(self) -> None:
        defaults = {
            "sharpness": 0.32,
            "exposure": 0.24,
            "object": 0.20,
            "person": 0.18,
            "style": 0.06,
        }
        if "face" in self.score_weights and "person" not in self.score_weights:
            self.score_weights["person"] = self.score_weights["face"]
        for key, val in defaults.items():
            self.score_weights.setdefault(key, val)

    def _save_learning_state(self) -> None:
        state = {
            "score_weights": self.score_weights,
            "waste_thresholds": self.waste_thresholds,
            "feedback_stats": self.feedback_stats,
            "face_identities": self.face_identities,
            "image_person_map": self.image_person_map,
            "confirmed_mappings": self.confirmed_mappings,
            "person_gallery": {k: [list(e) for e in v] for k, v in self.person_gallery.items()},
            "next_person_id": self.next_person_id,
        }
        try:
            self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as ex:
            logger.warning("Failed to save learning state: %s", ex)

    def _resolve_person_label(self, face_signature: list, person_count: int, phash: str, image_path: str = "") -> str:
        if not face_signature or person_count <= 0:
            if phash and phash in self.image_person_map:
                return self.image_person_map[phash]
            return "none"

        # Check confirmed mappings first (user-verified, highest priority)
        if phash and phash in self.confirmed_mappings:
            return self.confirmed_mappings[phash]

        # Check phash cache
        if phash:
            if phash in self.image_person_map:
                return self.image_person_map[phash]
            near = self._find_phash_match(phash)
            if near is not None:
                return self.image_person_map[near]

        try:
            query = np.asarray(face_signature, dtype=np.float32)
        except Exception:
            return "none"

        # Multi-gallery matching: compare against person_gallery (each person has multiple reference embeddings)
        best_label = ""
        best_sim = -1.0
        for label, embeddings in self.person_gallery.items():
            for emb in embeddings:
                try:
                    base = np.asarray(emb, dtype=np.float32)
                except Exception:
                    continue
                if base.size != query.size:
                    continue
                denom = (np.linalg.norm(base) * np.linalg.norm(query)) + 1e-8
                sim = float(np.dot(base, query) / denom)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

        # Fallback to single-vector face_identities
        if best_label == "" or best_sim < 0.5:
            for label, vector in self.face_identities.items():
                try:
                    base = np.asarray(vector, dtype=np.float32)
                except Exception:
                    continue
                if base.size != query.size:
                    continue
                denom = (np.linalg.norm(base) * np.linalg.norm(query)) + 1e-8
                sim = float(np.dot(base, query) / denom)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

        # Multi-frame confirmation: require consistent match across multiple images
        if best_label and best_sim >= 0.82:
            prev = np.asarray(self.face_identities.get(best_label, query), dtype=np.float32)
            merged = (prev * 0.75) + (query * 0.25)
            self.face_identities[best_label] = merged.tolist()
            if phash:
                self.image_person_map[phash] = best_label
            self._add_to_gallery(best_label, query.tolist())
            return best_label

        # High-confidence match with lower threshold for confirmed identities
        if best_label and best_sim >= 0.60 and best_label in self.confirmed_mappings.values():
            prev = np.asarray(self.face_identities.get(best_label, query), dtype=np.float32)
            merged = (prev * 0.75) + (query * 0.25)
            self.face_identities[best_label] = merged.tolist()
            if phash:
                self.image_person_map[phash] = best_label
            self._add_to_gallery(best_label, query.tolist())
            return best_label

        # No match: create new identity
        new_label = f"person_{self.next_person_id}"
        self.next_person_id += 1
        self.face_identities[new_label] = query.tolist()
        if phash:
            self.image_person_map[phash] = new_label
        self._add_to_gallery(new_label, query.tolist())
        return new_label

    def _add_to_gallery(self, label: str, embedding: list[float]) -> None:
        if label not in self.person_gallery:
            self.person_gallery[label] = []
        gallery = self.person_gallery[label]
        max_gallery = 5
        if len(gallery) >= max_gallery:
            gallery.pop(0)
        gallery.append(embedding)

    def rename_person(self, old_label: str, new_label: str) -> dict:
        old = (old_label or "").strip()
        new = (new_label or "").strip()
        if not new:
            return {"status": "ignored", "reason": "empty_new_label"}
        if not old or old == new:
            return {"status": "noop", "label": new}

        if old in self.face_identities:
            vector = self.face_identities.pop(old)
            gallery = self.person_gallery.pop(old, [])

            if new in self.face_identities:
                prev = np.asarray(self.face_identities[new], dtype=np.float32)
                incoming = np.asarray(vector, dtype=np.float32)
                merged = (prev * 0.6) + (incoming * 0.4)
                self.face_identities[new] = merged.tolist()
                # Merge galleries
                existing_gallery = self.person_gallery.get(new, [])
                self.person_gallery[new] = (existing_gallery + gallery)[:5]
                status = "merged"
            else:
                self.face_identities[new] = vector
                self.person_gallery[new] = gallery
                status = "renamed"

            if self.image_person_map:
                for key, val in list(self.image_person_map.items()):
                    if val == old:
                        self.image_person_map[key] = new
            if old in self.confirmed_mappings:
                del self.confirmed_mappings[old]
            self._save_learning_state()
            return {"status": status, "label": new}

        return {"status": "not_found", "label": new}

    def assign_person(self, image_path: str, new_label: str) -> dict:
        label = (new_label or "").strip()
        if not label:
            return {"status": "ignored", "reason": "empty_label"}

        # Mark as confirmed mapping
        image = cv2.imread(str(image_path))
        if image is None:
            return {"status": "error", "reason": "decode_failed"}

        phash = self._perceptual_hash(image)
        if phash:
            self.image_person_map[phash] = label
            self.confirmed_mappings[phash] = label

        face_signature: list[float] = []
        for plugin in self.plugins:
            if getattr(plugin, "name", "") == "yolo":
                try:
                    output = plugin.analyze(str(image_path), image)
                    face_signature = output.features.get("face_signature", [])
                except Exception:
                    face_signature = []
                break

        if face_signature:
            try:
                query = np.asarray(face_signature, dtype=np.float32)
                if label in self.face_identities:
                    prev = np.asarray(self.face_identities[label], dtype=np.float32)
                    merged = (prev * 0.7) + (query * 0.3)
                    self.face_identities[label] = merged.tolist()
                else:
                    self.face_identities[label] = query.tolist()
                self._add_to_gallery(label, query.tolist())
            except Exception:
                pass

        self._save_learning_state()
        return {"status": "ok", "label": label}

    @staticmethod
    def _perceptual_hash(image: np.ndarray) -> str:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized))
            block = dct[:8, :8]
            med = float(np.median(block))
            bits = (block > med).flatten().tolist()
            value = 0
            for bit in bits:
                value = (value << 1) | (1 if bit else 0)
            return f"{value:016x}"
        except Exception:
            return ""

    def _find_phash_match(self, phash: str, max_distance: int = 6) -> str | None:
        try:
            target = int(phash, 16)
        except Exception:
            return None

        best_key = None
        best_dist = max_distance + 1
        for key in self.image_person_map.keys():
            try:
                val = int(key, 16)
            except Exception:
                continue
            dist = (target ^ val).bit_count()
            if dist < best_dist:
                best_dist = dist
                best_key = key

        return best_key if best_key is not None and best_dist <= max_distance else None
