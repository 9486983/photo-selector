from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PluginConfig:
    enabled: bool = True
    model_path: str | None = None
    device: str = "auto"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    gpu_enabled: bool = True
    cpu_fallback: bool = True
    batch_size: int = 12
    gpu_workers: int = 1
    cpu_workers: int = 6
    batch_parallelism: int = 8
    score_weights: dict[str, float] = field(default_factory=dict)
    waste_thresholds: dict[str, float] = field(default_factory=dict)
    plugins: dict[str, PluginConfig] = field(default_factory=dict)


_DEFAULTS = {
    "score_weights": {
        "sharpness": 0.32,
        "exposure": 0.24,
        "object": 0.20,
        "person": 0.18,
        "style": 0.06,
    },
    "waste_thresholds": {
        "hard_blur": 0.018,
        "soft_blur": 0.045,
        "bad_exposure": 0.12,
        "very_low_score": 0.10,
        "low_score_no_person": 0.15,
        "mid_low_score": 0.22,
    },
}


def _parse_plugins(raw: dict[str, Any] | None) -> dict[str, PluginConfig]:
    result: dict[str, PluginConfig] = {}
    for name, cfg in (raw or {}).items():
        if not isinstance(cfg, dict):
            result[name] = PluginConfig()
            continue
        result[name] = PluginConfig(
            enabled=bool(cfg.get("enabled", True)),
            model_path=str(cfg["model_path"]) if "model_path" in cfg else None,
            device=str(cfg.get("device", "auto")),
            options={k: v for k, v in cfg.items() if k not in ("enabled", "model_path", "device")},
        )
    return result


def load_config(path: str | Path | None = None) -> EngineConfig:
    raw: dict[str, Any] = {}
    search = Path(path) if path else Path(__file__).resolve().parent.parent / "appsettings.json"
    if search.exists():
        raw = json.loads(search.read_text(encoding="utf-8-sig"))

    engine_raw = raw.get("engine", {})

    sw: dict[str, float] = {}
    for k, v in _DEFAULTS["score_weights"].items():
        sw[k] = float(engine_raw.get("score_weights", {}).get(k, v))

    wt: dict[str, float] = {}
    for k, v in _DEFAULTS["waste_thresholds"].items():
        wt[k] = float(engine_raw.get("waste_thresholds", {}).get(k, v))

    return EngineConfig(
        gpu_enabled=bool(engine_raw.get("gpu_enabled", True)),
        cpu_fallback=bool(engine_raw.get("cpu_fallback", True)),
        batch_size=int(engine_raw.get("batch_size", 12)),
        gpu_workers=int(engine_raw.get("gpu_workers", 1)),
        cpu_workers=int(engine_raw.get("cpu_workers", 6)),
        batch_parallelism=int(engine_raw.get("batch_parallelism", 8)),
        score_weights=sw,
        waste_thresholds=wt,
        plugins=_parse_plugins(raw.get("plugins")),
    )
