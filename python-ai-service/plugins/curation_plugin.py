from __future__ import annotations

import cv2
import numpy as np

from core.plugin_base import BasePlugin, PluginOutput


class CurationPlugin(BasePlugin):
    name = "curation"
    requires_gpu = False
    priority = 20

    def analyze(self, image_path: str, image: np.ndarray | None = None) -> PluginOutput:
        if image is None:
            image = cv2.imread(image_path)
        if image is None:
            return PluginOutput(plugin_name=self.name, score=0.0, features={"status": "decode_failed"})

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))

        if mean_s < 30:
            style_label = "low_saturation"
        elif 10 <= mean_h <= 42:
            style_label = "warm"
        elif 90 <= mean_h <= 140:
            style_label = "cool"
        else:
            style_label = "neutral"

        tone_label = "bright" if mean_v > 172 else "dark" if mean_v < 70 else "normal"
        color_label, dominant_colors = self._extract_dominant_colors(image)

        return PluginOutput(
            plugin_name=self.name,
            score=0.0,
            features={
                "style_label": style_label,
                "tone_label": tone_label,
                "color_label": color_label,
                "dominant_colors": dominant_colors,
                "mean_saturation": round(mean_s, 2),
                "mean_brightness": round(mean_v, 2),
            },
        )

    @staticmethod
    def _extract_dominant_colors(image: np.ndarray) -> tuple[str, list[str]]:
        small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3).astype(np.float32)
        k = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(counts)[::-1]
        colors = []
        for idx in order:
            b, g, r = centers[idx]
            colors.append(f"#{int(r):02X}{int(g):02X}{int(b):02X}")

        def is_neutral(hsv_val: np.ndarray) -> bool:
            s = int(hsv_val[1])
            v = int(hsv_val[2])
            return s < 30 or v < 35 or v > 220

        def label_from_hsv(hsv_val: np.ndarray) -> str:
            hue = int(hsv_val[0])
            sat = int(hsv_val[1])
            val = int(hsv_val[2])
            if sat < 26:
                return "gray"
            if val < 35:
                return "black"
            if hue < 10 or hue >= 170:
                return "red"
            if hue < 22:
                return "orange"
            if hue < 33:
                return "yellow"
            if hue < 78:
                return "green"
            if hue < 95:
                return "cyan"
            if hue < 131:
                return "blue"
            if hue < 160:
                return "purple"
            return "magenta"

        hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        v = hsv_small[:, :, 2]
        shadow_mask = v < 70
        mid_mask = (v >= 70) & (v < 180)
        highlight_mask = v >= 180

        def dominant_from_mask(mask: np.ndarray) -> np.ndarray | None:
            if mask.sum() < 20:
                return None
            masked = small[mask]
            if masked.size == 0:
                return None
            masked = masked.reshape(-1, 3).astype(np.float32)
            kk = 3 if masked.shape[0] > 50 else 1
            _, m_labels, m_centers = cv2.kmeans(masked, kk, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(m_labels.flatten(), minlength=kk)
            idx = int(np.argmax(counts))
            return m_centers[idx].astype(np.uint8)

        def best_non_neutral(candidates: list[np.ndarray | None]) -> np.ndarray | None:
            for cand in candidates:
                if cand is None:
                    continue
                hsv = cv2.cvtColor(np.array([[cand]]), cv2.COLOR_BGR2HSV)[0][0]
                if not is_neutral(hsv):
                    return cand
            return None

        highlight = dominant_from_mask(highlight_mask)
        midtone = dominant_from_mask(mid_mask)
        shadow = dominant_from_mask(shadow_mask)

        chosen = best_non_neutral([midtone, highlight, shadow])
        if chosen is None:
            chosen = centers[order[0]].astype(np.uint8)
            for idx in order:
                cand = centers[idx].astype(np.uint8)
                hsv = cv2.cvtColor(np.array([[cand]]), cv2.COLOR_BGR2HSV)[0][0]
                if not is_neutral(hsv):
                    chosen = cand
                    break

        hsv = cv2.cvtColor(np.array([[chosen]]), cv2.COLOR_BGR2HSV)[0][0]
        label = label_from_hsv(hsv)
        return label, colors
