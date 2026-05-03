"""Photography exposure and tonal analysis plugin.

Evaluates exposure quality based on:
- Histogram distribution and clipping
- Exposure accuracy (properly exposed vs underexposed/overexposed)
- Tonal range (high-key, low-key, normal)
- Contrast evaluation

CPU-only, no ML model required.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.plugin_base import BasePlugin, PluginOutput


class ExposurePlugin(BasePlugin):
    name = "exposure"
    requires_gpu = False
    priority = 30
    version = "1.0.0"

    def analyze(self, image_path: str, image: np.ndarray | None = None) -> PluginOutput:
        if image is None:
            image = cv2.imread(image_path)
        if image is None:
            return PluginOutput(plugin_name=self.name, score=0.0, features={"status": "decode_failed"})

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1. Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total_pixels = gray.size

        # 2. Highlight and shadow clipping
        highlight_clipping = float(np.sum(hist[245:])) / total_pixels
        shadow_clipping = float(np.sum(hist[:10])) / total_pixels

        # 3. Exposure accuracy
        mean_brightness = float(np.mean(gray))
        exposure_deviation = abs(mean_brightness - 128.0) / 128.0
        exposure_accuracy = max(0.0, 1.0 - exposure_deviation)

        # 4. Histogram spread (dynamic range)
        cumulative = np.cumsum(hist) / total_pixels
        low_bound = int(np.searchsorted(cumulative, 0.05))
        high_bound = int(np.searchsorted(cumulative, 0.95))
        dynamic_range = (high_bound - low_bound) / 255.0

        # 5. Contrast via RMS contrast and histogram std
        contrast_std = float(np.std(gray)) / 255.0
        rms_contrast = float(np.sqrt(np.mean((gray.astype(float) - mean_brightness) ** 2))) / 255.0
        contrast_score = min(1.0, (contrast_std + rms_contrast) * 1.5)

        # 6. Tonal type classification
        shadows = float(np.sum(hist[:85])) / total_pixels
        midtones = float(np.sum(hist[85:171])) / total_pixels
        highlights = float(np.sum(hist[171:])) / total_pixels

        if shadows > 0.55 and highlights < 0.15:
            tone_type = "low_key"
            tone_bonus = 0.8  # Low-key can be artistic
        elif highlights > 0.55 and shadows < 0.15:
            tone_type = "high_key"
            tone_bonus = 0.8  # High-key can be artistic
        elif highlight_clipping > 0.08:
            tone_type = "overexposed"
            tone_bonus = 0.3
        elif shadow_clipping > 0.08:
            tone_type = "underexposed"
            tone_bonus = 0.3
        elif midtones > 0.6:
            tone_type = "balanced"
            tone_bonus = 0.9
        else:
            tone_type = "normal"
            tone_bonus = 0.75

        # 7. Overall exposure quality score
        clipping_penalty = min(1.0, (highlight_clipping + shadow_clipping) * 3.0)
        exposure_quality = (
            exposure_accuracy * 0.35
            + dynamic_range * 0.20
            + contrast_score * 0.15
            + tone_bonus * 0.30
        )
        exposure_quality = max(0.0, min(1.0, exposure_quality - clipping_penalty * 0.3))

        # 8. V channel analysis for saturation-aware exposure
        v_channel = hsv[:, :, 2]
        mean_v = float(np.mean(v_channel)) / 255.0
        std_v = float(np.std(v_channel)) / 255.0

        return PluginOutput(
            plugin_name=self.name,
            score=round(exposure_quality, 4),
            features={
                "exposure_quality": round(exposure_quality, 4),
                "histogram_brightness": round(mean_brightness, 2),
                "highlight_clipping": round(highlight_clipping, 4),
                "shadow_clipping": round(shadow_clipping, 4),
                "dynamic_range": round(dynamic_range, 4),
                "contrast_score": round(contrast_score, 4),
                "tone_type": tone_type,
                "exposure_accuracy": round(exposure_accuracy, 4),
                "mean_v": round(mean_v, 4),
                "std_v": round(std_v, 4),
            },
        )
