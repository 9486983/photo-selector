"""Photography composition analysis plugin.

Evaluates composition quality based on:
- Rule of thirds: subject placement near third-lines
- Symmetry: horizontal/vertical mirror similarity
- Leading lines: edge lines that guide the eye
- Simplicity: subject-to-background complexity ratio

CPU-only, no ML model required.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.plugin_base import BasePlugin, PluginOutput


class CompositionPlugin(BasePlugin):
    name = "composition"
    requires_gpu = False
    priority = 25
    version = "1.0.0"

    def analyze(self, image_path: str, image: np.ndarray | None = None) -> PluginOutput:
        if image is None:
            image = cv2.imread(image_path)
        if image is None:
            return PluginOutput(plugin_name=self.name, score=0.0, features={"status": "decode_failed"})

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Rule of thirds analysis
        thirds_score, third_balance = self._rule_of_thirds(image, gray, w, h)

        # 2. Symmetry analysis
        symmetry_score = self._symmetry_analysis(gray, w, h)

        # 3. Leading lines analysis
        leading_lines_score, has_leading_lines = self._leading_lines(gray)

        # 4. Simplicity / clutter analysis
        simplicity_score = self._simplicity_analysis(image, gray, w, h)

        # Determine composition type
        composition_type = "center"
        if symmetry_score > 0.6:
            composition_type = "symmetry"
        elif third_balance > 0.5:
            composition_type = "rule_of_thirds"
        elif has_leading_lines:
            composition_type = "leading_lines"

        # Weighted composition score
        composition_score = (
            thirds_score * 0.35
            + symmetry_score * 0.20
            + leading_lines_score * 0.20
            + simplicity_score * 0.25
        )
        composition_score = max(0.0, min(1.0, composition_score))

        return PluginOutput(
            plugin_name=self.name,
            score=round(composition_score, 4),
            features={
                "composition_score": round(composition_score, 4),
                "rule_of_thirds": round(third_balance, 4),
                "symmetry_score": round(symmetry_score, 4),
                "has_leading_lines": has_leading_lines,
                "leading_lines_score": round(leading_lines_score, 4),
                "simplicity_score": round(simplicity_score, 4),
                "composition_type": composition_type,
            },
        )

    @staticmethod
    def _rule_of_thirds(image: np.ndarray, gray: np.ndarray, w: int, h: int) -> tuple[float, float]:
        """Evaluate how well the subject aligns with rule-of-thirds grid lines."""
        third_x = w // 3
        third_y = h // 3

        # Use edge detection + saliency to find main subject regions
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((third_y // 4, third_x // 4), np.uint8)
        density = cv2.dilate(edges, kernel, iterations=1)

        # Define 3x3 grid cells
        grid = np.zeros((3, 3), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                x1 = int(col * third_x)
                y1 = int(row * third_y)
                x2 = int((col + 1) * third_x) if col < 2 else w
                y2 = int((row + 1) * third_y) if row < 2 else h
                cell = density[y1:y2, x1:x2]
                grid[row, col] = float(np.mean(cell))

        # Check if edge density concentrates near intersection points
        intersection_weight = np.zeros((3, 3), dtype=np.float32)
        # Weight intersections at (0,0), (0,2), (2,0), (2,2) for quadrants
        # and (0,1), (1,0), (1,2), (2,1) for third-lines
        intersection_weight[0, 0] = 0.15
        intersection_weight[0, 2] = 0.15
        intersection_weight[2, 0] = 0.15
        intersection_weight[2, 2] = 0.15
        # Center is less desirable for rule-of-thirds
        intersection_weight[1, 1] = -0.20

        third_balance = float(np.sum(grid * intersection_weight))
        third_balance = max(0.0, min(1.0, (third_balance + 0.2) / 0.4))

        subject_score = third_balance
        return subject_score, third_balance

    @staticmethod
    def _symmetry_analysis(gray: np.ndarray, w: int, h: int) -> float:
        """Check horizontal and vertical symmetry."""
        mid_x = w // 2
        mid_y = h // 2

        # Horizontal symmetry (left-right mirror)
        left = gray[:, :mid_x]
        right = gray[:, w - mid_x:]
        right_flipped = cv2.flip(right, 1)
        h_sym = 0.0
        if left.shape == right_flipped.shape:
            diff = cv2.absdiff(left, right_flipped)
            h_sym = 1.0 - float(np.mean(diff)) / 255.0

        # Vertical symmetry (top-bottom mirror)
        top = gray[:mid_y, :]
        bottom = gray[h - mid_y :, :]
        bottom_flipped = cv2.flip(bottom, 0)
        v_sym = 0.0
        if top.shape == bottom_flipped.shape:
            diff = cv2.absdiff(top, bottom_flipped)
            v_sym = 1.0 - float(np.mean(diff)) / 255.0

        return float(np.clip(max(h_sym, v_sym), 0.0, 1.0))

    @staticmethod
    def _leading_lines(gray: np.ndarray) -> tuple[float, bool]:
        """Detect prominent lines that could serve as leading lines."""
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min(gray.shape[1], gray.shape[0]) // 4,
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            return 0.0, False

        h, w = gray.shape
        cx, cy = w / 2, h / 2
        score = 0.0
        significant_lines = 0

        for line in lines[:20]:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < min(w, h) * 0.15:
                continue

            # Lines pointing toward center or going from edges inward score higher
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            dist_to_center = np.sqrt((mid_x - cx) ** 2 + (mid_y - cy) ** 2)

            # Check if line starts near edge and points inward
            at_edge = (
                x1 < w * 0.1
                or x2 < w * 0.1
                or x1 > w * 0.9
                or x2 > w * 0.9
                or y1 < h * 0.1
                or y2 < h * 0.1
                or y1 > h * 0.9
                or y2 > h * 0.9
            )

            # Diagonal lines are more dynamic
            angle = abs(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
            is_diagonal = 20 < angle < 70

            line_score = min(1.0, length / max(w, h)) * 1.5
            if at_edge:
                line_score *= 1.3
            if is_diagonal:
                line_score *= 1.2
            if dist_to_center < max(w, h) * 0.3:
                line_score *= 1.2

            score += line_score
            significant_lines += 1

        score = min(1.0, score / max(1, significant_lines))
        return score, significant_lines >= 2

    @staticmethod
    def _simplicity_analysis(image: np.ndarray, gray: np.ndarray, w: int, h: int) -> float:
        """Evaluate image simplicity vs clutter using edge density and color complexity."""
        # Edge density as a proxy for complexity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        # Color complexity via channel variance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        color_std = float(np.mean([np.std(lab[:, :, c]) for c in range(3)])) / 128.0

        # Texture complexity via local binary pattern variance
        texture_var = float(np.var(gray.astype(np.float32))) / (255.0 ** 2)

        # Simpler images have fewer edges, lower color variance, lower texture
        simplicity = 1.0 - (edge_density * 0.4 + color_std * 0.3 + texture_var * 0.3)
        return float(np.clip(simplicity, 0.0, 1.0))
