"""3D Perspective & Vanishing Point Geometry Inconsistency Extractor (Module 5).

Evaluates whether lines that should be parallel in 3D projective space converge
strictly to consistent 1-, 2-, or 3-point Vanishing Points (VPs).
Diffusion models generate objects patch-by-patch without enforcing projective 3D camera
matrices, resulting in scattered, inconsistent vanishing point intersections.
"""

from typing import Tuple, Union
import numpy as np
import torch
import cv2

from .base import BasePhysicsExtractor
from ..data.preprocessor import validate_and_load_image


class PerspectiveVPExtractor(BasePhysicsExtractor):
    """Computes Vanishing Point (VP) intersection dispersion and projective geometry violations.

    Outputs:
    - Delta_persp in R^3:
        1. dispersion_variance: Variance of normalized pairwise intersection coordinates
        2. cluster_scatter: Mean residual distance to dominant vanishing point clusters
        3. collinearity_divergence: Deviation of candidate vanishing points from a common horizon
    - Confidence c_persp in [0, 1]:
        Conditioned on the count and structural saliency of detected linear segments.
    """

    def __init__(self, min_line_length: float = 15.0, max_intersections: int = 200):
        super().__init__(feature_dim=3)
        self.min_line_length = min_line_length
        self.max_intersections = max_intersections
        # Fast Line Segment Detector
        self.lsd = cv2.createLineSegmentDetector(0)

    def _extract_lines(self, gray: np.ndarray) -> np.ndarray:
        """Detects salient line segments [x1, y1, x2, y2]."""
        lines = self.lsd.detect(gray)[0]
        if lines is None or len(lines) == 0:
            return np.empty((0, 4), dtype=np.float32)

        lines = lines.reshape(-1, 4)
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.sqrt(dx**2 + dy**2)
        valid = lengths >= self.min_line_length
        return lines[valid]

    def _compute_intersections(
        self, lines: np.ndarray, h: int, w: int
    ) -> Tuple[np.ndarray, float]:
        """Computes normalized pairwise intersection points (X, Y)."""
        num_lines = len(lines)
        if num_lines < 4:
            return np.empty((0, 2), dtype=np.float32), 0.0

        # Homogeneous representation of lines: ax + by + c = 0
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        norm = np.sqrt(a**2 + b**2) + 1e-7
        a, b, c = a / norm, b / norm, c / norm

        intersections = []
        step = max(1, num_lines // 25)
        for i in range(0, num_lines, step):
            for j in range(i + 1, min(num_lines, i + 35)):
                det = a[i] * b[j] - a[j] * b[i]
                if abs(det) > 0.15:  # Angle > ~8.5 degrees
                    ix = (b[i] * c[j] - b[j] * c[i]) / det
                    iy = (c[i] * a[j] - c[j] * a[i]) / det

                    norm_x = (ix - w / 2.0) / (w / 2.0)
                    norm_y = (iy - h / 2.0) / (h / 2.0)

                    if abs(norm_x) <= 5.0 and abs(norm_y) <= 5.0:
                        intersections.append([norm_x, norm_y])

                if len(intersections) >= self.max_intersections:
                    break
            if len(intersections) >= self.max_intersections:
                break

        pts = np.array(intersections, dtype=np.float32)
        confidence = float(np.clip(len(pts) / 40.0, 0.05, 1.0))
        return pts, confidence

    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        lines = self._extract_lines(gray)
        pts, confidence = self._compute_intersections(lines, h, w)

        if len(pts) < 6:
            delta = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float32)
            return delta, 0.02

        var_x = float(np.var(pts[:, 0]))
        var_y = float(np.var(pts[:, 1]))
        dispersion_var = float(np.clip((var_x + var_y) / 2.0, 0.0, 10.0))

        median_y = np.median(pts[:, 1])
        c1 = pts[pts[:, 1] <= median_y]
        c2 = pts[pts[:, 1] > median_y]

        scatter1 = np.mean(np.linalg.norm(c1 - np.median(c1, axis=0), axis=1)) if len(c1) > 0 else 0.0
        scatter2 = np.mean(np.linalg.norm(c2 - np.median(c2, axis=0), axis=1)) if len(c2) > 0 else 0.0
        cluster_scatter = float(np.clip((scatter1 + scatter2) / 2.0, 0.0, 5.0))

        if len(pts) >= 4:
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            poly = np.polyfit(x_vals, y_vals, deg=1)
            y_pred = np.polyval(poly, x_vals)
            collinearity_div = float(np.clip(np.mean(np.abs(y_vals - y_pred)), 0.0, 5.0))
        else:
            collinearity_div = 0.5

        delta = torch.tensor([dispersion_var, cluster_scatter, collinearity_div], dtype=torch.float32)
        return delta, confidence
