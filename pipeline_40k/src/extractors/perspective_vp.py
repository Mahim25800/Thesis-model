"""Confidence-aware image-space projective geometry descriptors.

The extractor does not assert that a single photograph has one vanishing point:
ordinary scenes can contain several 3-D directions and curved objects. It
instead measures whether long, observable line families form stable projective
hypotheses. Low-line or texture-only images receive low confidence so this
entity cannot turn missing geometry into evidence.
"""

from __future__ import annotations

from typing import Tuple, Union

import cv2
import numpy as np
import torch

from .base import BasePhysicsExtractor
from ..data.preprocessor import validate_and_load_image


GEOMETRY_DIM = 8


class PerspectiveVPExtractor(BasePhysicsExtractor):
    """Describe line-family, vanishing-point, and cross-scale consistency.

    Features are bounded summaries, not direct fake scores. They are learned
    only in combination with the other physics entities and are confidence
    gated when the image does not contain enough straight structure.
    """

    def __init__(self, min_line_length: float = 18.0, max_lines: int = 160):
        super().__init__(feature_dim=GEOMETRY_DIM)
        if min_line_length <= 0 or max_lines < 12:
            raise ValueError("Projective geometry settings are invalid")
        self.min_line_length = float(min_line_length)
        self.max_lines = int(max_lines)
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def _lines(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        detected = self.lsd.detect(gray)[0]
        if detected is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        lines = detected.reshape(-1, 4).astype(np.float32)
        dx, dy = lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]
        lengths = np.hypot(dx, dy)
        keep = lengths >= self.min_line_length
        lines, lengths = lines[keep], lengths[keep]
        if not len(lines):
            return lines, lengths, np.empty(0, dtype=np.float32)
        order = np.argsort(lengths)[::-1][: self.max_lines]
        lines, lengths = lines[order], lengths[order]
        angles = np.mod(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]), np.pi)
        return lines, lengths, angles.astype(np.float32)

    @staticmethod
    def _orientation_distribution(angles: np.ndarray, lengths: np.ndarray, bins: int = 12) -> np.ndarray:
        if not len(angles):
            return np.full(bins, 1.0 / bins, dtype=np.float32)
        values, _ = np.histogram(angles, bins=bins, range=(0.0, np.pi), weights=lengths)
        return (values.astype(np.float32) + 1e-4) / (values.sum() + 1e-4 * bins)

    @staticmethod
    def _vp_summary(lines: np.ndarray, lengths: np.ndarray, height: int, width: int) -> tuple[float, float]:
        """Return dominant projective-hypothesis support and compactness."""
        if len(lines) < 6:
            return 0.0, 0.0
        x1, y1, x2, y2 = (lines[:, index] for index in range(4))
        a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
        norm = np.hypot(a, b) + 1e-7
        a, b, c = a / norm, b / norm, c / norm
        points, weights = [], []
        for left in range(len(lines)):
            for right in range(left + 1, len(lines)):
                determinant = a[left] * b[right] - a[right] * b[left]
                if abs(determinant) < 0.08:
                    continue
                px = (b[left] * c[right] - b[right] * c[left]) / determinant
                py = (c[left] * a[right] - c[right] * a[left]) / determinant
                nx, ny = (px - width / 2) / (width / 2), (py - height / 2) / (height / 2)
                if abs(nx) <= 4.0 and abs(ny) <= 4.0:
                    points.append((nx, ny))
                    weights.append(float(min(lengths[left], lengths[right])))
        if len(points) < 8:
            return 0.0, 0.0
        pts, weight_values = np.asarray(points), np.asarray(weights)
        grid = np.floor((pts + 4.0) * 2.0).astype(int)
        keys, inverse = np.unique(grid, axis=0, return_inverse=True)
        support = np.bincount(inverse, weights=weight_values, minlength=len(keys))
        best = int(np.argmax(support))
        selected = inverse == best
        share = float(support[best] / max(float(support.sum()), 1e-6))
        center = np.average(pts[selected], axis=0, weights=weight_values[selected])
        distance = np.linalg.norm(pts[selected] - center, axis=1)
        compactness = float(np.exp(-np.average(distance, weights=weight_values[selected])))
        return float(np.clip(share, 0.0, 1.0)), float(np.clip(compactness, 0.0, 1.0))

    def extract(self, image: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)
        height, width = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        lines, lengths, angles = self._lines(gray)
        if len(lines) < 6:
            return torch.zeros(GEOMETRY_DIM, dtype=torch.float32), 0.02

        diagonal = float(np.hypot(height, width))
        orientation = self._orientation_distribution(angles, lengths)
        entropy = float(-(orientation * np.log(orientation)).sum() / np.log(len(orientation)))
        dominant_orientation = float(orientation.max())
        line_density = float(np.clip(np.log1p(len(lines)) / np.log1p(self.max_lines), 0.0, 1.0))
        mean_length = float(np.clip(np.average(lengths) / max(diagonal, 1.0), 0.0, 1.0))
        coverage = float(np.clip(lengths.sum() / max(4.0 * diagonal, 1.0), 0.0, 1.0))
        vp_support, vp_compactness = self._vp_summary(lines, lengths, height, width)

        small = cv2.resize(gray, (max(16, width // 2), max(16, height // 2)), interpolation=cv2.INTER_AREA)
        _, small_lengths, small_angles = self._lines(small)
        small_orientation = self._orientation_distribution(small_angles, small_lengths)
        midpoint = 0.5 * (orientation + small_orientation)
        js_divergence = 0.5 * (
            np.sum(orientation * np.log(orientation / midpoint))
            + np.sum(small_orientation * np.log(small_orientation / midpoint))
        )
        cross_scale_agreement = float(np.clip(1.0 - js_divergence / np.log(2.0), 0.0, 1.0))

        features = torch.tensor([
            line_density, mean_length, coverage, entropy, dominant_orientation,
            vp_support, vp_compactness, cross_scale_agreement,
        ], dtype=torch.float32)
        evidence = min(1.0, len(lines) / 48.0)
        confidence = float(np.clip(evidence * (0.35 + 0.65 * max(vp_support, dominant_orientation)), 0.02, 1.0))
        return features, confidence
