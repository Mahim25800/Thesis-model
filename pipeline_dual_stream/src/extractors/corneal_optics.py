"""Specular Reflection Inconsistency Extractor (Corneal & Generalized Specular Highlights).

Generalizes beyond human iris glints to all specular materials (glass, metal, wet roads, eyes).
Computes the high-frequency specular highlight masks across the top two reflective regions
and measures the angular and displacement delta of their apparent directional centers.
"""

from typing import Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import BasePhysicsExtractor, validate_and_load_image


def differential_soft_argmax2d(
    heatmap: torch.Tensor, temperature: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes differential soft-argmax 2D coordinates [0, 1]."""
    h, w = heatmap.shape
    flat = heatmap.view(-1) / max(temperature, 1e-4)
    weights = F.softmax(flat, dim=0).view(h, w)

    y_coords = torch.linspace(0.0, 1.0, steps=h, device=heatmap.device)
    x_coords = torch.linspace(0.0, 1.0, steps=w, device=heatmap.device)

    wy = weights.sum(dim=1)
    wx = weights.sum(dim=0)

    cy = torch.sum(wy * y_coords)
    cx = torch.sum(wx * x_coords)

    return cx, cy


class CornealOpticsExtractor(BasePhysicsExtractor):
    """Specular Reflection Inconsistency Extractor.

    Evaluates consistency across top two specular regions (eyes or general specular highlights).
    Outputs:
    - Delta_cornea in R^4:
        1. Relative horizontal centroid displacement: |cx_1 - cx_2|
        2. Relative vertical centroid displacement: |cy_1 - cy_2|
        3. Geometric angular delta: |theta_1 - theta_2| / pi
        4. Specular reflection profile discrepancy: 1 - profile_correlation
    - Confidence c_cornea in [0, 1]
    """

    def __init__(
        self,
        min_resolution: int = 24,
        temperature: float = 0.05,
        region_mode: str = "general",
    ):
        super().__init__(feature_dim=4)
        self.min_resolution = min_resolution
        self.temperature = temperature
        self.region_mode = region_mode

    def _extract_glint_stats(
        self, crop: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return torch.tensor(0.5), torch.tensor(0.5), torch.zeros(16), 0.0

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = crop.astype(np.float32)
            if float(np.max(gray)) > 1.0:
                gray /= 255.0

        max_val = float(np.max(gray))
        med_val = float(np.median(gray))
        prominence = max(0.0, max_val - med_val)

        # Enhance specular peaks
        specular_map = np.power(gray / (max_val + 1e-6), 3)

        t_map = torch.from_numpy(specular_map).float()
        cx, cy = differential_soft_argmax2d(t_map, temperature=self.temperature)

        # Extract 16-d reflection radial profile around centroid
        center_x = int(float(cx) * (w - 1))
        center_y = int(float(cy) * (h - 1))
        half_k = max(4, min(h, w) // 4)
        y1, y2 = max(0, center_y - half_k), min(h, center_y + half_k)
        x1, x2 = max(0, center_x - half_k), min(w, center_x + half_k)
        patch = gray[y1:y2, x1:x2]
        if patch.size > 0 and patch.shape[0] >= 2 and patch.shape[1] >= 2:
            patch_resized = cv2.resize(patch, (4, 4), interpolation=cv2.INTER_AREA).flatten()
            profile = torch.from_numpy(patch_resized).float()
            profile_norm = torch.norm(profile) + 1e-6
            profile = profile / profile_norm
        else:
            profile = torch.zeros(16)

        return cx, cy, profile, prominence

    def _find_top_specular_regions(
        self, gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        h, w = gray.shape[:2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        base = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        specular_residual = np.maximum(0.0, gray - base)

        thresh = float(np.percentile(specular_residual, 98.0))
        mask = (specular_residual >= max(thresh, 0.05)).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 2:
            mid_w = w // 2
            return gray[:, :mid_w], gray[:, mid_w:], float(np.max(specular_residual))

        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)

        crops = []
        for idx, _ in areas[:2]:
            x = stats[idx, cv2.CC_STAT_LEFT]
            y = stats[idx, cv2.CC_STAT_TOP]
            bw = stats[idx, cv2.CC_STAT_WIDTH]
            bh = stats[idx, cv2.CC_STAT_HEIGHT]
            pad = max(8, max(bw, bh) // 2)
            y1, y2 = max(0, y - pad), min(h, y + bh + pad)
            x1, x2 = max(0, x - pad), min(w, x + bw + pad)
            crops.append(gray[y1:y2, x1:x2])

        if len(crops) < 2:
            mid_w = w // 2
            return gray[:, :mid_w], gray[:, mid_w:], float(np.max(specular_residual))

        prominence = float(np.max(specular_residual))
        return crops[0], crops[1], prominence

    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        crop1, crop2, prominence = self._find_top_specular_regions(gray)
        cx1, cy1, prof1, _ = self._extract_glint_stats(crop1)
        cx2, cy2, prof2, _ = self._extract_glint_stats(crop2)

        # 1. Delta x displacement
        delta_x = torch.abs(cx1 - cx2)

        # 2. Delta y displacement
        delta_y = torch.abs(cy1 - cy2)

        # 3. Geometric angular delta from respective region centers (0.5, 0.5)
        angle1 = torch.atan2(cy1 - 0.5, cx1 - 0.5)
        angle2 = torch.atan2(cy2 - 0.5, cx2 - 0.5)
        delta_angle = torch.abs(angle1 - angle2) / torch.pi
        if delta_angle > 1.0:
            delta_angle = 2.0 - delta_angle

        # 4. Profile similarity discrepancy (1 - correlation)
        sim = torch.dot(prof1, prof2) / (
            (torch.norm(prof1) * torch.norm(prof2)) + 1e-6
        )
        delta_profile = torch.clamp(1.0 - sim, 0.0, 2.0)

        delta = torch.stack([delta_x, delta_y, delta_angle, delta_profile]).float()
        confidence = float(np.clip(prominence * 1.8, 0.05, 1.0))
        return delta, confidence
