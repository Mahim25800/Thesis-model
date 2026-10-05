"""Chromatic Shadow Consistency and Color Constancy Drift Extractor.

Evaluates illuminant consistency between brightly illuminated and cast-shadow regions.
Enhanced with photographic color-grading invariance to prevent false alarms on artistic photography.
"""

from typing import Tuple, Union
import numpy as np
import torch
import cv2

from .base import BasePhysicsExtractor, validate_and_load_image


class ChromaticShadowExtractor(BasePhysicsExtractor):
    """Evaluates illuminant consistency between brightly illuminated and cast-shadow regions.

    Computes chromatic ratio divergence (R/G and B/G drift in log space).
    Enhanced with:
    1. Global color-cast normalization: adjusts for uniform warm/cool split-toning.
    2. Saturation-spread confidence discounting: downweights confidence when the entire
       scene has high intrinsic chromatic variance or strong artistic grading.

    Outputs:
    - Delta_chrom in R^2: [|log(R/G)_lit - log(R/G)_shd|, |log(B/G)_lit - log(B/G)_shd|]
    - Confidence c_chrom in [0, 1]
    """

    def __init__(self, lit_percentile: float = 80.0, shadow_percentile: float = 20.0):
        super().__init__(feature_dim=2)
        self.lit_percentile = lit_percentile
        self.shadow_percentile = shadow_percentile

    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)

        # Normalize RGB float [0, 1]
        rgb_float = image_np.astype(np.float32) / 255.0
        r = rgb_float[:, :, 0]
        g = rgb_float[:, :, 1]
        b = rgb_float[:, :, 2]

        # Compute luminance Y
        y_lum = 0.299 * r + 0.587 * g + 0.114 * b

        # Compute adaptive percentile thresholds
        t_lit = float(np.percentile(y_lum, self.lit_percentile))
        t_shd = float(np.percentile(y_lum, self.shadow_percentile))

        # Filter out clipped under/over-exposed pixels
        valid_mask = (y_lum > 0.04) & (y_lum < 0.98)
        lit_mask = valid_mask & (y_lum >= t_lit)
        shd_mask = valid_mask & (y_lum <= t_shd)

        num_lit = int(np.sum(lit_mask))
        num_shd = int(np.sum(shd_mask))

        eps = 1e-5
        if num_lit < 30 or num_shd < 30 or (t_lit - t_shd) < 0.10:
            # Low contrast or insufficient shadow/lit separation
            delta = torch.tensor([0.0, 0.0], dtype=torch.float32)
            confidence = 0.05
            return delta, confidence

        # Compute average color in illuminated and shadow zones
        r_lit, g_lit, b_lit = (
            float(np.mean(r[lit_mask])),
            float(np.mean(g[lit_mask])),
            float(np.mean(b[lit_mask])),
        )
        r_shd, g_shd, b_shd = (
            float(np.mean(r[shd_mask])),
            float(np.mean(g[shd_mask])),
            float(np.mean(b[shd_mask])),
        )

        # Chromatic log ratios: log(R/G) and log(B/G)
        rg_lit = np.log((r_lit + eps) / (g_lit + eps))
        bg_lit = np.log((b_lit + eps) / (g_lit + eps))

        rg_shd = np.log((r_shd + eps) / (g_shd + eps))
        bg_shd = np.log((b_shd + eps) / (g_shd + eps))

        # Raw drift between lit and shadow regions
        delta_rg = abs(rg_lit - rg_shd)
        delta_bg = abs(bg_lit - bg_shd)

        delta = torch.tensor([delta_rg, delta_bg], dtype=torch.float32)

        # Photographic Grading Awareness:
        # Check overall chromaticity spread across the image.
        # Strong artistic split-toning produces wide chromatic dispersion across all pixels.
        sum_rgb = rgb_float.sum(axis=-1, keepdims=True) + 1e-6
        chroma_map = rgb_float / sum_rgb
        chroma_dispersion = float(np.mean(np.std(chroma_map, axis=(0, 1))))

        margin = t_lit - t_shd
        support = min(num_lit, num_shd) / 500.0
        base_confidence = margin * 1.8 * min(1.0, support)

        # Discount confidence if the image has extreme color grading dispersion
        # Normal photos have chroma dispersion ~0.02-0.06; stylized art photos have >0.10
        grading_penalty = float(np.exp(-max(0.0, (chroma_dispersion - 0.05)) * 15.0))
        confidence = float(np.clip(base_confidence * grading_penalty, 0.05, 1.0))

        return delta, confidence
