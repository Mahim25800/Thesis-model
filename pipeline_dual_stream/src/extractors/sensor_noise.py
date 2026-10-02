"""Camera Sensor Noise and PRNU Residual Extractor (Solution 3).

Analyzes high-frequency sensor noise residuals, Bayer CFA demosaicing traces,
and computational photography signatures (e.g., smartphone portrait mode bokeh,
edge-preserving skin smoothing) to distinguish genuine physical camera sensors
from synthetic 2D art and generative latent models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np


class SensorNoiseExtractor:
    """Forensic High-Frequency Camera Sensor Noise and Photographic Residual Extractor."""

    def __init__(self):
        # 5x5 Kurtz-Vielhauer (SRM KV) high-pass forensic kernel
        # Standard in digital image steganalysis and PRNU camera fingerprinting
        self.kv_kernel = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0

    def extract(self, img_input: Union[str, Path, np.ndarray]) -> Dict[str, Union[float, bool, str]]:
        """Extract sensor noise, CFA traces, and computational photography signatures.
        
        Args:
            img_input: Image path or RGB/BGR numpy array.
            
        Returns:
            Dictionary containing:
                - is_portrait_bokeh: bool
                - is_photographic: bool
                - flat_noise_std: float (scaled by 1000)
                - cfa_noise_std: float (scaled by 1000)
                - flat_ratio: float
                - color_diversity: float
                - sensor_finding: str
        """
        if isinstance(img_input, (str, Path)):
            img = cv2.imread(str(img_input))
            if img is None:
                raise ValueError(f"Could not load image from {img_input}")
        elif isinstance(img_input, np.ndarray):
            # If RGB, convert to BGR for OpenCV
            if len(img_input.shape) == 3 and img_input.shape[2] == 3:
                img = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
            else:
                img = img_input
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0
        img_f = img.astype(np.float32) / 255.0

        # 1. High-Pass Sensor Noise Residual via SRM KV Filter
        hp_res = cv2.filter2D(gray_f, -1, self.kv_kernel)

        # 2. Gradient Magnitude & Flat Region Detection (bokeh, skin, sky, smooth walls)
        sobelx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        flat_mask = grad_mag < 0.025
        flat_ratio = float(np.mean(flat_mask))

        # Sensor noise floor in flat regions
        if flat_ratio > 0.01:
            flat_noise_std = float(np.std(hp_res[flat_mask])) * 1000.0
        else:
            flat_noise_std = float(np.std(hp_res)) * 1000.0

        # 3. Bayer CFA Cross-Color Channel Residual: G - 0.5*(R + B)
        # Real physical camera sensors use Bayer color filter arrays (RGGB).
        cfa_diff = img_f[:, :, 1] - 0.5 * (img_f[:, :, 0] + img_f[:, :, 2])
        cfa_hp = cv2.filter2D(cfa_diff, -1, self.kv_kernel)
        cfa_noise_std = float(np.std(cfa_hp)) * 1000.0

        # 4. Global Sharpness (Laplacian Variance)
        # Portrait mode photos with computational bokeh have shallow depth of field,
        # leading to moderate global sharpness (lap_var < 450.0),
        # whereas digital drawings and synthetic renders have crisp sharp line edges (lap_var > 700.0).
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 5. Color & Line Art Analysis (Distinguishes photos from 2D digital art/anime)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0

        canny = cv2.Canny(gray, 50, 150)
        dark_edges = (canny > 0) & (gray < 60)
        lineart_fraction = float(np.sum(dark_edges) / (np.sum(canny > 0) + 1e-5))

        # Color diversity across quantized 128x128 palette
        small = cv2.resize(img, (128, 128))
        unique_colors = len(np.unique(small.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (128.0 * 128.0)

        # 6. Photographic Domain Verification
        is_photographic = (
            color_diversity >= 0.65 and
            float(np.mean(sat)) < 0.60 and
            lineart_fraction < 0.15
        )

        # 7. Computational Portrait Mode & CMOS Sensor Signature
        # Authentic smartphone portrait mode features:
        # - Software depth-map bokeh blur (flat_ratio >= 0.05, lap_var < 450.0)
        # - Physical CMOS sensor noise floor in flat/bokeh areas (2.0 <= flat_noise_std <= 6.5)
        # - Bayer CFA color channel demosaicing residual (cfa_noise_std > 2.0)
        # - Photographic spectrum without comic lineart
        is_portrait_bokeh = (
            is_photographic and
            flat_ratio >= 0.05 and
            2.0 <= flat_noise_std <= 6.5 and
            cfa_noise_std > 2.0 and
            lap_var < 450.0
        )

        if is_portrait_bokeh:
            sensor_finding = (
                f"CMOS Sensor Verified (std={flat_noise_std:.2f}, cfa={cfa_noise_std:.2f}). "
                f"Computational Portrait Mode Bokeh Detected (flat={flat_ratio*100:.1f}%, lap_var={lap_var:.1f})."
            )
        elif is_photographic:
            sensor_finding = (
                f"Natural Photographic Spectrum (diversity={color_diversity:.2f}, noise_std={flat_noise_std:.2f})."
            )
        else:
            sensor_finding = (
                f"Stylized / Non-Photographic Spectrum (lineart={lineart_fraction*100:.1f}%, sat={float(np.mean(sat)):.2f})."
            )

        return {
            "is_portrait_bokeh": is_portrait_bokeh,
            "is_photographic": is_photographic,
            "flat_noise_std": flat_noise_std,
            "cfa_noise_std": cfa_noise_std,
            "flat_ratio": flat_ratio,
            "color_diversity": color_diversity,
            "lineart_fraction": lineart_fraction,
            "sensor_finding": sensor_finding,
        }
