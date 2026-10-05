"""Resolution and Contrast-Normalized Surface Normals Descriptor.

Extracts scale-invariant and resolution-invariant Lambertian residual statistics
from estimated surface normal fields (via DSINE or Sobel).
Eliminates high-resolution photographic texture kurtosis/variance artifacts.
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from .base import BasePhysicsExtractor, validate_and_load_image

# Path to pipeline_40k DSINE checkpoint
DEFAULT_DSINE_CHECKPOINT = (
    Path("G:/Thesis/pipeline_40k/models/normal_estimators/dsine/exp002_kappa/dsine.pt")
)


class SurfaceNormalsExtractor(BasePhysicsExtractor):
    """Summarize shading residuals using Sobel or a trained DSINE normal model.

    Features output in R^3:
    1. Contrast-normalized residual energy:
       norm_variance = var(error_map) / (contrast^2 + 1e-4)
    2. Residual skewness (clipped to [-3, 3])
    3. Residual kurtosis centered and normalized to avoid high-res macro texture explosions.
    """

    def __init__(
        self,
        normal_backend: str = "sobel",
        dsine_checkpoint: Optional[Union[str, Path]] = None,
        dsine_device: str = "cpu",
        dsine_fov_degrees: float = 60.0,
    ):
        super().__init__(feature_dim=3)
        if normal_backend not in {"sobel", "dsine"}:
            raise ValueError("normal_backend must be 'sobel' or 'dsine'")
        self.normal_backend = normal_backend
        self.dsine = None

        if normal_backend == "dsine":
            from .dsine_normals import DSINENormalEstimator

            ckpt = Path(dsine_checkpoint) if dsine_checkpoint else DEFAULT_DSINE_CHECKPOINT
            self.dsine = DSINENormalEstimator(
                ckpt,
                device=dsine_device,
                fov_degrees=dsine_fov_degrees,
            )

    @staticmethod
    def _sobel_normals(gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * (4.0 / max(height, width))
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * (4.0 / max(height, width))
        denom = np.sqrt(gx**2 + gy**2 + 1.0)
        return np.stack([-gx / denom, -gy / denom, 1.0 / denom], axis=-1)

    def _estimate_fields(
        self, image_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        if self.dsine is None:
            normals = self._sobel_normals(gray)
            kappa = None
        else:
            normals, kappa = self.dsine.predict(image_np)
        return gray, normals, kappa

    @staticmethod
    def _summarize_fields(
        gray: np.ndarray, normals: np.ndarray, kappa: Optional[np.ndarray]
    ) -> Tuple[torch.Tensor, float]:
        if gray.ndim != 2 or normals.shape != (*gray.shape, 3):
            raise ValueError("Expected aligned grayscale and HxWx3 normal fields")
        if gray.size == 0 or not np.isfinite(gray).all() or not np.isfinite(normals).all():
            raise ValueError("Normal summary fields must be non-empty and finite")

        if kappa is None:
            dsine_confidence = 1.0
        else:
            if kappa.shape != gray.shape or not np.isfinite(kappa).all():
                raise ValueError("DSINE concentration must align with the normal field")
            median_kappa = float(np.median(kappa[np.isfinite(kappa)]))
            dsine_confidence = float(np.clip(np.log1p(median_kappa) / np.log1p(20.0), 0.05, 1.0))

        # Contrast-aware normalization factor
        p95 = float(np.percentile(gray, 95))
        p5 = float(np.percentile(gray, 5))
        contrast = max(0.05, p95 - p5)

        # Estimate dominant lighting vector from normal-weighted irradiance
        weights = np.maximum(0.0, gray - np.mean(gray))[..., None]
        light_vector = np.sum(normals * weights, axis=(0, 1))
        light_vector /= np.linalg.norm(light_vector) + 1e-6
        lambertian = np.maximum(0.0, np.sum(normals * light_vector, axis=-1))

        # Smooth albedo via edge-preserving filter
        albedo = cv2.bilateralFilter(gray, d=9, sigmaColor=0.2, sigmaSpace=9)
        albedo = np.clip(albedo, 0.05, 1.0)

        # Contrast-normalized shading error map
        raw_error_map = np.abs(gray - albedo * lambertian)
        norm_error_map = raw_error_map / contrast

        variance = float(np.var(norm_error_map))
        standard_deviation = float(np.sqrt(variance) + 1e-7)
        centered = norm_error_map - np.mean(norm_error_map)

        skewness = float(np.clip(np.mean((centered / standard_deviation) ** 3), -3.0, 3.0))
        # Bounded kurtosis around normal distribution reference (0.0)
        kurtosis = float(np.clip(np.mean((centered / standard_deviation) ** 4) - 3.0, -3.0, 5.0))

        delta = torch.tensor([variance, skewness, kurtosis], dtype=torch.float32)

        # Physical reliability confidence
        image_confidence = float(np.clip(contrast * 1.5, 0.1, 1.0))
        confidence = float(np.clip(image_confidence * dsine_confidence, 0.05, 1.0))

        return delta, confidence

    def extract_regions(
        self,
        image: Union[np.ndarray, torch.Tensor],
        boxes: Iterable[Tuple[int, int, int, int]],
    ) -> list[Tuple[torch.Tensor, float]]:
        image_np = validate_and_load_image(image)
        gray, normals, kappa = self._estimate_fields(image_np)
        height, width = gray.shape
        summaries = []
        for y1, y2, x1, x2 in boxes:
            if not (0 <= y1 < y2 <= height and 0 <= x1 < x2 <= width):
                raise ValueError("Region box lies outside the image")
            region_kappa = None if kappa is None else kappa[y1:y2, x1:x2]
            summaries.append(
                self._summarize_fields(
                    gray[y1:y2, x1:x2], normals[y1:y2, x1:x2], region_kappa
                )
            )
        return summaries

    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)
        gray, normals, kappa = self._estimate_fields(image_np)
        return self._summarize_fields(gray, normals, kappa)
