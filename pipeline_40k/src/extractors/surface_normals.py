"""Surface-normal and Lambertian-residual physics-inspired descriptors."""

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .base import BasePhysicsExtractor
from .dsine_normals import DEFAULT_CHECKPOINT, DSINENormalEstimator
from ..data.preprocessor import validate_and_load_image


class SurfaceNormalsExtractor(BasePhysicsExtractor):
    """Summarize shading residuals using Sobel or a trained DSINE normal model.

    ``sobel`` is deterministic and lightweight, suitable for unit tests and
    low-resource experiments. ``dsine`` uses the official uncertainty-aware
    DSINE checkpoint stored locally; no random normal prediction head or
    network download is permitted.
    """

    def __init__(
        self,
        normal_backend: str = "sobel",
        dsine_checkpoint: Optional[Union[str, Path]] = None,
        dsine_device: str = "cpu",
        dsine_fov_degrees: float = 60.0,
        use_foundation_backbone: bool = False,
        normal_head_weights: Optional[Union[str, Path]] = None,
    ):
        super().__init__(feature_dim=3)
        if use_foundation_backbone or normal_head_weights is not None:
            raise ValueError(
                "The historical random foundation normal head is retired. "
                "Use normal_backend='dsine' with an official trained checkpoint."
            )
        if normal_backend not in {"sobel", "dsine"}:
            raise ValueError("normal_backend must be 'sobel' or 'dsine'")
        self.normal_backend = normal_backend
        self.dsine = None
        if normal_backend == "dsine":
            self.dsine = DSINENormalEstimator(
                Path(dsine_checkpoint) if dsine_checkpoint else DEFAULT_CHECKPOINT,
                device=dsine_device,
                fov_degrees=dsine_fov_degrees,
            )

    def provenance(self) -> dict:
        if self.dsine is None:
            return {"backend": "deterministic_sobel", "trained_checkpoint": None}
        return self.dsine.provenance()

    @staticmethod
    def _sobel_normals(gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * (4.0 / max(height, width))
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * (4.0 / max(height, width))
        denom = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
        return np.stack([-gx / denom, -gy / denom, 1.0 / denom], axis=-1)

    def _estimate_fields(
        self, image_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Estimate grayscale, normals, and optional DSINE concentration once."""
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
        """Summarize aligned image/normal fields without rerunning the estimator."""
        if gray.ndim != 2 or normals.shape != (*gray.shape, 3):
            raise ValueError("Expected aligned grayscale and HxWx3 normal fields")
        if gray.size == 0 or not np.isfinite(gray).all() or not np.isfinite(normals).all():
            raise ValueError("Normal summary fields must be non-empty and finite")
        if kappa is None:
            dsine_confidence = 1.0
        else:
            if kappa.shape != gray.shape or not np.isfinite(kappa).all():
                raise ValueError("DSINE concentration must align with the normal field")
            # DSINE's fourth output is a positive concentration parameter. A
            # logarithmic mapping keeps one extreme region from dominating the
            # image-level confidence while preserving its uncertainty ordering.
            median_kappa = float(np.median(kappa[np.isfinite(kappa)]))
            dsine_confidence = float(np.clip(np.log1p(median_kappa) / np.log1p(20.0), 0.05, 1.0))

        weights = np.maximum(0.0, gray - np.mean(gray))[..., None]
        light_vector = np.sum(normals * weights, axis=(0, 1))
        light_vector /= np.linalg.norm(light_vector) + 1e-6
        lambertian = np.maximum(0.0, np.sum(normals * light_vector, axis=-1))

        albedo = cv2.bilateralFilter(gray, d=9, sigmaColor=0.2, sigmaSpace=9)
        albedo = np.clip(albedo, 0.05, 1.0)
        error_map = np.abs(gray - albedo * lambertian)
        variance = float(np.var(error_map))
        standard_deviation = float(np.sqrt(variance) + 1e-7)
        centered = error_map - np.mean(error_map)
        skewness = float(np.clip(np.mean((centered / standard_deviation) ** 3), -5.0, 5.0))
        kurtosis = float(np.clip(np.mean((centered / standard_deviation) ** 4) - 3.0, -3.0, 20.0))
        delta = torch.tensor([variance, skewness, kurtosis], dtype=torch.float32)

        contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        image_confidence = float(np.clip(contrast * 1.6 + 0.1, 0.1, 1.0))
        confidence = float(np.clip(image_confidence * dsine_confidence, 0.05, 1.0))
        return delta, confidence

    def extract_regions(
        self,
        image: Union[np.ndarray, torch.Tensor],
        boxes: Iterable[Tuple[int, int, int, int]],
    ) -> list[Tuple[torch.Tensor, float]]:
        """Summarize several regions after one full-image normal prediction.

        Boxes use ``(y1, y2, x1, x2)`` coordinates. Estimating DSINE on the
        complete image preserves scene context and makes regional extraction
        substantially faster than predicting each crop independently.
        """
        image_np = validate_and_load_image(image)
        gray, normals, kappa = self._estimate_fields(image_np)
        height, width = gray.shape
        summaries = []
        for y1, y2, x1, x2 in boxes:
            if not (0 <= y1 < y2 <= height and 0 <= x1 < x2 <= width):
                raise ValueError("Region box lies outside the image")
            region_kappa = None if kappa is None else kappa[y1:y2, x1:x2]
            summaries.append(self._summarize_fields(
                gray[y1:y2, x1:x2], normals[y1:y2, x1:x2], region_kappa
            ))
        return summaries

    def extract(self, image: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)
        gray, normals, kappa = self._estimate_fields(image_np)
        return self._summarize_fields(gray, normals, kappa)
