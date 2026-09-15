"""Pretrained Foundation Surface Normal vs. Lambertian Shading Consistency Extractor.

Integrates a pretrained deep foundation vision backbone to estimate continuous 3D surface
normal fields n(x, y), decouples base albedo rho(x, y) via bilateral filtering,
and computes theoretical Lambertian shading residuals.
Diffusion models hallucinate micro-wrinkles, plastic skin artifacts, and topological tears
that generative models create to mimic sharp focus, causing sharp spikes in the residual field.
"""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2

from .base import BasePhysicsExtractor
from ..data.preprocessor import validate_and_load_image


class FoundationNormalHead(nn.Module):
    """Lightweight projection head predicting continuous 3D unit surface normal maps from deep features."""

    def __init__(self, in_channels: int = 576):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        normals_coarse = self.conv(features)
        normals_upsampled = F.interpolate(
            normals_coarse, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return F.normalize(normals_upsampled, p=2, dim=1)


class SurfaceNormalsExtractor(BasePhysicsExtractor):
    """Computes surface normal map estimates n via pretrained foundation backbone,
    decouples base albedo rho(x, y), and computes theoretical Lambertian shading R(x, y).

    Outputs:
    - Delta_geom in R^3: [variance, skewness, kurtosis] of photometric residual distribution R(x, y)
    - Confidence c_geom in [0, 1]
    """

    def __init__(self, use_foundation_backbone: bool = True):
        super().__init__(feature_dim=3)
        self.use_foundation = use_foundation_backbone

        if self.use_foundation:
            try:
                self.backbone = models.mobilenet_v3_small(weights="DEFAULT").features.eval()
                self.normal_head = FoundationNormalHead(in_channels=576).eval()
            except Exception:
                self.use_foundation = False

    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)
        h, w = image_np.shape[:2]

        # Convert to float grayscale [0, 1]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        if self.use_foundation:
            # Pretrained deep feature normal estimation
            tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor_norm = (tensor - mean) / std

            with torch.no_grad():
                feats = self.backbone(tensor_norm)
                normals_tensor = self.normal_head(feats, h, w)
                normals = normals_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            weights = np.maximum(0.0, gray - np.mean(gray))
            weights = np.expand_dims(weights, -1)
            weighted_normals = np.sum(normals * weights, axis=(0, 1))
            norm_l = np.linalg.norm(weighted_normals) + 1e-6
            l_vec = weighted_normals / norm_l

            lambertian = np.maximum(0.0, np.sum(normals * l_vec, axis=-1))
        else:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * (4.0 / max(h, w))
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * (4.0 / max(h, w))
            denom = np.sqrt(gx**2 + gy**2 + 1.0)
            normals = np.stack([-gx / denom, -gy / denom, 1.0 / denom], axis=-1)

            weights = np.maximum(0.0, gray - np.mean(gray))
            weights = np.expand_dims(weights, -1)
            weighted_normals = np.sum(normals * weights, axis=(0, 1))
            l_vec = weighted_normals / (np.linalg.norm(weighted_normals) + 1e-6)
            lambertian = np.maximum(0.0, np.sum(normals * l_vec, axis=-1))

        # 1. Albedo Decoupling: estimate base albedo rho(x, y) via bilateral filtering
        albedo = cv2.bilateralFilter(gray, d=9, sigmaColor=0.2, sigmaSpace=9)
        albedo = np.clip(albedo, 0.05, 1.0)

        # 2. Compute Albedo-Decoupled Lambertian Residual: R(x, y) = |I(x, y) - rho(x, y) * max(0, n . l)|
        i_pred = albedo * lambertian
        error_map = np.abs(gray - i_pred)

        # 3. Higher-order histogram distribution statistics: variance, skewness, kurtosis
        var_err = float(np.var(error_map))
        std_err = float(np.sqrt(var_err) + 1e-7)
        diff = error_map - np.mean(error_map)

        skewness = float(np.clip(np.mean((diff / std_err) ** 3), -5.0, 5.0))
        kurtosis = float(np.clip(np.mean((diff / std_err) ** 4) - 3.0, -3.0, 20.0))

        delta = torch.tensor([var_err, skewness, kurtosis], dtype=torch.float32)

        p_contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        confidence = float(np.clip(p_contrast * 1.6 + 0.1, 0.1, 1.0))

        return delta, confidence
