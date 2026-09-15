"""Multi-Source Illumination Spherical Harmonics (SH) Regional Extractor.

Models multi-illuminant physical environments (e.g., ceiling bulb + desk lamp + window)
by decomposing regional light vectors into K=2 dominant directional modes via spherical clustering:
    r_i = min_{k in {1, 2}} (1 - d_i . m_k)
This prevents legitimate indoor and multi-light real photos from being penalized as fake,
recovering real-photo recall and cutting false positive rates.
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from .base import BasePhysicsExtractor
from ..data.preprocessor import extract_face_patches, validate_and_load_image


class ResidualBlock(nn.Module):
    """Lightweight residual block for patch-level feature extraction."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class LightweightResidualSHNet(nn.Module):
    """Lightweight residual CNN for estimating 9 order-2 spherical harmonics coefficients."""

    def __init__(self, sh_dim: int = 9):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, sh_dim)

        with torch.no_grad():
            self.head.weight.normal_(mean=0.0, std=0.02)
            self.head.bias.zero_()
            self.head.bias[0] = 1.0  # DC order-0 ambient baseline

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        sh_coeffs = self.head(features)
        sh_norm = torch.norm(sh_coeffs, p=2, dim=1, keepdim=True) + 1e-6
        return sh_coeffs / sh_norm


class IlluminationSHExtractor(BasePhysicsExtractor):
    """Estimates multi-source illumination consistency across 4x4 regional patches.

    Fits K=2 dominant directional light modes and computes minimum mode angular residuals.
    Outputs:
    - Delta_light in R^5:
        1. mean_multi_source_res: Mean angular residual to nearest light source mode
        2. max_multi_source_res: Peak isolated directional residual
        3. var_multi_source_res: Variance of multi-source residual field
        4. mode_separation: Angular separation between the two dominant sources
        5. spatial_coherence: Fraction of adjacent patches sharing the same mode assignment
    - Confidence c_light in [0, 1]
    """

    def __init__(self, patch_size: int = 64, grid_size: int = 4):
        super().__init__(feature_dim=5)
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.model = LightweightResidualSHNet(sh_dim=9)
        self.model.eval()

    def _prepare_patch(self, patch: np.ndarray) -> torch.Tensor:
        if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        resized = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return (tensor - 0.5) / 0.5

    def _extract_grid_patches(self, image_np: np.ndarray) -> torch.Tensor:
        h, w = image_np.shape[:2]
        step_h = h // self.grid_size
        step_w = w // self.grid_size
        patches = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                y1, y2 = r * step_h, (r + 1) * step_h
                x1, x2 = c * step_w, (c + 1) * step_w
                patch = image_np[y1:y2, x1:x2]
                patches.append(self._prepare_patch(patch))

        return torch.stack(patches, dim=0)

    def _fit_bimodal_sources(self, dirs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fits K=2 dominant directional light modes via spherical k-means."""
        # dirs: [16, 3] unit directional vectors
        # Initialize mode 1 as overall mean direction
        m1 = F.normalize(dirs.mean(dim=0, keepdim=True), p=2, dim=1)  # [1, 3]

        # Initialize mode 2 as the direction with largest angular distance from m1
        sim_to_m1 = torch.matmul(dirs, m1.T).squeeze(-1)  # [16]
        furthest_idx = torch.argmin(sim_to_m1)
        m2 = dirs[furthest_idx : furthest_idx + 1]  # [1, 3]

        # 3 iterations of spherical k-means
        modes = torch.cat([m1, m2], dim=0)  # [2, 3]
        assignments = torch.zeros(dirs.shape[0], dtype=torch.long, device=dirs.device)

        for _ in range(3):
            cos_sims = torch.matmul(dirs, modes.T)  # [16, 2]
            assignments = torch.argmax(cos_sims, dim=1)  # [16]

            mask0 = assignments == 0
            mask1 = assignments == 1

            if mask0.sum() > 0:
                modes[0] = F.normalize(dirs[mask0].mean(dim=0), p=2, dim=0)
            if mask1.sum() > 0:
                modes[1] = F.normalize(dirs[mask1].mean(dim=0), p=2, dim=0)

        # Compute residuals to nearest mode: r_i = 1 - max(d_i . m_k)
        cos_sims = torch.matmul(dirs, modes.T)
        nearest_sim, _ = torch.max(cos_sims, dim=1)
        residuals = torch.clamp(1.0 - nearest_sim, 0.0, 2.0)

        return modes, assignments, residuals

    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)
        grid_tensors = self._extract_grid_patches(image_np)

        with torch.no_grad():
            sh_coeffs = self.model(grid_tensors)  # [16, 9]

        # Extract order-1 directional components (channels 1, 2, 3 correspond to X, Y, Z directions)
        directional_sh = sh_coeffs[:, 1:4]
        dir_norms = torch.norm(directional_sh, p=2, dim=1, keepdim=True) + 1e-6
        dirs = directional_sh / dir_norms  # [16, 3]

        # Fit K=2 directional modes
        modes, assignments, residuals = self._fit_bimodal_sources(dirs)

        # 1. Mean residual to nearest light source mode
        mean_res = residuals.mean()

        # 2. Maximum isolated residual
        max_res = residuals.max()

        # 3. Variance of multi-source residual field
        var_res = residuals.var()

        # 4. Mode angular separation: 1 - (m1 . m2)
        mode_sep = torch.clamp(1.0 - torch.dot(modes[0], modes[1]), 0.0, 2.0)

        # 5. Spatial mode coherence across the 4x4 grid
        grid_assign = assignments.view(self.grid_size, self.grid_size)
        h_same = (grid_assign[:, :-1] == grid_assign[:, 1:]).float().mean()
        v_same = (grid_assign[:-1, :] == grid_assign[1:, :]).float().mean()
        spatial_coherence = (h_same + v_same) / 2.0

        delta = torch.stack([mean_res, max_res, var_res, mode_sep, spatial_coherence]).float()

        # Physical confidence conditioned on image contrast and texture dynamic range
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) / 255.0
        p_std = float(np.std(gray))
        p_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        confidence = float(np.clip((p_std * 2.5) * (p_range * 1.5), 0.05, 1.0))

        return delta, confidence
