"""Multi-Source Illumination Spherical Harmonics (SH) Regional Extractor.

Decomposes local patch irradiance into analytical order-2 Spherical Harmonics (SH)
representations (Ramamoorthi & Hanrahan 2001, Basri & Jacobs 2003) via deterministic
closed-form least-squares fitting against gradient-derived surface normals.

Regional light vectors across a 4x4 spatial grid are clustered into K=2 dominant
directional modes via spherical k-means:
    r_i = min_{k in {1, 2}} (1 - d_i . m_k)
This models complex multi-illuminant physical environments (e.g. ambient + key light)
without un-trained neural network weights or heuristic shortcuts.
"""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from .base import BasePhysicsExtractor
from ..data.preprocessor import validate_and_load_image


def fit_sh_patch_least_squares(
    patch_gray: np.ndarray, reg_lambda: float = 1e-3
) -> np.ndarray:
    """Fits 9 Spherical Harmonics coefficients to a grayscale image patch via least squares.

    Uses gradient-derived surface normals n = (-gx, -gy, 1) / sqrt(gx^2 + gy^2 + 1).
    Order-2 basis functions (9 coefficients):
        Y0 = 1.0
        Y1 = ny, Y2 = nz, Y3 = nx  (Order 1 directional lighting)
        Y4 = nx*ny, Y5 = ny*nz, Y6 = (3*nz^2 - 1)/2, Y7 = nx*nz, Y8 = nx^2 - ny^2 (Order 2)

    Returns:
        coeffs: 9-dimensional float array of SH coefficients.
    """
    h, w = patch_gray.shape
    if h < 4 or w < 4:
        return np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    scale = 2.0 / max(h, w)
    gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3) * scale
    gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3) * scale
    denom = np.sqrt(gx**2 + gy**2 + 1.0)

    nx = -gx / denom
    ny = -gy / denom
    nz = 1.0 / denom

    y = patch_gray.reshape(-1, 1).astype(np.float32)
    nx_f = nx.reshape(-1, 1)
    ny_f = ny.reshape(-1, 1)
    nz_f = nz.reshape(-1, 1)

    Y0 = np.ones_like(nx_f)
    Y1 = ny_f
    Y2 = nz_f
    Y3 = nx_f
    Y4 = nx_f * ny_f
    Y5 = ny_f * nz_f
    Y6 = (3.0 * nz_f**2 - 1.0) / 2.0
    Y7 = nx_f * nz_f
    Y8 = (nx_f**2 - ny_f**2)

    B = np.hstack([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8])
    BTB = B.T @ B
    reg = reg_lambda * np.eye(9, dtype=np.float32)
    BTy = B.T @ y

    try:
        coeffs = np.linalg.solve(BTB + reg, BTy).flatten()
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(BTB + reg) @ BTy.flatten()

    return coeffs.astype(np.float32)


class IlluminationSHExtractor(BasePhysicsExtractor):
    """Estimates multi-source illumination consistency across 4x4 regional patches

    using deterministic closed-form Spherical Harmonics decomposition.

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

    def _fit_bimodal_sources(
        self, dirs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fits K=2 dominant directional light modes via spherical k-means."""
        m1 = F.normalize(dirs.mean(dim=0, keepdim=True), p=2, dim=1)

        sim_to_m1 = torch.matmul(dirs, m1.T).squeeze(-1)
        furthest_idx = torch.argmin(sim_to_m1)
        m2 = dirs[furthest_idx : furthest_idx + 1]

        modes = torch.cat([m1, m2], dim=0)
        assignments = torch.zeros(dirs.shape[0], dtype=torch.long, device=dirs.device)

        for _ in range(3):
            cos_sims = torch.matmul(dirs, modes.T)
            assignments = torch.argmax(cos_sims, dim=1)

            mask0 = assignments == 0
            mask1 = assignments == 1

            if mask0.sum() > 0:
                modes[0] = F.normalize(dirs[mask0].mean(dim=0), p=2, dim=0)
            if mask1.sum() > 0:
                modes[1] = F.normalize(dirs[mask1].mean(dim=0), p=2, dim=0)

        cos_sims = torch.matmul(dirs, modes.T)
        nearest_sim, _ = torch.max(cos_sims, dim=1)
        residuals = torch.clamp(1.0 - nearest_sim, 0.0, 2.0)

        return modes, assignments, residuals

    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        image_np = validate_and_load_image(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        h, w = gray.shape[:2]
        step_h = h // self.grid_size
        step_w = w // self.grid_size

        dirs = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                patch = gray[r * step_h : (r + 1) * step_h, c * step_w : (c + 1) * step_w]
                coeffs = fit_sh_patch_least_squares(patch)
                # Directional lighting vector from order-1 basis coefficients:
                # Y3=nx -> coeffs[3], Y1=ny -> coeffs[1], Y2=nz -> coeffs[2]
                d = torch.tensor([coeffs[3], coeffs[1], coeffs[2]], dtype=torch.float32)
                d_norm = torch.norm(d, p=2) + 1e-6
                dirs.append(d / d_norm)

        dirs_tensor = torch.stack(dirs, dim=0)  # [16, 3]

        # Fit K=2 directional modes via spherical clustering
        modes, assignments, residuals = self._fit_bimodal_sources(dirs_tensor)

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
        p_std = float(np.std(gray))
        p_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        confidence = float(np.clip((p_std * 2.5) * (p_range * 1.5), 0.05, 1.0))

        return delta, confidence
