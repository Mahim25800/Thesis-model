"""Local adapter for the official DSINE uncertainty-aware checkpoint."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2

P40K_ROOT = Path("G:/Thesis/pipeline_40k")
DSINE_SOURCE = P40K_ROOT / "third_party" / "DSINE"
DSINE_RUNTIME = P40K_ROOT / "third_party" / "python_packages"
DEFAULT_CHECKPOINT = P40K_ROOT / "models" / "normal_estimators" / "dsine" / "exp002_kappa" / "dsine.pt"
DSINE_SOURCE_COMMIT = "ef0c2afa32b4dd19cb8ca4567c652802cd92591c"
OFFICIAL_CHECKPOINT_SHA256 = "1BDDAE86D7D8F59912C50F3164C2543B1B80FDADFBF1C5DD0AB2D85108BBA955"
MAX_INPUT_DIMENSION = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_local_import_paths() -> None:
    for location in (DSINE_RUNTIME, DSINE_SOURCE):
        value = str(location)
        if not location.is_dir():
            raise FileNotFoundError(f"Required DSINE runtime path is missing: {location}")
        if value not in sys.path:
            sys.path.insert(0, value)


def _dsine_args() -> SimpleNamespace:
    return SimpleNamespace(
        NNET_encoder_B=5,
        NNET_decoder_NF=2048,
        NNET_decoder_BN=False,
        NNET_decoder_down=8,
        NNET_learned_upsampling=True,
        NNET_output_dim=4,
        NNET_output_type="G",
        NNET_feature_dim=64,
        NNET_hidden_dim=64,
        NRN_prop_ps=5,
        NRN_num_iter_train=5,
        NRN_num_iter_test=5,
        NRN_ray_relu=True,
    )


class DSINENormalEstimator:
    """Estimate unit normals and DSINE's per-pixel concentration on one image."""

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        device: str = "cpu",
        fov_degrees: float = 60.0,
    ):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Official DSINE checkpoint is missing: {self.checkpoint}")
        if not 1.0 < fov_degrees < 179.0:
            raise ValueError("DSINE field of view must lie between 1 and 179 degrees")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DSINE but is unavailable")
        self.fov_degrees = float(fov_degrees)
        _install_local_import_paths()
        import geffnet

        create_model = geffnet.create_model

        def local_encoder(*args, **kwargs):
            kwargs["pretrained"] = False
            return create_model(*args, **kwargs)

        geffnet.create_model = local_encoder
        try:
            from models.dsine.v02_kappa import DSINE_v02_kappa

            model = DSINE_v02_kappa(_dsine_args())
        finally:
            geffnet.create_model = create_model
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)["model"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        self.model.pixel_coords = self.model.pixel_coords.to(self.device)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        if (
            self.checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve()
            and self.checkpoint_sha256.upper() != OFFICIAL_CHECKPOINT_SHA256
        ):
            raise ValueError("Official DSINE checkpoint hash does not match the recorded provenance")

    def _intrinsics(self, height: int, width: int) -> torch.Tensor:
        focal = (max(height, width) / 2.0) / math.tan(math.radians(self.fov_degrees / 2.0))
        return torch.tensor(
            [[focal, 0.0, width / 2.0 - 0.5], [0.0, focal, height / 2.0 - 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    @staticmethod
    def _padding(height: int, width: int) -> Tuple[int, int, int, int]:
        padded_width = 32 * math.ceil(width / 32)
        padded_height = 32 * math.ceil(height / 32)
        pad_w = padded_width - width
        pad_h = padded_height - height
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return left, right, top, bottom

    def predict(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("DSINE expects an RGB uint8 image with shape [height, width, 3]")
        original_height, original_width = image.shape[:2]
        scale = min(1.0, MAX_INPUT_DIMENSION / max(original_height, original_width))
        if scale < 1.0:
            width = max(1, round(original_width * scale))
            height = max(1, round(original_height * scale))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        else:
            height, width = original_height, original_width
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
        tensor = (
            tensor
            - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        ) / torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        left, right, top, bottom = self._padding(height, width)
        tensor = F.pad(tensor, (left, right, top, bottom), mode="constant", value=0.0)
        intrins = self._intrinsics(height, width)
        intrins[:, 0, 2] += left
        intrins[:, 1, 2] += top
        with torch.inference_mode():
            prediction = self.model(tensor, intrins=intrins, mode="test")[-1]
        prediction = prediction[:, :, top : top + height, left : left + width]
        normals = F.normalize(prediction[:, :3], dim=1)[0].permute(1, 2, 0).cpu().numpy()
        kappa = prediction[0, 3].clamp_min(0).cpu().numpy()
        if (height, width) != (original_height, original_width):
            normals_t = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0)
            normals = (
                F.normalize(
                    F.interpolate(
                        normals_t,
                        size=(original_height, original_width),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    dim=1,
                )[0]
                .permute(1, 2, 0)
                .numpy()
            )
            kappa = F.interpolate(
                torch.from_numpy(kappa).unsqueeze(0).unsqueeze(0),
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
        if not np.isfinite(normals).all() or not np.isfinite(kappa).all():
            raise RuntimeError("DSINE returned non-finite geometry values")
        return normals.astype(np.float32), kappa.astype(np.float32)
