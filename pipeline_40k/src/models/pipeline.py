"""Unified Physics-Based Deepfake Detection Pipeline."""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn

from ..extractors import (
    IlluminationSHExtractor,
    CornealOpticsExtractor,
    SurfaceNormalsExtractor,
    ChromaticShadowExtractor,
)
from .cross_gen_gated import GatedCrossGenClassifier
from ..data.preprocessor import validate_and_load_image


class PhysicsDeepfakePipeline(nn.Module):
    """Binds the 4 physics extractors and the attention-gated classifier.

    Ensures downstream classification operates strictly on physical discrepancies
    and extraction confidence rather than raw pixels.
    """

    def __init__(
        self,
        classifier: Optional[GatedCrossGenClassifier] = None,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        # 4 Physics Extractors
        self.extractor_sh = IlluminationSHExtractor()
        self.extractor_cornea = CornealOpticsExtractor()
        self.extractor_normals = SurfaceNormalsExtractor()
        self.extractor_chrom = ChromaticShadowExtractor()

        # Classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = GatedCrossGenClassifier()

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.classifier.load_state_dict(state_dict)

        self.classifier.to(self.device)
        self.classifier.eval()

    def extract_features(
        self, image: Union[np.ndarray, torch.Tensor, str, Path]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Extracts 14-dim discrepancy vector and 4-dim confidence vector."""
        img_np = validate_and_load_image(image)

        # 1. Illumination Spherical Harmonics (5D)
        delta_light, c_light = self.extractor_sh(img_np)

        # 2. Corneal Specular Glints (4D)
        delta_cornea, c_cornea = self.extractor_cornea(img_np)

        # 3. Surface Normals Residuals (3D)
        delta_geom, c_geom = self.extractor_normals(img_np)

        # 4. Chromatic Shadow Drift (2D)
        delta_chrom, c_chrom = self.extractor_chrom(img_np)

        # Concatenate into 14D discrepancy vector
        delta = torch.cat([delta_light, delta_cornea, delta_geom, delta_chrom], dim=0)

        # 4D confidence vector
        confidences = torch.tensor(
            [c_light, c_cornea, c_geom, c_chrom], dtype=torch.float32
        )

        breakdown = {
            "light": {"delta": delta_light.cpu().tolist(), "conf": c_light},
            "cornea": {"delta": delta_cornea.cpu().tolist(), "conf": c_cornea},
            "geom": {"delta": delta_geom.cpu().tolist(), "conf": c_geom},
            "chrom": {"delta": delta_chrom.cpu().tolist(), "conf": c_chrom},
        }

        return delta, confidences, breakdown

    def forward(
        self,
        image: Union[np.ndarray, torch.Tensor, str, Path],
        obs_threshold: float = 1.0,
    ) -> Dict[str, Any]:
        """End-to-end inference from an image to physics-based deepfake verdict with observability gating."""
        delta, confidences, breakdown = self.extract_features(image)
        observability = float(confidences.sum().item())

        delta_tensor = delta.unsqueeze(0).to(self.device)
        conf_tensor = confidences.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.classifier(delta_tensor, conf_tensor)
            if isinstance(output, tuple):
                logit = output[0].squeeze().item()
            else:
                logit = output.squeeze().item()
            prob = torch.sigmoid(torch.tensor(logit)).item()

        is_observable = observability >= obs_threshold
        raw_pred = 1 if prob >= 0.5 else 0

        if not is_observable:
            verdict = "INDETERMINATE (Low Physical Evidence)"
        else:
            verdict = "FAKE" if raw_pred == 1 else "REAL"

        return {
            "logit": logit,
            "probability_fake": prob,
            "prediction": raw_pred,
            "verdict": verdict,
            "is_observable": is_observable,
            "observability_score": observability,
            "delta_features": delta.cpu().tolist(),
            "confidences": confidences.cpu().tolist(),
            "physics_breakdown": breakdown,
        }
