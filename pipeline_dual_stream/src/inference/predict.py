"""End-to-End Inference and Explainability Engine for Dual-Stream Hybrid Detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

import sys
from pathlib import Path
if "G:/Thesis" not in sys.path:
    sys.path.insert(0, "G:/Thesis")

from pipeline_40k.src.extractors.surface_normals import SurfaceNormalsExtractor
from pipeline_40k.src.extractors.regional_physics import RegionalPhysicsExtractor

from ..models.hybrid_detector import DualStreamHybridDetector


class DualStreamPredictor:
    """Predictor for analyzing individual images with physical-semantic explainability."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "G:/Thesis/pipeline_dual_stream/models/dual_stream_v1/best_model.pt",
        dsine_checkpoint: Union[str, Path] = "G:/Thesis/pipeline_40k/models/normal_estimators/dsine/exp002_kappa/dsine.pt",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        print(f"Loading Dual-Stream model from {checkpoint_path} to {self.device}...")
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.standardizer = ckpt["standardizer"]
        config = ckpt.get("config", {})

        # Initialize detector with DINOv2 weights loaded
        self.model = DualStreamHybridDetector(
            dinov2_model_name="vit_base_patch14_dinov2",
            freeze_dinov2=True,
            img_size=224,
            phys_dim=config.get("phys_dim", 64),
            proj_dim=config.get("proj_dim", 128),
            num_heads=config.get("num_heads", 4),
            dropout=0.0,
            load_pretrained_dinov2=True,
        ).to(self.device)

        # Load only trained head and physics parameters, preserving official pretrained DINOv2
        state_dict = ckpt["model_state_dict"]
        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("semantic_stream.backbone")}
        self.model.load_state_dict(filtered_dict, strict=False)
        self.model.eval()

        # Initialize Regional Physics Extractor
        print("Initializing Regional Physics extractor (DSINE + SH + Optics)...")
        normals_extractor = SurfaceNormalsExtractor(
            normal_backend="dsine",
            dsine_checkpoint=Path(dsine_checkpoint),
            dsine_device=self.device,
        )
        self.physics_extractor = RegionalPhysicsExtractor(normals=normals_extractor)

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict[str, Union[float, str, dict]]:
        """Run full dual-stream inference with explainable breakdown.
        
        Returns:
            dict containing:
                - verdict: "AI-Generated" or "Authentic Camera"
                - final_prob_fake: float in [0, 1]
                - semantic_prob_fake: float in [0, 1]
                - physics_prob_fake: float in [0, 1]
                - trust_alpha: float in [0, 1] (1 = Semantic trust, 0 = Physics trust)
                - quadrant_inconsistencies: dict of 4 quadrant scores
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        # 1. Prepare image tensor for DINOv2
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(pil_img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        # 2. Extract Regional Physics features
        phys_feats, phys_confs = self.physics_extractor.extract(np.array(pil_img))
        phys_feats = phys_feats.unsqueeze(0)  # [1, 5, 14]
        phys_confs = phys_confs.unsqueeze(0)  # [1, 5, 4]

        # Apply standardizer
        mean = torch.tensor(self.standardizer["mean"], dtype=torch.float32)
        scale = torch.tensor(self.standardizer["scale"], dtype=torch.float32)
        phys_feats_norm = ((phys_feats - mean.unsqueeze(0)) / scale.unsqueeze(0)).to(self.device)
        phys_confs = phys_confs.to(self.device)

        # 3. Forward pass
        with torch.no_grad():
            out = self.model(
                images=img_tensor,
                physics_features=phys_feats_norm,
                physics_confidences=phys_confs,
            )

        prob_final = float(out["prob_final"].item())
        prob_sem = float(out["prob_semantic"].item())
        prob_phys = float(out["prob_physics"].item())
        alpha = float(out["alpha"].item())

        # Forensic Evidence Asymmetry Rule:
        # A real camera photograph is never an anime/digital illustration.
        # If DINOv2 identifies synthetic/digital art with high confidence (prob_sem >= 0.70)
        # while the scene geometry/lighting happens to be mathematically smooth (prob_phys < 0.40),
        # smooth physics cannot veto the fake detection.
        if prob_sem >= 0.70 and prob_phys < 0.40:
            alpha = max(alpha, 0.95)
            prob_final = alpha * prob_sem + (1.0 - alpha) * prob_phys

        discrepancies = out["quad_discrepancy"].squeeze(0).cpu().numpy().tolist()

        quad_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        quad_dict = {name: float(score) for name, score in zip(quad_names, discrepancies)}

        return {
            "verdict": "AI-Generated Image" if prob_final >= 0.5 else "Authentic Camera Photo",
            "confidence": f"{max(prob_final, 1.0 - prob_final) * 100:.1f}%",
            "final_prob_fake": prob_final,
            "semantic_prob_fake": prob_sem,
            "physics_prob_fake": prob_phys,
            "trust_alpha": alpha,
            "trust_interpretation": (
                f"{alpha*100:.1f}% Semantic Foundation vs {(1.0-alpha)*100:.1f}% Physical Consistency"
            ),
            "quadrant_inconsistencies": quad_dict,
        }
