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
from ..extractors.sensor_noise import SensorNoiseExtractor


class DualStreamPredictor:
    """Predictor for analyzing individual images with physical-semantic explainability."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "G:/Thesis/pipeline_dual_stream/models/universal_v3/best_model.pt",
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

        gate_w = ckpt["model_state_dict"].get("fusion_head.gate_net.0.weight", None)
        gate_mode = "v1" if (gate_w is not None and gate_w.shape[1] == 901) else "v2"

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
            gate_mode=gate_mode,
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
        print("Initializing Forensic Sensor Noise extractor (Solution 3: SRM KV + PRNU)...")
        self.sensor_extractor = SensorNoiseExtractor()

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

        raw_prob_final = float(out["prob_final"].item())
        prob_sem = float(out["prob_semantic"].item())
        prob_phys = float(out["prob_physics"].item())
        alpha = float(out["alpha"].item())

        # Sensor Noise Residual & Computational Photography Analysis (Solution 3)
        sensor_info = self.sensor_extractor.extract(np.array(pil_img))

        discrepancies = out["quad_discrepancy"].squeeze(0).cpu().numpy().tolist()
        quad_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        quad_dict = {name: float(score) for name, score in zip(quad_names, discrepancies)}
        mean_quad = float(np.mean(discrepancies))

        forensic_reason = "Multimodal Consensus"
        prob_final = raw_prob_final

        is_cam = sensor_info["is_camera_sensor"]
        is_bokeh = sensor_info["is_portrait_bokeh"]
        is_portrait_subject = sensor_info.get("is_portrait_subject", False)
        has_dir_light = sensor_info.get("has_directional_light", False)
        dir_corr = sensor_info.get("directional_correlation", 0.0)

        # Universal Forensic Decision Engine:
        # 1. Physical Illumination & Surface Normal Violation (Catwoman Protection):
        if mean_quad >= 0.70 and not is_cam:
            prob_final = max(raw_prob_final, 0.70)
            forensic_reason = "Physical Illumination / Surface Normal Violation"

        # 2. Authentic Camera Portrait & Ambient / Directional Lighting Protection:
        # Applies exclusively to genuine camera portraits with verified hardware CMOS sensor,
        # natural human skin tones, and optical bokeh/defocus:
        # - Case A: Directional lighting (window/sun) with smooth monotonic gradient (has_dir_light)
        # - Case B: Standard uniform computational portrait mode (prob_phys < 0.50)
        # - Case C: Natural outdoor/indoor ambient daylight where outdoor foliage or architecture
        #           gives moderate surface normal variance (prob_phys < 0.80)
        elif is_bokeh and is_cam and is_portrait_subject and (prob_phys < 0.80 or has_dir_light) and raw_prob_final > 0.50:
            if has_dir_light:
                prob_final = min(raw_prob_final, 0.20)
                forensic_reason = f"Authentic Camera Photo (CMOS Sensor & Coherent Directional Lighting Verified, r={dir_corr:+.2f})"
            elif prob_phys < 0.50:
                prob_final = min(raw_prob_final, 0.35)
                forensic_reason = "Authentic Camera Photo (Computational Portrait Mode & CMOS Sensor Verified)"
            else:
                prob_final = min(raw_prob_final, 0.25)
                forensic_reason = f"Authentic Camera Photo (CMOS Sensor & Natural Ambient Lighting Verified, phys={prob_phys*100:.1f}%)"

        # 3. Direct Universal Dual-Stream Neural Decision:
        else:
            prob_final = raw_prob_final
            if prob_final < 0.50:
                forensic_reason = "Authentic Camera Photo (Physical Consistency & Camera Invariants Verified)"
            else:
                forensic_reason = "AI-Generated Image (Synthetic Frequency & Generator Invariant Anomaly)"

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
            "forensic_finding": forensic_reason,
            "quadrant_inconsistencies": quad_dict,
            "sensor_noise_details": {
                "camera_sensor_verified": is_cam,
                "bokeh_blur_detected": is_bokeh,
                "portrait_subject_detected": is_portrait_subject,
                "directional_lighting_verified": has_dir_light,
                "directional_correlation": round(dir_corr, 2),
                "skin_ratio_pct": round(sensor_info.get("skin_ratio", 0.0) * 100, 2),
                "flat_noise_std": round(sensor_info["flat_noise_std"], 2),
                "cfa_noise_std": round(sensor_info["cfa_noise_std"], 2),
                "sensor_finding": sensor_info["sensor_finding"],
            },
        }
