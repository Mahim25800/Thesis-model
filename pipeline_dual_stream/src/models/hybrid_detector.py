"""Dual-Stream Hybrid Deepfake Detector.
Unifies DINOv2 vision foundation features and Regional Multi-Physics representations.
Supports both end-to-end raw-image inference and high-throughput cached-token training.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .dinov2_stream import DINOv2Stream
from .gated_fusion import GatedCrossAttentionFusion, dual_stream_hybrid_loss
from .physics_stream import RegionalPhysicsStream


class DualStreamHybridDetector(nn.Module):
    """Complete Dual-Stream Deepfake Detector.
    
    Streams:
    1. Semantic Foundation Stream: DINOv2 ViT-Base extracting global and quadrant tokens.
    2. Regional Multi-Physics Stream: 4 physical entities across 5 spatial regions.
    3. Gated Cross-Attention Fusion: Dynamic trust gating and spatial discrepancy analysis.
    """

    def __init__(
        self,
        dinov2_model_name: str = "vit_base_patch14_dinov2",
        freeze_dinov2: bool = True,
        img_size: int = 224,
        phys_dim: int = 64,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.15,
        load_pretrained_dinov2: bool = True,
    ):
        super().__init__()
        self.img_size = img_size

        # Semantic Stream
        self.semantic_stream = DINOv2Stream(
            model_name=dinov2_model_name,
            pretrained=load_pretrained_dinov2,
            freeze_backbone=freeze_dinov2,
            img_size=img_size,
        )

        # Physics Stream
        self.physics_stream = RegionalPhysicsStream(
            d_model=phys_dim,
            dropout=dropout,
        )

        # Gated Cross-Attention Fusion Head
        self.fusion_head = GatedCrossAttentionFusion(
            sem_dim=768,
            phys_dim=phys_dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        physics_features: Optional[torch.Tensor] = None,
        physics_confidences: Optional[torch.Tensor] = None,
        # Cached feature inputs for fast training
        dinov2_cls: Optional[torch.Tensor] = None,
        dinov2_regional: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass supporting both raw images and precomputed caches.
        
        Args:
            images: [B, 3, H, W] RGB images (if not using precomputed dinov2)
            physics_features: [B, 5, 14]
            physics_confidences: [B, 5, 4]
            dinov2_cls: [B, 768] (optional precomputed CLS token)
            dinov2_regional: [B, 5, 768] (optional precomputed regional tokens)
        """
        if physics_features is None or physics_confidences is None:
            raise ValueError("physics_features and physics_confidences are required")

        # 1. Semantic Stream features
        if dinov2_cls is not None and dinov2_regional is not None:
            sem_cls = dinov2_cls
            sem_reg = dinov2_regional
        elif images is not None:
            sem_out = self.semantic_stream(images)
            sem_cls = sem_out["cls_token"]
            sem_reg = sem_out["regional_tokens"]
        else:
            raise ValueError("Either images or precomputed dinov2 tokens must be provided")

        # 2. Physics Stream features
        phys_out = self.physics_stream(
            features=physics_features,
            confidences=physics_confidences,
        )

        # 3. Gated Cross-Attention Fusion
        fusion_out = self.fusion_head(
            sem_tokens=sem_reg,
            sem_global=sem_cls,
            phys_tokens=phys_out["regional_tokens"],
            phys_global=phys_out["global_summary"],
            phys_conf=physics_confidences,
            physics_logits=phys_out["physics_logits"],
        )

        # 4. Attach probabilities
        fusion_out["prob_final"] = torch.sigmoid(fusion_out["logits"])
        fusion_out["prob_semantic"] = torch.sigmoid(fusion_out["sem_logits"])
        fusion_out["prob_physics"] = torch.sigmoid(fusion_out["phys_logits"])

        return fusion_out
