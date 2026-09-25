"""DINOv2 Foundation Model Feature Extractor.
Extracts global and spatially aligned quadrant patch tokens using frozen DINOv2 ViT backbone.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms


class DINOv2Stream(nn.Module):
    """Frozen DINOv2 vision transformer stream.
    
    Provides:
    - Global CLS token: [B, 768]
    - Patch tokens: [B, N, 768]
    - 5 Regional tokens: [B, 5, 768] aligned with (global, top_left, top_right, bottom_left, bottom_right)
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.embed_dim = 768

        # Create model with dynamic image size support
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.freeze_backbone = freeze_backbone

        # Standard ViT bicubic preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def train(self, mode: bool = True):
        """Maintain frozen evaluation mode for backbone if freeze_backbone is True."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def extract_patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract CLS token and spatial patch tokens from images [B, 3, H, W].
        
        Returns:
            cls_token: [B, 768]
            patch_tokens: [B, N, 768]
        """
        # forward_features returns [B, 1 + num_patches, 768]
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:]
        return cls_token, patch_tokens

    def pool_quadrants(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Pool patch tokens into 4 spatial quadrants: (TL, TR, BL, BR).
        
        Args:
            patch_tokens: [B, N, 768] where N = H_p * W_p (e.g. 16 * 16 = 256)
        Returns:
            quadrant_tokens: [B, 4, 768]
        """
        batch_size, num_patches, dim = patch_tokens.shape
        grid_size = int(math.isqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Number of patches ({num_patches}) is not a square grid")

        # Reshape to spatial feature map: [B, H_p, W_p, D]
        spatial = patch_tokens.view(batch_size, grid_size, grid_size, dim)
        mid = grid_size // 2

        # 4 Quadrants aligned with Regional Physics layout
        q_tl = spatial[:, :mid, :mid, :].reshape(batch_size, -1, dim).mean(dim=1)
        q_tr = spatial[:, :mid, mid:, :].reshape(batch_size, -1, dim).mean(dim=1)
        q_bl = spatial[:, mid:, :mid, :].reshape(batch_size, -1, dim).mean(dim=1)
        q_br = spatial[:, mid:, mid:, :].reshape(batch_size, -1, dim).mean(dim=1)

        # [B, 4, 768]
        return torch.stack([q_tl, q_tr, q_bl, q_br], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass extracting global and regional semantic tokens.
        
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            dict containing cls_token, regional_tokens, and patch_tokens
        """
        # If input is in [0, 1], apply standard preprocessing; otherwise assume already normalized
        if x.min() >= 0.0 and x.max() <= 1.0:
            x_norm = self.preprocess(x)
        else:
            x_norm = x

        if self.freeze_backbone:
            with torch.no_grad():
                cls_token, patch_tokens = self.extract_patch_tokens(x_norm)
        else:
            cls_token, patch_tokens = self.extract_patch_tokens(x_norm)

        quadrant_tokens = self.pool_quadrants(patch_tokens)
        
        # Combine global CLS and 4 quadrants into 5 regional tokens
        regional_tokens = torch.cat([cls_token.unsqueeze(1), quadrant_tokens], dim=1)  # [B, 5, 768]

        return {
            "cls_token": cls_token,
            "regional_tokens": regional_tokens,
            "patch_tokens": patch_tokens,
        }
