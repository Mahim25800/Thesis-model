"""Regional Multi-Physics Stream.
Encodes 5 spatial regions and 4 physical entities into structured regional physics tokens.
Fully compatible with the regional DSINE cache schema [5, 14] features and [5, 4] confidences.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Physics entities: (Illumination: 5, Specular: 4, Normals: 3, Chromatic Shadow: 2)
PHYSICS_ENTITY_DIMS: Tuple[int, ...] = (5, 4, 3, 2)
REGION_COUNT: int = 5  # Global, TL, TR, BL, BR


class RegionalPhysicsStream(nn.Module):
    """Encodes regional physical descriptors and pairwise physical interactions."""

    def __init__(
        self,
        d_model: int = 64,
        dropout: float = 0.15,
        entity_dims: Tuple[int, ...] = PHYSICS_ENTITY_DIMS,
    ):
        super().__init__()
        self.d_model = d_model
        self.entity_dims = entity_dims
        self.total_feat_dim = sum(entity_dims)  # 14

        # Per-entity encoders: takes (feat_dim + 1 confidence scalar) -> d_model
        self.entity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 1, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
            for dim in self.entity_dims
        ])

        # Learned entity type and missing-cue tokens
        self.entity_type = nn.Parameter(torch.randn(len(entity_dims), d_model) * 0.02)
        self.missing_tokens = nn.Parameter(torch.randn(len(entity_dims), d_model) * 0.02)

        # Pairwise entity interactions (6 pairs among 4 entities)
        pairs = tuple(
            (i, j)
            for i in range(len(entity_dims))
            for j in range(i + 1, len(entity_dims))
        )
        self.register_buffer("pair_indices", torch.tensor(pairs, dtype=torch.long), persistent=False)

        # Pair interaction encoder: takes [left, right, |left-right|, left*right, c_left, c_right] -> d_model
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * d_model + 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.pair_attention = nn.Linear(d_model, 1)

        # Projection combining entity summary and pair summary
        self.region_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.region_type = nn.Parameter(torch.randn(REGION_COUNT, d_model) * 0.02)

        # 2-layer Regional Transformer comparing the 5 regions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.region_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Standalone physics classifier head (for ablation and multi-task supervision)
        self.standalone_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        confidences: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode regional physics features into physics tokens.
        
        Args:
            features: [B, 5, 14]
            confidences: [B, 5, 4]
        Returns:
            dict containing:
                'regional_tokens': [B, 5, d_model]
                'global_summary': [B, 2 * d_model] (global token + attention-pooled regional token)
                'physics_logits': [B, 1]
                'observability': [B, 5]
        """
        batch_size = features.shape[0]

        # 1. Encode entities with confidence gating
        entity_rows = []
        start_idx = 0
        for e_idx, (dim, encoder) in enumerate(zip(self.entity_dims, self.entity_encoders)):
            conf = confidences[:, :, e_idx : e_idx + 1]  # [B, 5, 1]
            feat = features[:, :, start_idx : start_idx + dim]  # [B, 5, dim]
            encoded = encoder(torch.cat([feat, conf], dim=-1))  # [B, 5, d_model]
            missing = self.missing_tokens[e_idx].view(1, 1, -1)  # [1, 1, d_model]
            
            # Confidence gating: softly interpolate with missing-cue token
            gated = conf * encoded + (1.0 - conf) * missing + self.entity_type[e_idx]
            entity_rows.append(gated)
            start_idx += dim

        entities = torch.stack(entity_rows, dim=2)  # [B, 5, 4, d_model]
        entity_summary = entities.mean(dim=2)  # [B, 5, d_model]

        # 2. Compute 6 pairwise interactions per region
        left = entities[:, :, self.pair_indices[:, 0]]
        right = entities[:, :, self.pair_indices[:, 1]]
        left_conf = confidences[:, :, self.pair_indices[:, 0]]
        right_conf = confidences[:, :, self.pair_indices[:, 1]]
        pair_conf = left_conf * right_conf

        pair_input = torch.cat([
            left, right, torch.abs(left - right), left * right,
            left_conf.unsqueeze(-1), right_conf.unsqueeze(-1),
        ], dim=-1)
        pairs = self.pair_encoder(pair_input)  # [B, 5, 6, d_model]

        # Confidence-weighted pair pooling
        pair_attn = torch.softmax(
            self.pair_attention(pairs).squeeze(-1) + torch.log(pair_conf + 1e-4), dim=2
        )
        pair_summary = torch.sum(pair_attn.unsqueeze(-1) * pairs, dim=2)  # [B, 5, d_model]

        # 3. Combine into 5 regional tokens & pass through Transformer
        regions = self.region_projection(torch.cat([entity_summary, pair_summary], dim=-1))
        regions = regions + self.region_type.unsqueeze(0)
        encoded_regions = self.region_transformer(regions)  # [B, 5, d_model]

        # 4. Regional pooling & standalone physics head
        observability = confidences.mean(dim=-1)  # [B, 5]
        region_weights = torch.softmax(observability, dim=1).unsqueeze(-1)  # [B, 5, 1]
        pooled_regions = torch.sum(region_weights * encoded_regions, dim=1)  # [B, d_model]

        global_token = encoded_regions[:, 0]  # [B, d_model]
        global_summary = torch.cat([global_token, pooled_regions], dim=-1)  # [B, 2 * d_model]

        physics_logits = self.standalone_head(global_summary)  # [B, 1]

        return {
            "regional_tokens": encoded_regions,
            "global_summary": global_summary,
            "physics_logits": physics_logits,
            "observability": observability,
        }
