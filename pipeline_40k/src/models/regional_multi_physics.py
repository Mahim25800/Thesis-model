"""Regional, confidence-aware fusion of four physics-inspired entities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_entity import ENTITY_DIMS


REGION_COUNT = 5
PAIR_INDICES: Tuple[Tuple[int, int], ...] = tuple(
    (left, right)
    for left in range(len(ENTITY_DIMS))
    for right in range(left + 1, len(ENTITY_DIMS))
)


def fit_regional_standardizer(features: np.ndarray, entity_dims: Tuple[int, ...] = ENTITY_DIMS) -> Dict[str, list]:
    """Fit train-only statistics for every region and feature column."""
    values = np.asarray(features, dtype=np.float32)
    entity_dims = tuple(int(value) for value in entity_dims)
    if not entity_dims or any(value <= 0 for value in entity_dims):
        raise ValueError("Entity dimensions must be positive")
    if values.ndim != 3 or values.shape[1:] != (REGION_COUNT, sum(entity_dims)):
        raise ValueError("Expected regional feature values with the supplied entity dimensions")
    if not np.isfinite(values).all():
        raise ValueError("Regional feature values must be finite")
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-6)
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def standardize_regional(features: torch.Tensor, standardizer: Dict[str, list]) -> torch.Tensor:
    mean = torch.as_tensor(standardizer["mean"], dtype=features.dtype, device=features.device)
    scale = torch.as_tensor(standardizer["scale"], dtype=features.dtype, device=features.device)
    if mean.ndim != 2 or mean.shape[0] != REGION_COUNT or scale.shape != mean.shape:
        raise ValueError("Regional standardizer statistics must have shape [5, feature dimensions]")
    if features.ndim != 3 or features.shape[1:] != mean.shape:
        raise ValueError("Regional standardizer does not match [batch, 5, 14] input")
    return (features - mean.unsqueeze(0)) / scale.unsqueeze(0)


class RegionalMultiPhysicsHead(nn.Module):
    """Fuse entity interactions locally, then compare them across image regions."""

    def __init__(
        self, d_model: int = 48, pair_dim: int = 64, dropout: float = 0.18,
        use_entity_attention: bool = False, entity_dims: Tuple[int, ...] = ENTITY_DIMS,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pair_dim = int(pair_dim)
        self.dropout_rate = float(dropout)
        self.use_entity_attention = bool(use_entity_attention)
        self.entity_dims = tuple(int(value) for value in entity_dims)
        if not self.entity_dims or any(value <= 0 for value in self.entity_dims):
            raise ValueError("Entity dimensions must be positive")
        self.entity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 1, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.GELU(),
            )
            for dim in self.entity_dims
        ])
        self.entity_type = nn.Parameter(torch.randn(len(self.entity_dims), d_model) * 0.02)
        self.missing_tokens = nn.Parameter(torch.randn(len(self.entity_dims), d_model) * 0.02)
        self.entity_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in self.entity_dims])
        self.entity_attention = nn.Linear(d_model, 1) if self.use_entity_attention else None
        pairs = tuple(
            (left, right) for left in range(len(self.entity_dims)) for right in range(left + 1, len(self.entity_dims))
        )
        self.register_buffer("pair_indices", torch.tensor(pairs, dtype=torch.long), persistent=False)
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * d_model + 2, pair_dim), nn.LayerNorm(pair_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(pair_dim, pair_dim), nn.GELU(),
        )
        self.pair_attention = nn.Linear(pair_dim, 1)
        self.region_projection = nn.Linear(d_model + pair_dim, pair_dim)
        self.region_type = nn.Parameter(torch.randn(REGION_COUNT, pair_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pair_dim, nhead=4, dim_feedforward=2 * pair_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.region_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.region_attention = nn.Linear(pair_dim, 1)
        self.region_head = nn.Linear(pair_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * pair_dim, pair_dim), nn.LayerNorm(pair_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(pair_dim, 1),
        )

    def _validate(self, features: torch.Tensor, confidences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if confidences.ndim == 2:
            confidences = confidences.unsqueeze(0)
        if features.ndim != 3 or features.shape[1:] != (REGION_COUNT, sum(self.entity_dims)):
            raise ValueError("Expected features with shape [batch, regions, entity dimensions]")
        if confidences.shape != (len(features), REGION_COUNT, len(self.entity_dims)):
            raise ValueError("Expected confidences aligned with configured entities")
        if not torch.isfinite(features).all() or not torch.isfinite(confidences).all():
            raise ValueError("Features and confidences must be finite")
        if ((confidences < 0) | (confidences > 1)).any():
            raise ValueError("Confidences must lie in [0, 1]")
        return features, confidences

    def forward(self, features: torch.Tensor, confidences: torch.Tensor, return_details: bool = False):
        features, confidences = self._validate(features, confidences)
        entity_rows, start = [], 0
        for entity_index, (dim, encoder) in enumerate(zip(self.entity_dims, self.entity_encoders)):
            confidence = confidences[:, :, entity_index:entity_index + 1]
            encoded = encoder(torch.cat([features[:, :, start:start + dim], confidence], dim=-1))
            missing = self.missing_tokens[entity_index].view(1, 1, -1)
            entity_rows.append(confidence * encoded + (1.0 - confidence) * missing + self.entity_type[entity_index])
            start += dim
        entities = torch.stack(entity_rows, dim=2)  # [B, R, E, D]
        entity_logits = torch.cat([
            head(entities[:, :, index].mean(dim=1)) for index, head in enumerate(self.entity_heads)
        ], dim=1)

        left = entities[:, :, self.pair_indices[:, 0]]
        right = entities[:, :, self.pair_indices[:, 1]]
        left_conf = confidences[:, :, self.pair_indices[:, 0]]
        right_conf = confidences[:, :, self.pair_indices[:, 1]]
        pair_confidence = left_conf * right_conf
        pair_input = torch.cat([
            left, right, torch.abs(left - right), left * right,
            left_conf.unsqueeze(-1), right_conf.unsqueeze(-1),
        ], dim=-1)
        pairs = self.pair_encoder(pair_input)
        pair_attention = torch.softmax(
            self.pair_attention(pairs).squeeze(-1) + torch.log(pair_confidence + 1e-4), dim=2
        )
        pair_summary = torch.sum(pair_attention.unsqueeze(-1) * pairs, dim=2)
        if self.entity_attention is None:
            entity_attention = torch.full(
                entities.shape[:3], 1.0 / len(self.entity_dims), dtype=entities.dtype, device=entities.device
            )
            entity_summary = entities.mean(dim=2)
        else:
            entity_attention = torch.softmax(
                self.entity_attention(entities).squeeze(-1) + torch.log(confidences + 1e-4), dim=2
            )
            entity_summary = torch.sum(entity_attention.unsqueeze(-1) * entities, dim=2)
        regions = self.region_projection(torch.cat([entity_summary, pair_summary], dim=-1))
        regions = self.region_encoder(regions + self.region_type.unsqueeze(0))
        observability = confidences.mean(dim=2)
        region_attention = torch.softmax(
            self.region_attention(regions).squeeze(-1) + torch.log(observability + 1e-4), dim=1
        )
        attended = torch.sum(region_attention.unsqueeze(-1) * regions, dim=1)
        global_region = regions[:, 0]
        logits = self.classifier(torch.cat([global_region, attended], dim=1)).squeeze(1)
        if not return_details:
            return logits
        return {
            "logits": logits,
            "embedding": torch.cat([global_region, attended], dim=1),
            "entity_logits": entity_logits,
            "entity_attention": entity_attention,
            "region_logits": self.region_head(regions).squeeze(-1),
            "pair_attention": pair_attention,
            "region_attention": region_attention,
        }


def regional_multi_physics_loss(
    output: Dict[str, torch.Tensor], labels: torch.Tensor,
    entity_weight: float = 0.03, region_weight: float = 0.08, diversity_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    labels = labels.float().reshape(-1)
    if labels.shape != output["logits"].shape or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected binary labels aligned with logits")
    main = F.binary_cross_entropy_with_logits(output["logits"], labels)
    entity = F.binary_cross_entropy_with_logits(output["entity_logits"], labels[:, None].expand_as(output["entity_logits"]))
    region = F.binary_cross_entropy_with_logits(output["region_logits"], labels[:, None].expand_as(output["region_logits"]))
    attention = output["region_attention"].clamp_min(1e-8)
    normalized_entropy = -(attention * attention.log()).sum(1).mean() / np.log(REGION_COUNT)
    diversity = 1.0 - normalized_entropy
    total = main + entity_weight * entity + region_weight * region + diversity_weight * diversity
    return {"total": total, "main": main, "entity": entity, "region": region, "diversity": diversity}
