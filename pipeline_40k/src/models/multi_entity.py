"""Confidence-aware fusion of multiple physics-inspired image entities.

The historical cache supplies four entity groups: illumination, specular
appearance, surface-normal residuals, and chromatic shadows.  This module
does not call them physical laws.  It learns whether the available entity
descriptors agree with one another, while retaining each entity's confidence
so an unavailable cue cannot be treated as evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ENTITY_DIMS: Tuple[int, ...] = (5, 4, 3, 2)
ENTITY_NAMES: Tuple[str, ...] = (
    "illumination",
    "specular_optics",
    "surface_normals",
    "chromatic_shadows",
)


def fit_feature_standardizer(features: np.ndarray) -> Dict[str, list]:
    """Fit a finite, train-only standardizer for 14 cached feature columns."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != sum(ENTITY_DIMS) or not np.isfinite(values).all():
        raise ValueError("Expected finite [samples, 14] feature values")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def standardize_features(features: torch.Tensor, standardizer: Dict[str, list]) -> torch.Tensor:
    """Apply a saved standardizer without refitting on development or test data."""
    mean = torch.as_tensor(standardizer["mean"], dtype=features.dtype, device=features.device)
    scale = torch.as_tensor(standardizer["scale"], dtype=features.dtype, device=features.device)
    if features.shape[-1] != len(mean) or len(mean) != len(scale):
        raise ValueError("Standardizer does not match cached feature schema")
    return (features - mean) / scale


class MultiEntityConsistencyHead(nn.Module):
    """Fuse four entity summaries through learned pairwise consistency tokens.

    Every entity is first encoded independently.  Six pair tokens then model
    agreement/disagreement between entities.  Confidence gates entity tokens,
    and pair attention is downweighted when either contributing cue has low
    confidence.  Auxiliary entity and pair logits make every branch receive a
    classification signal during training; they are not exposed as claims that
    an individual feature is a verified physical test.
    """

    def __init__(self, d_model: int = 48, pair_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        if d_model < 8 or pair_dim < 8 or not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid multi-entity model dimensions or dropout")
        self.d_model = int(d_model)
        self.pair_dim = int(pair_dim)
        self.dropout_rate = float(dropout)
        self.entity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 1, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.GELU(),
            )
            for dim in ENTITY_DIMS
        ])
        self.entity_type = nn.Parameter(torch.randn(len(ENTITY_DIMS), d_model) * 0.02)
        self.missing_tokens = nn.Parameter(torch.randn(len(ENTITY_DIMS), d_model) * 0.02)
        self.entity_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in ENTITY_DIMS])

        pairs = [(left, right) for left in range(len(ENTITY_DIMS)) for right in range(left + 1, len(ENTITY_DIMS))]
        self.register_buffer("pair_indices", torch.tensor(pairs, dtype=torch.long), persistent=False)
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * d_model + 2, pair_dim), nn.LayerNorm(pair_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(pair_dim, pair_dim), nn.GELU(),
        )
        self.pair_attention = nn.Linear(pair_dim, 1)
        self.pair_head = nn.Linear(pair_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + pair_dim, pair_dim), nn.LayerNorm(pair_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(pair_dim, 1),
        )

    def _validate(self, features: torch.Tensor, confidences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if confidences.ndim == 1:
            confidences = confidences.unsqueeze(0)
        if features.ndim != 2 or confidences.ndim != 2 or features.shape != (len(confidences), sum(ENTITY_DIMS)):
            raise ValueError("Expected aligned [batch, 14] features and [batch, 4] confidences")
        if confidences.shape[1] != len(ENTITY_DIMS):
            raise ValueError("Expected four entity confidence values")
        if not torch.isfinite(features).all() or not torch.isfinite(confidences).all():
            raise ValueError("Features and confidences must be finite")
        if ((confidences < 0) | (confidences > 1)).any():
            raise ValueError("Confidences must lie in [0, 1]")
        return features, confidences

    def forward(self, features: torch.Tensor, confidences: torch.Tensor, return_details: bool = False):
        features, confidences = self._validate(features, confidences)
        entities = []
        start = 0
        for entity_index, (dim, encoder) in enumerate(zip(ENTITY_DIMS, self.entity_encoders)):
            confidence = confidences[:, entity_index:entity_index + 1]
            encoded = encoder(torch.cat([features[:, start:start + dim], confidence], dim=1))
            token = confidence * encoded + (1.0 - confidence) * self.missing_tokens[entity_index]
            entities.append(token + self.entity_type[entity_index])
            start += dim
        entity_tokens = torch.stack(entities, dim=1)
        entity_logits = torch.cat([head(entity_tokens[:, index]).squeeze(1).unsqueeze(1)
                                   for index, head in enumerate(self.entity_heads)], dim=1)

        left = entity_tokens[:, self.pair_indices[:, 0]]
        right = entity_tokens[:, self.pair_indices[:, 1]]
        pair_confidence = confidences[:, self.pair_indices[:, 0]] * confidences[:, self.pair_indices[:, 1]]
        pair_input = torch.cat([
            left, right, torch.abs(left - right), left * right,
            confidences[:, self.pair_indices[:, 0]].unsqueeze(-1),
            confidences[:, self.pair_indices[:, 1]].unsqueeze(-1),
        ], dim=-1)
        pair_tokens = self.pair_encoder(pair_input)
        attention_logits = self.pair_attention(pair_tokens).squeeze(-1) + torch.log(pair_confidence + 1e-4)
        pair_attention = torch.softmax(attention_logits, dim=1)
        pair_summary = torch.sum(pair_attention.unsqueeze(-1) * pair_tokens, dim=1)
        entity_summary = torch.mean(entity_tokens, dim=1)
        logits = self.classifier(torch.cat([entity_summary, pair_summary], dim=1)).squeeze(1)

        if not return_details:
            return logits
        return {
            "logits": logits,
            "entity_logits": entity_logits,
            "pair_logits": self.pair_head(pair_tokens).squeeze(-1),
            "pair_attention": pair_attention,
            "pair_confidence": pair_confidence,
        }


def multi_entity_loss(output: Dict[str, torch.Tensor], labels: torch.Tensor,
                      entity_weight: float = 0.10, pair_weight: float = 0.10,
                      diversity_weight: float = 0.01) -> Dict[str, torch.Tensor]:
    """Classification loss plus weak supervision for all entity interactions."""
    labels = labels.float().reshape(-1)
    if labels.shape != output["logits"].shape or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected binary labels aligned with multi-entity logits")
    targets = labels.unsqueeze(1)
    main = F.binary_cross_entropy_with_logits(output["logits"], labels)
    entity = F.binary_cross_entropy_with_logits(output["entity_logits"], targets.expand_as(output["entity_logits"]))
    pair = F.binary_cross_entropy_with_logits(output["pair_logits"], targets.expand_as(output["pair_logits"]))
    attention = output["pair_attention"].clamp_min(1e-8)
    normalized_entropy = -(attention * attention.log()).sum(dim=1).mean() / np.log(attention.shape[1])
    diversity = 1.0 - normalized_entropy
    total = main + entity_weight * entity + pair_weight * pair + diversity_weight * diversity
    return {"total": total, "main": main, "entity": entity, "pair": pair, "diversity": diversity}


class MultiEntityCachedClassifier:
    """Apply the v3 model to an audited cache with the historical 14+4 schema.

    The model intentionally accepts cached descriptors only.  It cannot provide
    valid raw-image inference until the normal and highlight extractors are
    regenerated under a versioned reproducible recipe.
    """

    def __init__(self, artifact_path, v2_artifact_path=None, legacy_checkpoint=None):
        from .cross_gen_gated import TransformerPhysicsCrossGenHead
        from .improved import legacy_cache_matrix, probability_logits
        from ..utils.decision_policy import apply_decision_policy

        self._legacy_cache_matrix = legacy_cache_matrix
        self._probability_logits = probability_logits
        self._apply_decision_policy = apply_decision_policy
        self.artifact_path = Path(artifact_path)
        self.payload = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        if self.payload.get("feature_schema") != "legacy_cache_14_plus_4_v1":
            raise ValueError("Artifact is not compatible with the audited legacy cache")
        if tuple(self.payload.get("entity_dims", ())) != ENTITY_DIMS:
            raise ValueError("Artifact entity schema differs from the installed model")
        v2_path = Path(v2_artifact_path) if v2_artifact_path else self.artifact_path.parent.parent / "improved_v2/model.joblib"
        expected_v2_hash = self.payload["v2_artifact_sha256"]
        if hashlib.sha256(v2_path.read_bytes()).hexdigest() != expected_v2_hash:
            raise ValueError("v2 artifact does not match the frozen v3 blend")
        self.v2 = joblib.load(v2_path)
        checkpoint = Path(legacy_checkpoint) if legacy_checkpoint else self.artifact_path.parent.parent / "gated_cross_gen_40k_best.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.payload["legacy_checkpoint_sha256"]:
            raise ValueError("Legacy checkpoint does not match the frozen v3 blend")
        self.legacy = TransformerPhysicsCrossGenHead().eval()
        self.legacy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        architecture = self.payload["architecture"]
        self.models = []
        for state in self.payload["model_states"]:
            model = MultiEntityConsistencyHead(**architecture).eval()
            model.load_state_dict(state)
            self.models.append(model)
        self.standardizer = self.payload["standardizer"]
        self.entity_weight = float(self.payload["entity_weight"])
        self.policy = self.payload["policy"]

    def predict_features(self, features, confidences):
        features_t = torch.as_tensor(features, dtype=torch.float32)
        confidences_t = torch.as_tensor(confidences, dtype=torch.float32)
        if features_t.ndim == 1:
            features_t = features_t.unsqueeze(0)
        if confidences_t.ndim == 1:
            confidences_t = confidences_t.unsqueeze(0)
        # Validate through the model before feeding cached values into either branch.
        standardized = standardize_features(features_t, self.standardizer)
        with torch.inference_mode():
            entity = np.mean([torch.sigmoid(model(standardized, confidences_t)).numpy() for model in self.models], axis=0)
            legacy = torch.sigmoid(self.legacy(features_t, confidences_t)[0]).numpy()
        tree = self.v2["estimator"].predict_proba(self._legacy_cache_matrix(features_t, confidences_t))[:, 1]
        v2_raw = float(self.v2["blend_weight"]) * tree + (1.0 - float(self.v2["blend_weight"])) * legacy
        raw = self.entity_weight * entity + (1.0 - self.entity_weight) * v2_raw
        return self._apply_decision_policy(self._probability_logits(raw), confidences_t.numpy(), self.policy)
