"""Single-image inference for the frozen regional multi-physics v14 ensemble.

The class in this module deliberately performs no fitting, threshold search, or
calibration.  It validates the recorded branch hashes before it makes a score,
then applies the policy embedded in the frozen v14 artifact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.extractors.regional_physics import REGION_NAMES, RegionalPhysicsExtractor
from src.extractors.surface_normals import SurfaceNormalsExtractor
from src.models.improved import probability_logits
from src.utils.decision_policy import apply_policy_calibration


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "physics_ensemble_v14" / "model.pt"
ENTITY_NAMES = ("illumination", "specular_optics", "surface_normals", "chromatic_shadows")
FEATURE_SCHEMA = "regional_physics_dsine_v2_5x14_features_5x4_confidences"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V14PhysicsPredictor:
    """Load and run the exact v14 ensemble on one RGB image.

    ``device`` defaults to CUDA when it is available, but callers can choose
    CPU for a local demonstration while a training process occupies the GPU.
    """

    def __init__(self, model_path: Path = MODEL_PATH, device: str | None = None):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_relative_to(ROOT):
            raise ValueError("The v14 model path must remain inside pipeline_40k")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.artifact, self.branches = self._load_ensemble()
        normals = SurfaceNormalsExtractor(normal_backend="dsine", dsine_device=str(self.device))
        self.extractor = RegionalPhysicsExtractor(normals)

    def _load_ensemble(self) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any], list[Any]]]]:
        # Reuse the released branch loader: it reconstructs the stored regional
        # architecture and applies its train-only standardizer exactly.
        from scripts.select_physics_ensemble_v7 import load_branch

        artifact = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if artifact.get("model_type") != "physics_probability_ensemble":
            raise ValueError("Expected a frozen physics probability ensemble")
        if artifact.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("The supplied ensemble has an incompatible feature schema")
        branches = []
        for specification in artifact.get("branches", []):
            branch_path = (ROOT / specification["path"]).resolve()
            if not branch_path.is_relative_to(ROOT):
                raise ValueError("A branch reference points outside pipeline_40k")
            if _sha256(branch_path) != specification["sha256"]:
                raise ValueError(f"Frozen branch hash mismatch: {branch_path}")
            branch_artifact, models = load_branch(branch_path, self.device)
            branches.append((specification, branch_artifact, models))
        if not branches or not any(float(specification["weight"]) > 0 for specification, _, _ in branches):
            raise ValueError("The ensemble has no active branch")
        return artifact, branches

    def _branch_probabilities(self, features: torch.Tensor, confidences: torch.Tensor) -> dict[str, float]:
        from scripts.select_physics_ensemble_v7 import branch_probability

        probabilities: dict[str, float] = {}
        for specification, artifact, models in self.branches:
            value = branch_probability(artifact, models, features, confidences, self.device)
            probabilities[str(specification["name"])] = float(value[0])
        return probabilities

    def predict(self, image: Any) -> dict[str, Any]:
        """Return the calibrated fake probability and descriptive diagnostics."""
        features, confidences = self.extractor.extract(image)
        batched_features = features.unsqueeze(0)
        batched_confidences = confidences.unsqueeze(0)
        branch_probabilities = self._branch_probabilities(batched_features, batched_confidences)
        raw_probability = sum(
            float(specification["weight"]) * branch_probabilities[str(specification["name"])]
            for specification, _, _ in self.branches
        )
        calibrated_probability = float(apply_policy_calibration(
            probability_logits(np.asarray([raw_probability], dtype=np.float64)), self.artifact["policy"]
        )[0])
        policy = self.artifact["policy"]
        # The frozen policy was fitted on per-entity confidences after they
        # were averaged across the five regions, then summed.  Preserve that
        # scale here instead of comparing a 0..1 global mean to its 0..4 gate.
        observability = float(confidences.mean(dim=0).sum().item())
        if observability < float(policy["obs_threshold"]):
            decision = "insufficient physical observability"
        elif calibrated_probability >= float(policy["tau_high"]):
            decision = "likely AI-generated"
        elif calibrated_probability <= float(policy["tau_low"]):
            decision = "likely camera-origin"
        else:
            decision = "uncertain"
        return {
            "model": "physics_ensemble_v14",
            "probability_label": policy.get("probability_label", "fake"),
            "fake_probability": calibrated_probability,
            "raw_ensemble_probability": float(raw_probability),
            "decision": decision,
            "decision_threshold": float(policy["decision_threshold"]),
            "uncertainty_band": [float(policy["tau_low"]), float(policy["tau_high"])],
            "aggregate_entity_observability": observability,
            "entity_observability": {
                name: float(confidences[:, index].mean().item())
                for index, name in enumerate(ENTITY_NAMES)
            },
            "region_observability": {
                name: float(confidences[index].mean().item())
                for index, name in enumerate(REGION_NAMES)
            },
            "branch_probabilities": branch_probabilities,
            "extractor": self.extractor.provenance(),
        }
