"""Select and calibrate a physics-only v5/v6 ensemble on held-out Wukong."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_scaled import evaluate_selective_abstention
from scripts.train_regional_multi_physics_v5 import (
    load_cache,
    probabilities,
    read_manifest,
    split_train_development_calibration,
    write_json,
)
from src.models.improved import probability_logits
from src.models.multi_entity import ENTITY_NAMES
from src.models.regional_multi_physics import RegionalMultiPhysicsHead
from src.utils.decision_policy import apply_policy_calibration, fit_decision_policy
from src.utils.metrics import evaluate_predictions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_branch(path: Path, device: torch.device):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    models = []
    for state in artifact["model_states"]:
        model = RegionalMultiPhysicsHead(**artifact["architecture"]).to(device).eval()
        model.load_state_dict(state)
        models.append(model)
    return artifact, models


def branch_probability(artifact: dict, models: list[RegionalMultiPhysicsHead], features, confidences, device):
    disabled = artifact.get("disabled_entities", [])
    if disabled:
        confidences = confidences.clone()
        indices = [ENTITY_NAMES.index(name) for name in disabled]
        confidences[:, :, indices] = 0.0
    permutations = artifact.get("tta_region_permutations", [[0, 1, 2, 3, 4]])
    rows = []
    for permutation in permutations:
        if sorted(permutation) != list(range(5)):
            raise ValueError(f"Invalid TTA region permutation: {permutation}")
        permuted_features = features[:, permutation]
        permuted_confidences = confidences[:, permutation]
        rows.extend(
            probabilities(model, permuted_features, permuted_confidences, artifact["standardizer"], device)
            for model in models
        )
    return np.mean(rows, axis=0)


def main() -> None:
    cache_dir = ROOT / "data/cache_regional_dsine_v2"
    wukong_path = ROOT / "data/cache_genimage_regional_v2/wukong_features.pt"
    branch_paths = [
        ROOT / "models/regional_multi_physics_v5/model.pt",
        ROOT / "models/generator_invariant_v6/model.pt",
    ]
    output = ROOT / "models/physics_ensemble_v7"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("physics_ensemble_v7 output already exists")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branches = [load_branch(path, device) for path in branch_paths]
    train_features, train_confidences, train_labels_t = load_cache(cache_dir / "train_features.pt")
    test_features, test_confidences, test_labels_t = load_cache(cache_dir / "test_unseen_features.pt")
    manifest = read_manifest(cache_dir / "train_manifest.jsonl")
    test_manifest = read_manifest(cache_dir / "test_unseen_manifest.jsonl")
    labels = train_labels_t.numpy().astype(int)
    groups = np.asarray([row["difference_hash_64"] for row in manifest])
    fit_indices, development_indices, calibration_indices = split_train_development_calibration(labels, groups)
    wukong = torch.load(wukong_path, map_location="cpu", weights_only=True)
    wukong_labels = wukong["labels"].numpy().astype(int)

    development_probabilities = [
        branch_probability(artifact, models, train_features[development_indices], train_confidences[development_indices], device)
        for artifact, models in branches
    ]
    wukong_probabilities = [
        branch_probability(artifact, models, wukong["features"], wukong["confidences"], device)
        for artifact, models in branches
    ]
    candidates = []
    for v5_weight in np.linspace(0.0, 1.0, 41):
        internal_probability = v5_weight * development_probabilities[0] + (1.0 - v5_weight) * development_probabilities[1]
        external_probability = v5_weight * wukong_probabilities[0] + (1.0 - v5_weight) * wukong_probabilities[1]
        internal = evaluate_predictions(labels[development_indices], internal_probability, 0.5)
        external = evaluate_predictions(wukong_labels, external_probability, 0.5)
        candidates.append({
            "v5_weight": float(v5_weight), "v6_weight": float(1.0 - v5_weight),
            "internal_auc": internal["auc_roc"], "wukong_auc": external["auc_roc"],
            "selection_score": [min(internal["auc_roc"], external["auc_roc"]),
                                0.5 * (internal["auc_roc"] + external["auc_roc"])],
        })
    selected = max(candidates, key=lambda row: tuple(row["selection_score"]))
    weight = selected["v5_weight"]

    calibration_probabilities = [
        branch_probability(artifact, models, train_features[calibration_indices], train_confidences[calibration_indices], device)
        for artifact, models in branches
    ]
    calibration_probability = weight * calibration_probabilities[0] + (1.0 - weight) * calibration_probabilities[1]
    policy = fit_decision_policy(
        probability_logits(calibration_probability), labels[calibration_indices],
        obs_threshold=1.0, abstention_margin=0.10,
    )
    artifact = {
        "schema_version": 1,
        "feature_schema": branches[0][0]["feature_schema"],
        "model_type": "physics_probability_ensemble",
        "branches": [
            {"name": "regional_v5", "path": str(branch_paths[0].relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha256(branch_paths[0]), "weight": weight},
            {"name": "generator_invariant_v6", "path": str(branch_paths[1].relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha256(branch_paths[1]), "weight": 1.0 - weight},
        ],
        "selection_domain": "GenImage Wukong held out from both branch gradient updates",
        "selection": selected,
        "policy": policy,
    }
    torch.save(artifact, output / "model.pt")
    write_json(output / "selection.json", {
        "protocol": "blend selected only on group-disjoint internal development and held-out Wukong; Synthbuster unread",
        "selected": selected, "candidates": candidates,
        "wukong": evaluate_predictions(
            wukong_labels, weight * wukong_probabilities[0] + (1.0 - weight) * wukong_probabilities[1], 0.5
        ),
        "internal_development": evaluate_predictions(
            labels[development_indices], weight * development_probabilities[0] + (1.0 - weight) * development_probabilities[1], 0.5
        ),
    })
    write_json(output / "policy.json", policy)
    test_groups = np.asarray([row["difference_hash_64"] for row in test_manifest])
    test_indices = np.flatnonzero(~np.isin(test_groups, np.unique(groups)))
    test_labels = test_labels_t.numpy().astype(int)
    test_branch_probabilities = [
        branch_probability(artifact_value, models, test_features[test_indices], test_confidences[test_indices], device)
        for artifact_value, models in branches
    ]
    raw = weight * test_branch_probabilities[0] + (1.0 - weight) * test_branch_probabilities[1]
    calibrated = apply_policy_calibration(probability_logits(raw), policy)
    write_json(output / "evaluation_internal.json", {
        "protocol": "frozen v7 ensemble on internal unseen test",
        "binary": evaluate_predictions(test_labels[test_indices], calibrated, policy["decision_threshold"]),
        "raw_threshold_0_5": evaluate_predictions(test_labels[test_indices], raw, 0.5),
        "selective": evaluate_selective_abstention(
            test_labels[test_indices], calibrated, test_confidences[test_indices].mean(dim=1).numpy(),
            policy["tau_low"], policy["tau_high"], policy["obs_threshold"],
        ),
    })
    print(json.dumps({"selected": selected, "policy": policy}, indent=2), flush=True)


if __name__ == "__main__":
    main()
