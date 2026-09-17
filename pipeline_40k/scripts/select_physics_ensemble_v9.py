"""Select a frozen v5/v6/v8 physics ensemble using development data only.

The weight grid is evaluated on the original group-disjoint development split
and GenImage Wukong. Synthbuster and the confirmatory GenImage GLIDE set are
never read here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_scaled import evaluate_selective_abstention
from scripts.select_physics_ensemble_v7 import branch_probability, load_branch, sha256
from scripts.train_regional_multi_physics_v5 import (
    load_cache,
    read_manifest,
    split_train_development_calibration,
    write_json,
)
from src.models.improved import probability_logits
from src.utils.decision_policy import apply_policy_calibration, fit_decision_policy
from src.utils.metrics import evaluate_predictions


BRANCH_PATHS = (
    ("regional_v5", ROOT / "models/regional_multi_physics_v5/model.pt"),
    ("generator_invariant_v6", ROOT / "models/generator_invariant_v6/model.pt"),
    ("reliability_gated_v8", ROOT / "models/reliability_gated_v8/model.pt"),
)


def probability_mix(probabilities: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.sum(np.stack(probabilities, axis=0) * weights[:, None], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branches", nargs="+",
        default=[f"{name}={path.relative_to(ROOT)}" for name, path in BRANCH_PATHS],
        help="Branch specifications formatted as name=path/to/model.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/physics_ensemble_v9")
    args = parser.parse_args()
    branch_paths = []
    for specification in args.branches:
        if "=" not in specification:
            raise ValueError(f"Invalid branch specification: {specification}")
        name, raw_path = specification.split("=", 1)
        path = (ROOT / raw_path).resolve()
        if not name or not path.is_relative_to(ROOT):
            raise ValueError(f"Invalid branch specification: {specification}")
        branch_paths.append((name, path))
    if len(branch_paths) != 3 or len({name for name, _ in branch_paths}) != 3:
        raise ValueError("Exactly three uniquely named physics branches are required")
    cache_dir = ROOT / "data/cache_regional_dsine_v2"
    wukong_path = ROOT / "data/cache_genimage_regional_v2/wukong_features.pt"
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain inside pipeline_40k")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("physics_ensemble_v9 output already exists")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [(name, path, *load_branch(path, device)) for name, path in branch_paths]
    train_features, train_confidences, train_labels_t = load_cache(cache_dir / "train_features.pt")
    test_features, test_confidences, test_labels_t = load_cache(cache_dir / "test_unseen_features.pt")
    manifest = read_manifest(cache_dir / "train_manifest.jsonl")
    test_manifest = read_manifest(cache_dir / "test_unseen_manifest.jsonl")
    labels = train_labels_t.numpy().astype(int)
    groups = np.asarray([row["difference_hash_64"] for row in manifest])
    _, development_indices, calibration_indices = split_train_development_calibration(labels, groups)
    wukong = torch.load(wukong_path, map_location="cpu", weights_only=True)
    wukong_labels = wukong["labels"].numpy().astype(int)

    development_probabilities = [
        branch_probability(artifact, models, train_features[development_indices],
                           train_confidences[development_indices], device)
        for _, _, artifact, models in loaded
    ]
    wukong_probabilities = [
        branch_probability(artifact, models, wukong["features"], wukong["confidences"], device)
        for _, _, artifact, models in loaded
    ]

    candidates = []
    grid = np.linspace(0.0, 1.0, 21)
    for first, second in itertools.product(grid, repeat=2):
        third = 1.0 - first - second
        if third < -1e-9:
            continue
        weights = np.asarray([first, second, max(third, 0.0)], dtype=np.float64)
        internal = evaluate_predictions(
            labels[development_indices], probability_mix(development_probabilities, weights), 0.5
        )
        external = evaluate_predictions(
            wukong_labels, probability_mix(wukong_probabilities, weights), 0.5
        )
        candidates.append({
            "weights": {loaded[index][0]: float(value) for index, value in enumerate(weights)},
            "internal_auc": internal["auc_roc"],
            "wukong_auc": external["auc_roc"],
            "selection_score": [
                min(internal["auc_roc"], external["auc_roc"]),
                0.5 * (internal["auc_roc"] + external["auc_roc"]),
            ],
        })
    selected = max(candidates, key=lambda row: tuple(row["selection_score"]))
    weights = np.asarray([selected["weights"][name] for name, *_ in loaded])

    calibration_probabilities = [
        branch_probability(artifact, models, train_features[calibration_indices],
                           train_confidences[calibration_indices], device)
        for _, _, artifact, models in loaded
    ]
    calibration_probability = probability_mix(calibration_probabilities, weights)
    policy = fit_decision_policy(
        probability_logits(calibration_probability), labels[calibration_indices],
        obs_threshold=1.0, abstention_margin=0.10,
    )
    artifact = {
        "schema_version": 1,
        "feature_schema": loaded[0][2]["feature_schema"],
        "model_type": "physics_probability_ensemble",
        "branches": [
            {
                "name": name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "weight": float(weights[index]),
            }
            for index, (name, path, _, _) in enumerate(loaded)
        ],
        "selection_domain": "group-disjoint internal development and held-out GenImage Wukong",
        "selection": selected,
        "policy": policy,
    }
    torch.save(artifact, output / "model.pt")
    write_json(output / "selection.json", {
        "protocol": "weights selected without reading Synthbuster or confirmatory GenImage GLIDE",
        "selected": selected,
        "candidates": candidates,
        "wukong": evaluate_predictions(
            wukong_labels, probability_mix(wukong_probabilities, weights), 0.5
        ),
        "internal_development": evaluate_predictions(
            labels[development_indices], probability_mix(development_probabilities, weights), 0.5
        ),
    })
    write_json(output / "policy.json", policy)

    test_groups = np.asarray([row["difference_hash_64"] for row in test_manifest])
    test_indices = np.flatnonzero(~np.isin(test_groups, np.unique(groups)))
    test_labels = test_labels_t.numpy().astype(int)
    test_probabilities = [
        branch_probability(artifact_value, models, test_features[test_indices],
                           test_confidences[test_indices], device)
        for _, _, artifact_value, models in loaded
    ]
    raw = probability_mix(test_probabilities, weights)
    calibrated = apply_policy_calibration(probability_logits(raw), policy)
    write_json(output / "evaluation_internal.json", {
        "protocol": "frozen v9 ensemble on the internal unseen test",
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
