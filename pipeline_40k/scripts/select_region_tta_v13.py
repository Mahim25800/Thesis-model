"""Select region-flip test-time averaging on development data only."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_physics_ensemble_v7 import branch_probability, load_branch, sha256
from scripts.train_regional_multi_physics_v5 import (
    load_cache, read_manifest, split_train_development_calibration, write_json,
)
from src.utils.metrics import evaluate_predictions


BASES = (
    ("regional_v5_tta", ROOT / "models/regional_multi_physics_v5/model.pt"),
    ("generator_invariant_v6_tta", ROOT / "models/generator_invariant_v6/model.pt"),
)
MODES = {
    "none": [[0, 1, 2, 3, 4]],
    "horizontal": [[0, 1, 2, 3, 4], [0, 2, 1, 4, 3]],
    "vertical": [[0, 1, 2, 3, 4], [0, 3, 4, 1, 2]],
    "horizontal_vertical": [
        [0, 1, 2, 3, 4], [0, 2, 1, 4, 3], [0, 3, 4, 1, 2], [0, 4, 3, 2, 1],
    ],
}


def main() -> None:
    output = ROOT / "models/region_tta_v13"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("region_tta_v13 output already exists")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    train_features, train_confidences, labels_t = load_cache(
        ROOT / "data/cache_regional_dsine_v2/train_features.pt"
    )
    manifest = read_manifest(ROOT / "data/cache_regional_dsine_v2/train_manifest.jsonl")
    labels = labels_t.numpy().astype(int)
    groups = np.asarray([row["difference_hash_64"] for row in manifest])
    _, development, _ = split_train_development_calibration(labels, groups)
    wukong = torch.load(
        ROOT / "data/cache_genimage_regional_v2/wukong_features.pt",
        map_location="cpu", weights_only=True,
    )
    wukong_labels = wukong["labels"].numpy().astype(int)
    report = {"protocol": "TTA selected on internal development and Wukong only", "branches": {}}

    for output_name, source_path in BASES:
        artifact, models = load_branch(source_path, device)
        candidates = []
        for mode, permutations in MODES.items():
            candidate = deepcopy(artifact)
            candidate["tta_region_permutations"] = permutations
            internal_probability = branch_probability(
                candidate, models, train_features[development], train_confidences[development], device
            )
            wukong_probability = branch_probability(
                candidate, models, wukong["features"], wukong["confidences"], device
            )
            internal = evaluate_predictions(labels[development], internal_probability, 0.5)
            external = evaluate_predictions(wukong_labels, wukong_probability, 0.5)
            candidates.append({
                "mode": mode, "permutations": permutations,
                "internal_auc": internal["auc_roc"], "wukong_auc": external["auc_roc"],
                "selection_score": [
                    min(internal["auc_roc"], external["auc_roc"]),
                    0.5 * (internal["auc_roc"] + external["auc_roc"]),
                ],
            })
        selected = max(candidates, key=lambda row: tuple(row["selection_score"]))
        result = deepcopy(artifact)
        result["tta_region_permutations"] = selected["permutations"]
        result["artifact_role"] = "development-selected TTA ensemble branch"
        result["tta_selection"] = selected
        result["source_model"] = {
            "path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(source_path),
        }
        branch_dir = output / output_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result, branch_dir / "model.pt")
        write_json(branch_dir / "selection.json", {"selected": selected, "candidates": candidates})
        report["branches"][output_name] = selected
    report["confirmatory_data_read"] = False
    write_json(output / "selection.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
