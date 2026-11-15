"""Select a fixed physics-entity mask on development data only.

This creates an ensemble-branch artifact. It does not read Synthbuster or the
confirmatory GenImage cache and does not refit any network parameters.
"""

from __future__ import annotations

import itertools
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
from src.models.multi_entity import ENTITY_NAMES
from src.utils.metrics import evaluate_predictions


BASES = (
    ("regional_v5", ROOT / "models/regional_multi_physics_v5/model.pt"),
    ("generator_invariant_v6", ROOT / "models/generator_invariant_v6/model.pt"),
)


def main() -> None:
    output = ROOT / "models/entity_mask_variant_v12"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("entity_mask_variant_v12 output already exists")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    cache = ROOT / "data/cache_regional_dsine_v2"
    train_features, train_confidences, labels_t = load_cache(cache / "train_features.pt")
    manifest = read_manifest(cache / "train_manifest.jsonl")
    labels = labels_t.numpy().astype(int)
    groups = np.asarray([row["difference_hash_64"] for row in manifest])
    _, development, _ = split_train_development_calibration(labels, groups)
    wukong = torch.load(
        ROOT / "data/cache_genimage_regional_v2/wukong_features.pt",
        map_location="cpu", weights_only=True,
    )
    wukong_labels = wukong["labels"].numpy().astype(int)

    candidates = []
    loaded = {}
    masks = [()]
    masks.extend(itertools.combinations(ENTITY_NAMES, 1))
    masks.extend(itertools.combinations(ENTITY_NAMES, 2))
    for base_name, path in BASES:
        base_artifact, models = load_branch(path, device)
        loaded[base_name] = (path, base_artifact, models)
        for disabled in masks:
            candidate_artifact = deepcopy(base_artifact)
            candidate_artifact["disabled_entities"] = list(disabled)
            internal_probability = branch_probability(
                candidate_artifact, models, train_features[development],
                train_confidences[development], device,
            )
            wukong_probability = branch_probability(
                candidate_artifact, models, wukong["features"], wukong["confidences"], device,
            )
            internal = evaluate_predictions(labels[development], internal_probability, 0.5)
            external = evaluate_predictions(wukong_labels, wukong_probability, 0.5)
            candidates.append({
                "base": base_name,
                "disabled_entities": list(disabled),
                "internal_auc": internal["auc_roc"],
                "wukong_auc": external["auc_roc"],
                "selection_score": [
                    min(internal["auc_roc"], external["auc_roc"]),
                    0.5 * (internal["auc_roc"] + external["auc_roc"]),
                ],
            })
    selected = max(candidates, key=lambda row: tuple(row["selection_score"]))
    source_path, source_artifact, _ = loaded[selected["base"]]
    result = deepcopy(source_artifact)
    result["disabled_entities"] = selected["disabled_entities"]
    result["artifact_role"] = "development-selected fixed-mask ensemble branch"
    result["mask_selection"] = selected
    result["source_model"] = {
        "path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(source_path),
    }
    torch.save(result, output / "model.pt")
    write_json(output / "selection.json", {
        "protocol": "fixed entity mask selected on group-disjoint internal development and Wukong only",
        "selected": selected,
        "candidates": candidates,
        "confirmatory_data_read": False,
    })
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
