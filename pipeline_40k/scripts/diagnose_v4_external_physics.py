"""Post-hoc v4 entity diagnostics on frozen Synthbuster caches.

This report explains a completed evaluation. It must not be used for model or
threshold selection because Synthbuster is the final external benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.multi_entity import ENTITY_DIMS, ENTITY_NAMES, MultiEntityConsistencyHead, standardize_features


def entity_ranges() -> dict[str, slice]:
    result, start = {}, 0
    for name, width in zip(ENTITY_NAMES, ENTITY_DIMS):
        result[name] = slice(start, start + width)
        start += width
    return result


def main() -> None:
    model_path = ROOT / "models/dsine_multi_entity_v4/model.pt"
    internal_path = ROOT / "data/cache_regenerated_dsine_v1/train_features.pt"
    external_dir = ROOT / "models/dsine_multi_entity_v4/external_synthbuster/cache"
    output_path = ROOT / "models/dsine_multi_entity_v4/external_synthbuster/entity_diagnostics.json"
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    internal = torch.load(internal_path, map_location="cpu", weights_only=True)
    models = []
    for state in artifact["model_states"]:
        model = MultiEntityConsistencyHead(**artifact["architecture"]).eval()
        model.load_state_dict(state)
        models.append(model)
    real_mask = internal["labels"] == 0
    internal_real = standardize_features(internal["features"][real_mask], artifact["standardizer"]).numpy()
    raise_cache = torch.load(external_dir / "raise_real.pt", map_location="cpu", weights_only=True)
    raise_values = standardize_features(raise_cache["features"], artifact["standardizer"]).numpy()
    shifts = {}
    for name, region in entity_ranges().items():
        internal_mean = internal_real[:, region].mean(axis=0)
        raise_mean = raise_values[:, region].mean(axis=0)
        shifts[name] = {
            "mean_absolute_standardized_shift": float(np.mean(np.abs(raise_mean - internal_mean))),
            "maximum_absolute_standardized_shift": float(np.max(np.abs(raise_mean - internal_mean))),
        }
    generators = {}
    for path in sorted(external_dir.glob("*.pt")):
        if path.stem == "raise_real":
            continue
        fake = torch.load(path, map_location="cpu", weights_only=True)
        features = torch.cat([raise_cache["features"], fake["features"]])
        confidences = torch.cat([raise_cache["confidences"], fake["confidences"]])
        labels = np.concatenate([np.zeros(len(raise_cache["features"])), np.ones(len(fake["features"]))])
        values = standardize_features(features, artifact["standardizer"])
        with torch.inference_mode():
            details = [model(values, confidences, return_details=True) for model in models]
        entity_logits = np.mean([item["entity_logits"].numpy() for item in details], axis=0)
        pair_attention = np.mean([item["pair_attention"].numpy() for item in details], axis=0)
        generators[path.stem] = {
            "entity_auc": {
                name: float(roc_auc_score(labels, entity_logits[:, index]))
                for index, name in enumerate(ENTITY_NAMES)
            },
            "mean_pair_attention": pair_attention.mean(axis=0).tolist(),
        }
    report = {
        "role": "post-hoc explanation only; prohibited for model/threshold selection",
        "real_domain_shift_coco_to_raise": shifts,
        "generators": generators,
        "pair_order": [
            [ENTITY_NAMES[left], ENTITY_NAMES[right]]
            for left in range(len(ENTITY_NAMES)) for right in range(left + 1, len(ENTITY_NAMES))
        ],
    }
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
