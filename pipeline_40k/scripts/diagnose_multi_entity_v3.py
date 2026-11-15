"""Report entity ablations for the v3 interaction branch without retraining it.

This is a descriptive diagnostic on the already-scored historical test split.
It is not used for model selection or calibration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import PhysicsFeatureDataset
from src.models.multi_entity import ENTITY_NAMES, MultiEntityConsistencyHead, standardize_features
from src.utils.metrics import evaluate_predictions


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/multi_entity_v3/model.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "models/multi_entity_v3/ablation.json")
    args = parser.parse_args()
    if not args.model.resolve().is_relative_to(ROOT) or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Model and output must remain inside pipeline_40k")
    payload = torch.load(args.model, map_location="cpu", weights_only=True)
    models = []
    for state in payload["model_states"]:
        model = MultiEntityConsistencyHead(**payload["architecture"]).eval()
        model.load_state_dict(state)
        models.append(model)
    with np.load(ROOT / "models/improved_v1/integrity/split_indices.npz") as splits:
        indices = splits["test_indices"].copy()
    dataset = PhysicsFeatureDataset.from_cache(ROOT / "data/cache_scaled/test_scaled_features.pt")
    features = dataset.features[indices]
    confidences = dataset.confidences[indices]
    labels = dataset.labels[indices].numpy().astype(int)
    standardized = standardize_features(features, payload["standardizer"])

    def score(confidence_values, include_attention=False):
        probability, attention = [], []
        with torch.inference_mode():
            for start in range(0, len(features), 1024):
                x, c = standardized[start:start + 1024], confidence_values[start:start + 1024]
                details = [model(x, c, return_details=True) for model in models]
                probability.append(np.mean([torch.sigmoid(item["logits"]).numpy() for item in details], axis=0))
                if include_attention:
                    attention.append(np.mean([item["pair_attention"].numpy() for item in details], axis=0))
        result = {"metrics": evaluate_predictions(labels, np.concatenate(probability), 0.5)}
        if include_attention:
            pair_names = [f"{ENTITY_NAMES[a]}__{ENTITY_NAMES[b]}" for a, b in models[0].pair_indices.tolist()]
            result["mean_pair_attention"] = {
                name: float(value) for name, value in zip(pair_names, np.concatenate(attention).mean(axis=0))
            }
        return result

    report = {
        "protocol": "Descriptive historical-test ablation after the v3 model was frozen; not used in selection or calibration.",
        "entity_branch_only": {"full": score(confidences, include_attention=True), "drop_one_entity": {}},
    }
    for index, name in enumerate(ENTITY_NAMES):
        ablated = confidences.clone()
        ablated[:, index] = 0.0
        report["entity_branch_only"]["drop_one_entity"][name] = score(ablated)
    write_json(args.output, report)
    full = report["entity_branch_only"]["full"]["metrics"]
    print(json.dumps({
        "entity_only_accuracy": full["accuracy"], "entity_only_auc": full["auc_roc"],
        "entity_drop_accuracy": {name: row["metrics"]["accuracy"] for name, row in report["entity_branch_only"]["drop_one_entity"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
