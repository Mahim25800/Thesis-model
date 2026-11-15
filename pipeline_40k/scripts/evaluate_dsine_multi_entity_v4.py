"""Re-evaluate a saved DSINE multi-entity artifact on the regenerated unseen cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_scaled import evaluate_selective_abstention
from src.models.improved import probability_logits
from src.models.multi_entity import ENTITY_DIMS, MultiEntityConsistencyHead, standardize_features
from src.utils.decision_policy import apply_policy_calibration
from src.utils.metrics import evaluate_predictions


SCHEMA = "physics_cache_dsine_v1_14_features_4_confidences"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_groups(path: Path) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [row.get("source_position") for row in rows] != list(range(len(rows))):
        raise ValueError(f"Invalid ordered manifest: {path}")
    return np.asarray([row["difference_hash_64"] for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache_regenerated_dsine_v1")
    parser.add_argument("--model", type=Path, default=ROOT / "models/dsine_multi_entity_v4/model.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "models/dsine_multi_entity_v4/reevaluation.json")
    args = parser.parse_args()
    cache_dir, model_path, output = args.cache_dir.resolve(), args.model.resolve(), args.output.resolve()
    if not all(path.is_relative_to(ROOT) for path in (cache_dir, model_path, output)):
        raise ValueError("Paths must remain inside pipeline_40k")
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    if artifact.get("feature_schema") != SCHEMA or tuple(artifact.get("entity_dims", ())) != ENTITY_DIMS:
        raise ValueError("Artifact is not compatible with the DSINE feature schema")
    provenance_path = cache_dir / "provenance.json"
    if artifact.get("cache_provenance_sha256") != sha256(provenance_path):
        raise ValueError("Artifact was trained against a different cache provenance record")
    train = torch.load(cache_dir / "train_features.pt", map_location="cpu", weights_only=True)
    test = torch.load(cache_dir / "test_unseen_features.pt", map_location="cpu", weights_only=True)
    features, confidences, labels = test["features"].float(), test["confidences"].float(), test["labels"].numpy().astype(int)
    if features.shape[1] != sum(ENTITY_DIMS) or confidences.shape != (len(features), len(ENTITY_DIMS)):
        raise ValueError("Unexpected test cache shape")
    train_groups = set(manifest_groups(cache_dir / "train_manifest.jsonl"))
    test_groups = manifest_groups(cache_dir / "test_unseen_manifest.jsonl")
    keep = np.flatnonzero(~np.isin(test_groups, list(train_groups)))
    if not len(keep):
        raise RuntimeError("No non-overlapping unseen-test rows remain")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for state in artifact["model_states"]:
        model = MultiEntityConsistencyHead(**artifact["architecture"]).eval()
        model.load_state_dict(state)
        models.append(model.to(device))
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(keep), 1024):
            rows = keep[start:start + 1024]
            values = standardize_features(features[rows].to(device), artifact["standardizer"])
            confidence = confidences[rows].to(device)
            probabilities.append(np.mean([torch.sigmoid(model(values, confidence)).cpu().numpy() for model in models], axis=0))
    raw_probability = np.concatenate(probabilities)
    policy = artifact["policy"]
    calibrated = apply_policy_calibration(probability_logits(raw_probability), policy)
    report = {
        "artifact": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "cache_provenance_sha256": artifact["cache_provenance_sha256"],
        "device": str(device),
        "test_samples": int(len(keep)),
        "excluded_train_overlap_test_rows": int(len(features) - len(keep)),
        "binary": evaluate_predictions(labels[keep], calibrated, policy["decision_threshold"]),
        "raw_threshold_0_5": evaluate_predictions(labels[keep], raw_probability, 0.5),
        "selective": evaluate_selective_abstention(
            labels[keep], calibrated, confidences[keep].numpy(), policy["tau_low"], policy["tau_high"], policy["obs_threshold"]
        ),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
