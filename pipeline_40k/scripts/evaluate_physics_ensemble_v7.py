"""Evaluate the Wukong-selected physics ensemble on cached Synthbuster features."""

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
from scripts.select_physics_ensemble_v7 import branch_probability, load_branch
from src.models.improved import probability_logits
from src.utils.decision_policy import apply_policy_calibration
from src.utils.metrics import evaluate_predictions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/physics_ensemble_v7/model.pt")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache_synthbuster_regional_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/physics_ensemble_v7/external_synthbuster")
    args = parser.parse_args()
    model_path, cache_dir, output = args.model.resolve(), args.cache_dir.resolve(), args.output_dir.resolve()
    if not all(path.is_relative_to(ROOT) for path in (model_path, cache_dir, output)):
        raise ValueError("All paths must remain inside pipeline_40k")
    output.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branches = []
    for branch in artifact["branches"]:
        path = ROOT / branch["path"]
        if sha256(path) != branch["sha256"]:
            raise ValueError(f"Frozen branch hash mismatch: {path}")
        branch_artifact, models = load_branch(path, device)
        branches.append((branch, branch_artifact, models))
    real = torch.load(cache_dir / "raise_real.pt", map_location="cpu", weights_only=True)
    report = {
        "protocol": "ensemble weights selected only on development data; post-freeze external evaluation",
        "selection": artifact["selection"], "generators": {},
    }
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(cache_dir.glob("*.pt")):
        if path.stem == "raise_real":
            continue
        fake = torch.load(path, map_location="cpu", weights_only=True)
        count = min(len(real["features"]), len(fake["features"]))
        features = torch.cat([real["features"][:count], fake["features"][:count]])
        confidences = torch.cat([real["confidences"][:count], fake["confidences"][:count]])
        labels = np.concatenate([np.zeros(count, dtype=int), np.ones(count, dtype=int)])
        raw = np.zeros(len(labels), dtype=np.float64)
        for branch, branch_artifact, models in branches:
            raw += float(branch["weight"]) * branch_probability(
                branch_artifact, models, features, confidences, device
            )
        calibrated = apply_policy_calibration(probability_logits(raw), artifact["policy"])
        report["generators"][path.stem] = {
            "pairs": count,
            "binary": evaluate_predictions(labels, calibrated, artifact["policy"]["decision_threshold"]),
            "raw_threshold_0_5": evaluate_predictions(labels, raw, 0.5),
            "selective": evaluate_selective_abstention(
                labels, calibrated, confidences.mean(dim=1).numpy(), artifact["policy"]["tau_low"],
                artifact["policy"]["tau_high"], artifact["policy"]["obs_threshold"],
            ),
        }
        np.savez_compressed(
            prediction_dir / f"{path.stem}.npz", labels=labels.astype(np.int8),
            raw_probability=raw.astype(np.float32), calibrated_probability=calibrated.astype(np.float32),
        )
        print(f"Evaluated physics ensemble on {path.stem}", flush=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
