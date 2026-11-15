"""Post-freeze entity and spatial ablations on cached Synthbuster physics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_external_synthbuster_regional import probability
from src.models.improved import probability_logits
from src.models.multi_entity import ENTITY_DIMS, ENTITY_NAMES
from src.models.regional_multi_physics import RegionalMultiPhysicsHead
from src.utils.decision_policy import apply_policy_calibration
from src.utils.metrics import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/generator_invariant_v6/model.pt")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache_synthbuster_regional_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "models/generator_invariant_v6/external_synthbuster/ablation.json")
    args = parser.parse_args()
    model_path, cache_dir, output = args.model.resolve(), args.cache_dir.resolve(), args.output.resolve()
    if not all(path.is_relative_to(ROOT) for path in (model_path, cache_dir, output)):
        raise ValueError("All paths must remain inside pipeline_40k")
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for state in artifact["model_states"]:
        model = RegionalMultiPhysicsHead(**artifact["architecture"]).to(device).eval()
        model.load_state_dict(state)
        models.append(model)
    real = torch.load(cache_dir / "raise_real.pt", map_location="cpu", weights_only=True)
    generator_paths = sorted(path for path in cache_dir.glob("*.pt") if path.stem != "raise_real")
    if not generator_paths:
        raise ValueError("No cached Synthbuster generator features found")
    variant_names = ["full", "global_only", *[f"without_{name}" for name in ENTITY_NAMES]]
    report = {
        "role": "post-freeze ablation; never used for selection",
        "model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "variants": {name: {"generators": {}} for name in variant_names},
    }
    for path in generator_paths:
        fake = torch.load(path, map_location="cpu", weights_only=True)
        count = min(len(real["features"]), len(fake["features"]))
        features = torch.cat([real["features"][:count], fake["features"][:count]])
        original_confidences = torch.cat([real["confidences"][:count], fake["confidences"][:count]])
        labels = np.concatenate([np.zeros(count, dtype=int), np.ones(count, dtype=int)])
        variants = {"full": original_confidences.clone()}
        global_only = original_confidences.clone()
        global_only[:, 1:] = 0
        variants["global_only"] = global_only
        for entity_index, name in enumerate(ENTITY_NAMES):
            confidence = original_confidences.clone()
            confidence[:, :, entity_index] = 0
            variants[f"without_{name}"] = confidence
        for name, confidences in variants.items():
            raw = probability(models, features, confidences, artifact["standardizer"], device)
            calibrated = apply_policy_calibration(probability_logits(raw), artifact["policy"])
            metrics = evaluate_predictions(labels, calibrated, artifact["policy"]["decision_threshold"])
            report["variants"][name]["generators"][path.stem] = {
                "auc_roc": metrics["auc_roc"], "accuracy": metrics["accuracy"], "eer": metrics["eer"],
            }
    full_auc = {
        generator: metrics["auc_roc"]
        for generator, metrics in report["variants"]["full"]["generators"].items()
    }
    for name, variant in report["variants"].items():
        aucs = [metrics["auc_roc"] for metrics in variant["generators"].values()]
        variant["mean_auc"] = float(np.mean(aucs))
        variant["mean_auc_delta_from_full"] = float(np.mean([
            metrics["auc_roc"] - full_auc[generator]
            for generator, metrics in variant["generators"].items()
        ]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
