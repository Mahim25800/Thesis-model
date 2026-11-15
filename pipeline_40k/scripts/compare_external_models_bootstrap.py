"""Paired generator-stratified bootstrap comparisons for frozen external models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_external_synthbuster_dsine import probability as v4_probability
from src.models.improved import probability_logits
from src.models.multi_entity import MultiEntityConsistencyHead
from src.utils.decision_policy import apply_policy_calibration


def load_new_predictions(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path) as data:
            result[path.stem] = {key: data[key].copy() for key in data.files}
    if not result:
        raise ValueError(f"No prediction files found in {directory}")
    return result


def reconstruct_v4(generators: list[str], device: torch.device) -> tuple[dict, float]:
    artifact = torch.load(ROOT / "models/dsine_multi_entity_v4/model.pt", map_location="cpu", weights_only=True)
    models = []
    for state in artifact["model_states"]:
        model = MultiEntityConsistencyHead(**artifact["architecture"]).to(device).eval()
        model.load_state_dict(state)
        models.append(model)
    cache_dir = ROOT / "models/dsine_multi_entity_v4/external_synthbuster/cache"
    real = torch.load(cache_dir / "raise_real.pt", map_location="cpu", weights_only=True)
    result = {}
    for generator in generators:
        fake = torch.load(cache_dir / f"{generator}.pt", map_location="cpu", weights_only=True)
        count = min(len(real["features"]), len(fake["features"]))
        features = torch.cat([real["features"][:count], fake["features"][:count]])
        confidences = torch.cat([real["confidences"][:count], fake["confidences"][:count]])
        labels = np.concatenate([np.zeros(count, dtype=np.int8), np.ones(count, dtype=np.int8)])
        raw = v4_probability(models, features, confidences, artifact["standardizer"], device)
        calibrated = apply_policy_calibration(probability_logits(raw), artifact["policy"])
        result[generator] = {
            "labels": labels,
            "calibrated_probability": calibrated.astype(np.float32),
        }
    return result, float(artifact["policy"]["decision_threshold"])


def paired_bootstrap(
    candidate: dict[str, dict[str, np.ndarray]],
    baseline: dict[str, dict[str, np.ndarray]],
    candidate_threshold: float,
    baseline_threshold: float,
    repeats: int,
) -> dict:
    generators = sorted(candidate)
    if generators != sorted(baseline):
        raise ValueError("Prediction sets cover different generators")
    rng = np.random.default_rng(20260917)
    auc_deltas, accuracy_deltas = [], []
    per_generator = {}
    for generator in generators:
        labels = candidate[generator]["labels"].astype(int)
        if not np.array_equal(labels, baseline[generator]["labels"].astype(int)):
            raise ValueError(f"Label mismatch for {generator}")
        candidate_probability = candidate[generator]["calibrated_probability"]
        baseline_probability = baseline[generator]["calibrated_probability"]
        per_generator[generator] = {
            "candidate_auc": float(roc_auc_score(labels, candidate_probability)),
            "baseline_auc": float(roc_auc_score(labels, baseline_probability)),
            "auc_delta": float(roc_auc_score(labels, candidate_probability) - roc_auc_score(labels, baseline_probability)),
            "candidate_accuracy": float(accuracy_score(labels, candidate_probability >= candidate_threshold)),
            "baseline_accuracy": float(accuracy_score(labels, baseline_probability >= baseline_threshold)),
        }
    for _ in range(repeats):
        repeat_auc, repeat_accuracy = [], []
        for generator in generators:
            labels = candidate[generator]["labels"].astype(int)
            count = len(labels) // 2
            sampled = rng.integers(0, count, size=count)
            indices = np.concatenate([sampled, count + sampled])
            y = labels[indices]
            candidate_probability = candidate[generator]["calibrated_probability"][indices]
            baseline_probability = baseline[generator]["calibrated_probability"][indices]
            repeat_auc.append(roc_auc_score(y, candidate_probability) - roc_auc_score(y, baseline_probability))
            repeat_accuracy.append(
                accuracy_score(y, candidate_probability >= candidate_threshold)
                - accuracy_score(y, baseline_probability >= baseline_threshold)
            )
        auc_deltas.append(np.mean(repeat_auc))
        accuracy_deltas.append(np.mean(repeat_accuracy))
    return {
        "per_generator": per_generator,
        "mean_auc_delta": float(np.mean(auc_deltas)),
        "mean_auc_delta_95ci": [float(np.percentile(auc_deltas, 2.5)), float(np.percentile(auc_deltas, 97.5))],
        "mean_accuracy_delta": float(np.mean(accuracy_deltas)),
        "mean_accuracy_delta_95ci": [
            float(np.percentile(accuracy_deltas, 2.5)), float(np.percentile(accuracy_deltas, 97.5))
        ],
        "bootstrap_repeats": repeats,
        "resampling_unit": "matched real/fake filename pair within each generator; generator means equally weighted",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "models/generator_invariant_v6/external_synthbuster")
    parser.add_argument("--no-adversary-dir", type=Path, default=ROOT / "models/generator_diverse_v6_no_adversary/external_synthbuster")
    parser.add_argument("--output", type=Path, default=ROOT / "models/generator_invariant_v6/external_synthbuster/comparison_bootstrap.json")
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    candidate_dir, no_adversary_dir, output = args.candidate_dir.resolve(), args.no_adversary_dir.resolve(), args.output.resolve()
    if not all(path.is_relative_to(ROOT) for path in (candidate_dir, no_adversary_dir, output)):
        raise ValueError("All paths must remain inside pipeline_40k")
    candidate = load_new_predictions(candidate_dir / "predictions")
    no_adversary = load_new_predictions(no_adversary_dir / "predictions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v4, v4_threshold = reconstruct_v4(sorted(candidate), device)
    candidate_artifact = torch.load(candidate_dir.parent / "model.pt", map_location="cpu", weights_only=True)
    no_adversary_artifact = torch.load(no_adversary_dir.parent / "model.pt", map_location="cpu", weights_only=True)
    report = {
        "role": "post-freeze paired comparison; not used for model selection",
        "candidate_vs_v4": paired_bootstrap(
            candidate, v4, float(candidate_artifact["policy"]["decision_threshold"]), v4_threshold, args.repeats
        ),
        "candidate_vs_no_adversary": paired_bootstrap(
            candidate, no_adversary,
            float(candidate_artifact["policy"]["decision_threshold"]),
            float(no_adversary_artifact["policy"]["decision_threshold"]), args.repeats,
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
