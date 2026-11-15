"""Evaluate all frozen regional physics models on untouched GenImage domains.

The script refuses to run if a confirmatory image's perceptual hash occurs in
any training, calibration, development, or prior internal-test cache.
"""

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

from scripts.select_physics_ensemble_v7 import branch_probability, load_branch
from scripts.train_regional_multi_physics_v5 import load_cache, read_manifest, write_json
from src.models.improved import probability_logits
from src.utils.decision_policy import apply_policy_calibration
from src.utils.metrics import evaluate_predictions


DOMAINS = ("glide", "midjourney", "sdv4", "sdv5")
MODEL_PATHS = {
    "regional_v5": ROOT / "models/regional_multi_physics_v5/model.pt",
    "generator_invariant_v6": ROOT / "models/generator_invariant_v6/model.pt",
    "reliability_gated_v8": ROOT / "models/reliability_gated_v8/model.pt",
    "physics_masked_v10": ROOT / "models/physics_masked_v10/model.pt",
}


def protected_hashes() -> set[str]:
    paths = [
        ROOT / "data/cache_regional_dsine_v2/train_manifest.jsonl",
        ROOT / "data/cache_regional_dsine_v2/test_unseen_manifest.jsonl",
    ]
    paths.extend(
        ROOT / f"data/cache_genimage_regional_v2/{domain}_manifest.jsonl"
        for domain in ("adm", "biggan", "vqdm", "wukong")
    )
    values: set[str] = set()
    for path in paths:
        values.update(row["difference_hash_64"] for row in read_manifest(path))
    return values


def summarize(generators: dict[str, dict]) -> dict:
    return {
        "mean_accuracy": float(np.mean([row["accuracy"] for row in generators.values()])),
        "mean_auc": float(np.mean([row["auc_roc"] for row in generators.values()])),
        "mean_eer": float(np.mean([row["eer"] for row in generators.values()])),
        "minimum_generator_auc": float(min(row["auc_roc"] for row in generators.values())),
    }


def bootstrap_delta(
    candidate: dict[str, dict[str, np.ndarray]], baseline: dict[str, dict[str, np.ndarray]],
    candidate_threshold: float, baseline_threshold: float, repeats: int,
) -> dict:
    rng = np.random.default_rng(20260917)
    auc_delta, accuracy_delta = [], []
    for _ in range(repeats):
        auc_rows, accuracy_rows = [], []
        for domain in DOMAINS:
            labels = candidate[domain]["labels"]
            real = np.flatnonzero(labels == 0)
            fake = np.flatnonzero(labels == 1)
            indices = np.concatenate([
                rng.choice(real, size=len(real), replace=True),
                rng.choice(fake, size=len(fake), replace=True),
            ])
            y = labels[indices]
            candidate_probability = candidate[domain]["probability"][indices]
            baseline_probability = baseline[domain]["probability"][indices]
            auc_rows.append(
                roc_auc_score(y, candidate_probability) - roc_auc_score(y, baseline_probability)
            )
            accuracy_rows.append(
                accuracy_score(y, candidate_probability >= candidate_threshold)
                - accuracy_score(y, baseline_probability >= baseline_threshold)
            )
        auc_delta.append(np.mean(auc_rows))
        accuracy_delta.append(np.mean(accuracy_rows))
    return {
        "mean_auc_delta": float(np.mean(auc_delta)),
        "mean_auc_delta_95ci": [float(np.percentile(auc_delta, 2.5)), float(np.percentile(auc_delta, 97.5))],
        "mean_accuracy_delta": float(np.mean(accuracy_delta)),
        "mean_accuracy_delta_95ci": [
            float(np.percentile(accuracy_delta, 2.5)), float(np.percentile(accuracy_delta, 97.5))
        ],
        "repeats": repeats,
        "resampling": "class-stratified within each generator; generator metrics equally weighted",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "data/cache_genimage_confirmatory_disjoint_regional_v2",
    )
    parser.add_argument("--ensemble", type=Path, default=ROOT / "models/physics_ensemble_v14/model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/physics_ensemble_v14/genimage_confirmatory")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    cache_dir, ensemble_path, output = args.cache_dir.resolve(), args.ensemble.resolve(), args.output_dir.resolve()
    if not all(path.is_relative_to(ROOT) for path in (cache_dir, ensemble_path, output)):
        raise ValueError("All paths must remain inside pipeline_40k")
    output.mkdir(parents=True, exist_ok=True)

    forbidden = protected_hashes()
    domain_data = {}
    overlap_report = {}
    confirmatory_hash_domains: dict[str, set[str]] = {}
    for domain in DOMAINS:
        features, confidences, labels_t = load_cache(cache_dir / f"{domain}_features.pt")
        manifest = read_manifest(cache_dir / f"{domain}_manifest.jsonl")
        hashes = [row["difference_hash_64"] for row in manifest]
        overlap = sorted(set(hashes) & forbidden)
        excluded_indices = [index for index, value in enumerate(hashes) if value in forbidden]
        overlap_report[domain] = {
            "samples": len(hashes), "unique_hashes": len(set(hashes)),
            "protected_hash_overlap": len(overlap), "rows_excluded": len(excluded_indices),
            "excluded": [
                {
                    "index": index,
                    "difference_hash_64": manifest[index]["difference_hash_64"],
                    "sha256": manifest[index]["sha256"],
                    "relative_path": manifest[index]["relative_path"],
                    "label": manifest[index]["label"],
                }
                for index in excluded_indices
            ],
        }
        keep = np.ones(len(hashes), dtype=bool)
        keep[excluded_indices] = False
        for value in set(value for index, value in enumerate(hashes) if keep[index]):
            confirmatory_hash_domains.setdefault(value, set()).add(domain)
        keep_tensor = torch.from_numpy(keep)
        domain_data[domain] = (
            features[keep_tensor], confidences[keep_tensor], labels_t.numpy().astype(int)[keep]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for name, path in MODEL_PATHS.items():
        artifact, models = load_branch(path, device)
        loaded[name] = (artifact, models)
    ensemble = torch.load(ensemble_path, map_location="cpu", weights_only=True)
    for branch in ensemble["branches"]:
        if branch["name"] not in loaded:
            branch_path = (ROOT / branch["path"]).resolve()
            loaded[branch["name"]] = load_branch(branch_path, device)

    report = {
        "protocol": "confirmatory evaluation; all model states, weights, calibration, and thresholds frozen first",
        "domains": list(DOMAINS),
        "samples_per_domain_after_overlap_exclusion": {
            domain: int(len(values[2])) for domain, values in domain_data.items()
        },
        "integrity": {
            "per_domain": overlap_report,
            "unique_confirmatory_hashes": len(confirmatory_hash_domains),
            "hashes_shared_across_confirmatory_domains": sum(
                len(domains) > 1 for domains in confirmatory_hash_domains.values()
            ),
            "shared_control_note": "GenImage generator domains may share the same ImageNet real controls; no shared hash may occur in protected development/training data",
        },
        "models": {},
    }
    prediction_sets: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    thresholds = {}
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for name, (artifact, models) in loaded.items():
        thresholds[name] = float(artifact["policy"]["decision_threshold"])
        metrics, predictions = {}, {}
        for domain, (features, confidences, labels) in domain_data.items():
            raw = branch_probability(artifact, models, features, confidences, device)
            calibrated = apply_policy_calibration(probability_logits(raw), artifact["policy"])
            metrics[domain] = evaluate_predictions(labels, calibrated, thresholds[name])
            predictions[domain] = {"labels": labels, "probability": calibrated}
            np.savez_compressed(
                prediction_dir / f"{name}__{domain}.npz", labels=labels.astype(np.int8),
                raw_probability=raw.astype(np.float32), calibrated_probability=calibrated.astype(np.float32),
            )
        report["models"][name] = {"summary": summarize(metrics), "generators": metrics}
        prediction_sets[name] = predictions

    ensemble_predictions, ensemble_metrics = {}, {}
    ensemble_threshold = float(ensemble["policy"]["decision_threshold"])
    thresholds["physics_ensemble_v14"] = ensemble_threshold
    for domain, (features, confidences, labels) in domain_data.items():
        raw = np.zeros(len(labels), dtype=np.float64)
        for branch in ensemble["branches"]:
            artifact, models = loaded[branch["name"]]
            raw += float(branch["weight"]) * branch_probability(artifact, models, features, confidences, device)
        calibrated = apply_policy_calibration(probability_logits(raw), ensemble["policy"])
        ensemble_metrics[domain] = evaluate_predictions(labels, calibrated, ensemble_threshold)
        ensemble_predictions[domain] = {"labels": labels, "probability": calibrated}
        np.savez_compressed(
            prediction_dir / f"physics_ensemble_v14__{domain}.npz", labels=labels.astype(np.int8),
            raw_probability=raw.astype(np.float32), calibrated_probability=calibrated.astype(np.float32),
        )
    report["models"]["physics_ensemble_v14"] = {
        "summary": summarize(ensemble_metrics), "generators": ensemble_metrics,
        "frozen_weights": {branch["name"]: branch["weight"] for branch in ensemble["branches"]},
    }
    prediction_sets["physics_ensemble_v14"] = ensemble_predictions

    report["paired_bootstrap"] = {
        f"v14_vs_{baseline}": bootstrap_delta(
            ensemble_predictions, prediction_sets[baseline], ensemble_threshold,
            thresholds[baseline], args.bootstrap_repeats,
        )
        for baseline in MODEL_PATHS
    }
    write_json(output / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
