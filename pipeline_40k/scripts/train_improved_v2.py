"""Train the accuracy-focused cached-feature ensemble with frozen splits."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from src.data.dataset import PhysicsFeatureDataset
from src.models.cross_gen_gated import TransformerPhysicsCrossGenHead
from src.models.improved import legacy_cache_matrix, probability_logits
from src.utils.decision_policy import apply_policy_calibration, fit_decision_policy
from src.utils.metrics import evaluate_predictions
from scripts.evaluate_scaled import evaluate_selective_abstention


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def legacy_probabilities(model, dataset, indices):
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            rows = indices[start:start + 256]
            output.append(torch.sigmoid(model(dataset.features[rows], dataset.confidences[rows])[0]).numpy())
    return np.concatenate(output)


def evaluate(labels, raw, confidences, policy):
    calibrated = apply_policy_calibration(probability_logits(raw), policy)
    return {
        "binary": evaluate_predictions(labels, calibrated, policy["decision_threshold"]),
        "raw_threshold_0_5": evaluate_predictions(labels, raw, 0.5),
        "selective": evaluate_selective_abstention(labels, calibrated, confidences,
            policy["tau_low"], policy["tau_high"], policy["obs_threshold"]),
    }, calibrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models/improved_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain inside pipeline_40k")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("model.joblib", "evaluation.json")):
        raise FileExistsError("Run output already exists; choose a new directory")
    split_path = ROOT / "models/improved_v1/integrity/split_indices.npz"
    manifest_path = ROOT / "models/improved_v1/integrity/split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_cache = ROOT / "data/cache_scaled/train_scaled_features.pt"
    test_cache = ROOT / "data/cache_scaled/test_scaled_features.pt"
    if sha256(train_cache) != manifest["source_caches"]["train"]["sha256"] or sha256(test_cache) != manifest["source_caches"]["test"]["sha256"]:
        raise ValueError("Caches changed after the integrity audit")
    with np.load(split_path) as splits:
        train_idx = splits["train_indices"].copy()
        development_idx = splits["development_indices"].copy()
        calibration_idx = splits["calibration_indices"].copy()
        test_idx = splits["test_indices"].copy()
    training = PhysicsFeatureDataset.from_cache(train_cache)
    matrix = legacy_cache_matrix(training.features, training.confidences)
    labels = training.labels.numpy().astype(int)
    legacy_path = ROOT / "models/gated_cross_gen_40k_best.pt"
    legacy = TransformerPhysicsCrossGenHead().eval()
    legacy.load_state_dict(torch.load(legacy_path, map_location="cpu", weights_only=True))
    model = HistGradientBoostingClassifier(max_leaf_nodes=63, learning_rate=.08,
        l2_regularization=10., max_iter=500, min_samples_leaf=20,
        early_stopping=True, validation_fraction=.1, n_iter_no_change=35,
        random_state=args.seed)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(matrix[train_idx], labels[train_idx])
    development_tree = model.predict_proba(matrix[development_idx])[:, 1]
    development_legacy = legacy_probabilities(legacy, training, development_idx)
    candidates = []
    for weight in np.linspace(0, 1, 11):
        probability = weight * development_tree + (1 - weight) * development_legacy
        candidates.append({"tree_weight": float(weight),
            "metrics": evaluate_predictions(labels[development_idx], probability, 0.5)})
    best = max(candidates, key=lambda row: (row["metrics"]["accuracy"], row["metrics"]["auc_roc"]))
    blend_weight = best["tree_weight"]
    legacy_development = evaluate_predictions(labels[development_idx], development_legacy, 0.5)
    promoted = best["metrics"]["accuracy"] > legacy_development["accuracy"]
    selection = {"protocol": "HGB parameters fixed by development-only round2 screening; blend selected on development only",
        "hgb_parameters": model.get_params(), "blend_candidates": candidates,
        "selected_tree_weight": blend_weight, "selected": best["metrics"],
        "legacy": legacy_development, "promoted": promoted,
        "training_seconds": time.perf_counter() - started}
    write_json(output / "selection.json", selection)
    if not promoted:
        raise RuntimeError("Candidate did not beat the legacy model on development; no artifact promoted")

    calibration_tree = model.predict_proba(matrix[calibration_idx])[:, 1]
    calibration_legacy = legacy_probabilities(legacy, training, calibration_idx)
    calibration_raw = blend_weight * calibration_tree + (1 - blend_weight) * calibration_legacy
    policy = fit_decision_policy(probability_logits(calibration_raw), labels[calibration_idx], 1.0, 0.10)
    metadata = {"schema_version": 1, "feature_schema": "legacy_cache_14_plus_4_v1",
        "raw_image_inference_supported": False,
        "reason": "Historical normal-head state and two historical highlight calculations cannot be reconstructed from raw images",
        "seed": args.seed, "selected_tree_weight": blend_weight,
        "legacy_checkpoint_sha256": sha256(legacy_path),
        "train_cache_sha256": sha256(train_cache), "test_cache_sha256": sha256(test_cache),
        "split_indices_sha256": sha256(split_path), "split_manifest_sha256": sha256(manifest_path),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "torch": str(torch.__version__)},
        "limitations": manifest["limitations"]}
    artifact = {"feature_schema": "legacy_cache_14_plus_4_v1", "estimator": model,
        "blend_weight": blend_weight, "policy": policy, "metadata": metadata}
    joblib.dump(artifact, output / "model.joblib", compress=3)
    write_json(output / "policy.json", policy)
    write_json(output / "metadata.json", metadata)

    # The candidate and policy are frozen before the historical test cache is scored.
    test = PhysicsFeatureDataset.from_cache(test_cache)
    test_matrix = legacy_cache_matrix(test.features[test_idx], test.confidences[test_idx])
    test_tree = model.predict_proba(test_matrix)[:, 1]
    test_legacy = legacy_probabilities(legacy, test, test_idx)
    test_raw = blend_weight * test_tree + (1 - blend_weight) * test_legacy
    test_labels = test.labels.numpy().astype(int)[test_idx]
    improved, improved_prob = evaluate(test_labels, test_raw, test.confidences.numpy()[test_idx], policy)
    # Compare fixed raw 0.5 legacy as the saved checkpoint's actual operational baseline.
    legacy_metrics = evaluate_predictions(test_labels, test_legacy, 0.5)
    new_correct = (improved_prob >= policy["decision_threshold"]) == test_labels
    old_correct = (test_legacy >= 0.5) == test_labels
    difference = new_correct.astype(float) - old_correct.astype(float)
    rng = np.random.default_rng(args.seed)
    bootstrap = [difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(2000)]
    comparison = {"accuracy_difference": float(difference.mean()),
        "paired_bootstrap_95_percent_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
        "bootstrap_repetitions": 2000}
    report = {"protocol": "Audited/deduplicated historical test; model selected on development and policy fit on calibration",
        "test_samples": len(test_idx), "improved": improved, "legacy_fixed_0_5": legacy_metrics,
        "comparison": comparison, "artifact_sha256": sha256(output / "model.joblib"),
        "external_validation_required": True, "limitations": metadata["limitations"]}
    write_json(output / "evaluation.json", report)
    np.savez_compressed(output / "test_predictions.npz", row_indices=test_idx, labels=test_labels,
        improved_probability=improved_prob, legacy_probability=test_legacy)
    print(json.dumps({"selected_tree_weight": blend_weight,
        "development_accuracy": best["metrics"]["accuracy"],
        "test_accuracy": improved["binary"]["accuracy"],
        "test_auc": improved["binary"]["auc_roc"],
        "legacy_test_accuracy": legacy_metrics["accuracy"],
        "comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
