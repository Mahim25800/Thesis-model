"""Evaluate the legacy physics transformer with a frozen validation policy.

Calibration and operational thresholds are determined before loading test labels.
The saved policy is the source of truth for binary and selective decisions.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cross_gen_gated import TransformerPhysicsCrossGenHead
from src.data.dataset import PhysicsFeatureDataset
from src.utils.decision_policy import (
    apply_policy_calibration,
    fit_decision_policy,
)
from src.utils.metrics import (
    _binary_arrays,
    evaluate_predictions,
    evaluate_observability_subsets,
    format_evaluation_summary,
)


def fit_platt_scaling(val_logits: np.ndarray, val_labels: np.ndarray):
    """Compatibility helper; callers must provide validation/calibration labels."""
    labels, logits = _binary_arrays(val_labels, val_logits)
    if len(np.unique(labels)) != 2:
        raise ValueError("Platt scaling requires both classes in validation data.")
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(logits.reshape(-1, 1), labels)
    return lr, float(lr.coef_[0, 0]), float(lr.intercept_[0])


# Public alias retained for integrations which import policy fitting here.
fit_validation_policy = fit_decision_policy


def evaluate_selective_abstention(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    confidences: np.ndarray,
    tau_low: float = 0.40,
    tau_high: float = 0.60,
    obs_threshold: float = 1.0,
):
    """Evaluate fixed rejection bands; class-specific results are precision.

    New metric names explicitly include percent. The three historical accuracy
    keys remain percentage aliases for compatibility; authentic/synthetic values
    measure precision among accepted predictions, not class recall.
    """
    labels, probs = _binary_arrays(y_true, y_probs)
    confs = np.asarray(confidences, dtype=float)
    if confs.ndim != 2 or len(confs) != len(labels) or not np.all(np.isfinite(confs)):
        raise ValueError("Confidences must be a finite [samples, modalities] array.")
    if not 0 <= tau_low <= tau_high <= 1 or not np.isfinite(obs_threshold):
        raise ValueError("Invalid abstention or observability thresholds.")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    observable = np.sum(confs, axis=1) >= obs_threshold
    authentic = (probs < tau_low) & observable
    synthetic = (probs >= tau_high) & observable
    decided = authentic | synthetic
    num_decided = int(np.sum(decided))
    correct = (authentic & (labels == 0)) | (synthetic & (labels == 1))
    accuracy = float(100.0 * np.sum(correct) / num_decided) if num_decided else None
    real_precision = float(100.0 * np.mean(labels[authentic] == 0)) if np.any(authentic) else None
    fake_precision = float(100.0 * np.mean(labels[synthetic] == 1)) if np.any(synthetic) else None
    return {
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "obs_threshold": float(obs_threshold),
        "total_samples": len(labels),
        "num_decided": num_decided,
        "num_abstained": len(labels) - num_decided,
        "num_authentic_predictions": int(np.sum(authentic)),
        "num_synthetic_predictions": int(np.sum(synthetic)),
        "coverage_percent": 100.0 * num_decided / len(labels) if len(labels) else None,
        "decided_accuracy_percent": accuracy,
        "authentic_precision_percent": real_precision,
        "synthetic_precision_percent": fake_precision,
        "decided_accuracy": accuracy,
        "authentic_accuracy": real_precision,
        "synthetic_accuracy": fake_precision,
        "legacy_alias_note": "Legacy accuracy keys are percentages; class-specific values are prediction precision.",
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_path(path):
    path = Path(path)
    path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Evaluation outputs must stay inside pipeline_40k.")
    return path


def _write_json(path, data):
    path = _output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def _predict_cache(model, cache_path, device):
    dataset = PhysicsFeatureDataset.from_cache(cache_path)
    if len(dataset) == 0:
        raise ValueError(f"Cannot evaluate empty cache: {cache_path}")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=device.type == "cuda")
    logits, targets, confidences = [], [], []
    with torch.inference_mode():
        for features, confs, labels in loader:
            batch_logits, _ = model(features.to(device), confs.to(device))
            # reshape(-1) also handles a singleton final batch.
            logits.append(batch_logits.reshape(-1).cpu().numpy())
            targets.append(labels.reshape(-1).cpu().numpy())
            confidences.append(confs.cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets), np.concatenate(confidences)


def evaluate_scaled_pipeline(
    checkpoint_path: Path,
    test_cache: Path,
    val_cache: Path = None,
    train_cache: Path = None,
    output_json: Path = None,
    obs_threshold: float = 1.0,
    policy_json: Path = None,
    policy_input: Path = None,
    abstention_margin: float = 0.10,
):
    """Evaluate a legacy checkpoint; never fit calibration to training/test data.

    train_cache is deprecated and deliberately unused. Supply a genuine
    validation cache or a previously frozen policy via policy_input.
    """
    checkpoint_path, test_cache = Path(checkpoint_path), Path(test_cache)
    val_cache = Path(val_cache) if val_cache is not None else None
    if policy_input is None:
        if val_cache is None or not val_cache.is_file():
            raise ValueError("A validation cache is required for calibration; training fallback is prohibited.")
        if val_cache.resolve() == test_cache.resolve() or _sha256(val_cache) == _sha256(test_cache):
            raise ValueError("Validation and test caches must be different data.")
    if train_cache is not None:
        print("--train-cache is deprecated and ignored; calibration uses validation only.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating legacy physics transformer: 14 features, four measured modalities.")
    print(f"Checkpoint: {checkpoint_path}; device: {device}")
    model = TransformerPhysicsCrossGenHead(in_features=14, conf_dim=4, d_model=64, nhead=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    checkpoint_hash = _sha256(checkpoint_path)

    # Freeze all operational decisions before reading any test labels.
    if policy_input is not None:
        policy = json.loads(Path(policy_input).read_text(encoding="utf-8"))
        if policy.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Saved policy does not match this checkpoint.")
        if policy.get("selection_split") not in ("validation", "calibration"):
            raise ValueError("Saved policy must identify a validation/calibration selection split.")
        print(f"Loaded frozen policy: {policy_input}")
    else:
        val_logits, val_labels, _ = _predict_cache(model, val_cache, device)
        policy = fit_decision_policy(val_logits, val_labels, obs_threshold, abstention_margin)
        policy.update({
            "selection_split": "validation",
            "validation_cache": str(val_cache.resolve()),
            "validation_cache_sha256": _sha256(val_cache),
            "checkpoint_sha256": checkpoint_hash,
            "validation_note": "Legacy validation was also used for checkpoint selection; use a separate calibration split for new experiments.",
        })
    if policy_json is None:
        policy_json = (
            _output_path(output_json).with_name(_output_path(output_json).stem + "_policy.json")
            if output_json is not None else PROJECT_ROOT / "models/legacy_validation_policy.json"
        )
    _write_json(policy_json, policy)

    test_logits, labels, confs = _predict_cache(model, test_cache, device)
    raw_probs = apply_policy_calibration(test_logits, {"calibration": {"method": "identity", "a": 1.0, "b": 0.0}})
    probs = apply_policy_calibration(test_logits, policy)
    static = evaluate_predictions(labels, raw_probs, threshold=0.5)
    calibrated_static = evaluate_predictions(labels, probs, threshold=0.5)
    frozen = evaluate_predictions(labels, probs, threshold=policy["decision_threshold"])
    selective = evaluate_selective_abstention(
        labels, probs, confs,
        tau_low=policy["tau_low"], tau_high=policy["tau_high"], obs_threshold=policy["obs_threshold"],
    )
    observability = evaluate_observability_subsets(
        labels, probs, confs, obs_threshold=policy["obs_threshold"], threshold=policy["decision_threshold"]
    )
    print("\n" + format_evaluation_summary(static, "Legacy baseline, fixed threshold 0.50"))
    print("\n" + format_evaluation_summary(frozen, "Frozen validation policy, all test samples"))
    print(f"Selective coverage: {selective['coverage_percent']:.2f}%")
    print(f"Selective accuracy (%): {selective['decided_accuracy_percent']}")
    full_output = {
        "schema_version": 2,
        "evaluation_protocol": "Calibration and decision threshold frozen on validation before test-label evaluation.",
        "test_status": "Existing development benchmark previously inspected; not a fresh confirmatory holdout.",
        "checkpoint": str(checkpoint_path.resolve()),
        "test_cache": str(test_cache.resolve()),
        "policy_path": str(_output_path(policy_json)),
        "policy": policy,
        "static_threshold_0_50": static,
        "platt_calibrated_0_50": calibrated_static,
        "frozen_validation_policy": frozen,
        "selective_abstention_policy": selective,
        "observability": observability,
    }
    if output_json is not None:
        _write_json(output_json, full_output)
        print(f"Saved evaluation: {_output_path(output_json)}")
    return full_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate legacy transformer with a frozen validation policy")
    parser.add_argument("--checkpoint", default="models/gated_cross_gen_40k_best.pt")
    parser.add_argument("--test-cache", default="data/cache_scaled/test_scaled_features.pt")
    parser.add_argument("--val-cache", default="data/cache_scaled/val_scaled_features.pt")
    parser.add_argument("--train-cache", default=None, help="Deprecated, ignored; never used for calibration")
    parser.add_argument("--output-json", default="models/evaluation_scaled_40k_validation_frozen.json")
    parser.add_argument("--policy-json", default=None, help="Output policy JSON; defaults beside the report")
    parser.add_argument("--policy-input", default=None, help="Reuse frozen policy instead of refitting calibration")
    parser.add_argument("--obs-threshold", type=float, default=1.0)
    parser.add_argument("--abstention-margin", type=float, default=0.10)
    args = parser.parse_args()

    def resolve(path):
        if not path:
            return None
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    evaluate_scaled_pipeline(
        resolve(args.checkpoint), resolve(args.test_cache), val_cache=resolve(args.val_cache),
        train_cache=resolve(args.train_cache), output_json=resolve(args.output_json),
        obs_threshold=args.obs_threshold, policy_json=resolve(args.policy_json),
        policy_input=resolve(args.policy_input), abstention_margin=args.abstention_margin,
    )


if __name__ == "__main__":
    main()
