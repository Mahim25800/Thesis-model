"""One serializable calibration and decision policy for evaluation and inference."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import _binary_arrays, compute_optimal_accuracy_threshold


def fit_decision_policy(logits, labels, obs_threshold=1.0, abstention_margin=0.10):
    """Fit Platt calibration and accuracy threshold on calibration data only.

    Callers must pass a held-out development/calibration split, never training or
    final test labels. The abstention margin and observability threshold are
    prespecified settings, not optimized against test outcomes.
    """
    labels, logits = _binary_arrays(labels, logits)
    if len(np.unique(labels)) != 2:
        raise ValueError("Decision policy calibration requires both real and fake validation samples.")
    if not np.isfinite(obs_threshold) or obs_threshold < 0:
        raise ValueError("Observability threshold must be finite and non-negative.")
    if not np.isfinite(abstention_margin) or not 0 <= abstention_margin <= 0.5:
        raise ValueError("Abstention margin must lie in [0, 0.5].")
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(logits.reshape(-1, 1), labels)
    policy = {
        "schema_version": 1,
        "selection_split": "calibration",
        "calibration_samples": len(labels),
        "calibration_class_counts": {"real": int(np.sum(labels == 0)), "fake": int(np.sum(labels == 1))},
        "calibration": {"method": "platt", "a": float(calibrator.coef_[0, 0]), "b": float(calibrator.intercept_[0])},
        "threshold_selection": "maximum_calibration_accuracy_ties_nearest_0.5",
        "obs_threshold": float(obs_threshold),
        "abstention_margin": float(abstention_margin),
    }
    probs = apply_policy_calibration(logits, policy)
    accuracy, threshold = compute_optimal_accuracy_threshold(labels, probs)
    policy.update({
        "decision_threshold": threshold,
        "tau_low": float(max(0, threshold - abstention_margin)),
        "tau_high": float(min(1, threshold + abstention_margin)),
        "calibration_accuracy": accuracy,
        "probability_label": "fake",
    })
    return policy


def apply_policy_calibration(logits, policy):
    """Apply saved Platt coefficients with a numerically stable sigmoid."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or not np.all(np.isfinite(logits)):
        raise ValueError("Logits must be a finite one-dimensional array.")
    calibration = policy["calibration"]
    if calibration.get("method") not in ("platt", "identity"):
        raise ValueError("Unsupported calibration method.")
    a, b = float(calibration["a"]), float(calibration["b"])
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Calibration coefficients must be finite.")
    scaled = a * logits + b
    probs = np.empty_like(scaled)
    positive = scaled >= 0
    probs[positive] = 1.0 / (1.0 + np.exp(-scaled[positive]))
    exp_scaled = np.exp(scaled[~positive])
    probs[~positive] = exp_scaled / (1.0 + exp_scaled)
    return probs


def apply_decision_policy(logits, confidences, policy):
    """Return probabilities, binary labels, and selective labels (-1 = abstain)."""
    probs = apply_policy_calibration(logits, policy)
    confs = np.asarray(confidences, dtype=float)
    if confs.ndim != 2 or len(confs) != len(probs) or not np.all(np.isfinite(confs)):
        raise ValueError("Confidences must be a finite [samples, modalities] array.")
    threshold = float(policy["decision_threshold"])
    low, high = float(policy["tau_low"]), float(policy["tau_high"])
    obs_threshold = float(policy["obs_threshold"])
    if not 0 <= low <= threshold <= high <= 1 or not np.isfinite(obs_threshold):
        raise ValueError("Invalid saved decision/abstention thresholds.")
    observable = np.sum(confs, axis=1) >= obs_threshold
    selective = np.full(len(probs), -1, dtype=int)
    selective[(probs < low) & observable] = 0
    selective[(probs >= high) & observable] = 1
    return {
        "probabilities": probs,
        "predictions": (probs >= threshold).astype(int),
        "selective_predictions": selective,
        "observable": observable,
    }
