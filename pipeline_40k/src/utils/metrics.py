"""Binary metrics evaluated at an explicitly supplied, frozen threshold.

Threshold-selection helpers are for development/calibration splits only. EER is
an ROC statistic; its threshold must not become a test-set policy. Undefined
metrics are None so reports remain valid, unambiguous JSON.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


def _binary_arrays(y_true, y_scores):
    labels = np.asarray(y_true)
    scores = np.asarray(y_scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be equally sized one-dimensional arrays.")
    if not np.all(np.isin(labels, [0, 1])) or not np.all(np.isfinite(scores)):
        raise ValueError("Labels must be binary and scores must be finite.")
    return labels.astype(int), scores


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Return empirical ROC EER and its descriptive threshold, or (None, None)."""
    labels, scores = _binary_arrays(y_true, y_scores)
    if len(np.unique(labels)) != 2:
        return None, None
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])


def compute_youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Select Youden's J threshold on development data; undefined with one class."""
    labels, scores = _binary_arrays(y_true, y_scores)
    if len(np.unique(labels)) != 2:
        return None, None
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    finite = np.isfinite(thresholds)
    thresholds, j_scores = thresholds[finite], (tpr - fpr)[finite]
    best = np.flatnonzero(j_scores == np.max(j_scores))
    index = best[np.argmin(np.abs(thresholds[best] - 0.5))]
    return float(j_scores[index]), float(thresholds[index])


def compute_optimal_accuracy_threshold(
    y_true: np.ndarray, y_probs: np.ndarray, num_steps: int = 200
) -> Tuple[Optional[float], Optional[float]]:
    """Find exact best development accuracy; ties favor threshold nearest 0.5.

    num_steps remains accepted for compatibility; the former coarse grid is
    replaced by observed score boundaries, computed in O(n log n).
    """
    labels, probs = _binary_arrays(y_true, y_probs)
    if len(labels) == 0:
        return None, None
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    order = np.argsort(probs, kind="stable")
    scores, sorted_labels = probs[order], labels[order]
    boundaries = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    count_below = np.searchsorted(scores, boundaries, side="left")
    positives_prefix = np.concatenate(([0], np.cumsum(sorted_labels)))
    positives_below = positives_prefix[count_below]
    correct = count_below - positives_below + labels.sum() - positives_below
    best = np.flatnonzero(correct == np.max(correct))
    index = best[np.argmin(np.abs(boundaries[best] - 0.5))]
    return float(correct[index] / len(labels)), float(boundaries[index])


def evaluate_predictions(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate probabilities without selecting a classification threshold."""
    labels, probs = _binary_arrays(y_true, y_probs)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be finite and lie in [0, 1].")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    predictions = (probs >= threshold).astype(int)
    count = len(labels)
    has_both_classes = len(np.unique(labels)) == 2
    auc = float(roc_auc_score(labels, probs)) if has_both_classes else None
    eer, eer_threshold = compute_eer(labels, probs)
    if count:
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
    else:
        cm = np.zeros((2, 2), dtype=int)
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    return {
        "threshold": float(threshold),
        "sample_count": count,
        "class_counts": {"real": int(np.sum(labels == 0)), "fake": int(np.sum(labels == 1))},
        "auc_roc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "eer_threshold_role": "descriptive_only_not_an_operational_threshold",
        "accuracy": float(np.mean(labels == predictions)) if count else None,
        "balanced_accuracy": float((tn / (tn + fp) + tp / (tp + fn)) / 2) if has_both_classes else None,
        "authentic_precision": float(tn / (tn + fn)) if tn + fn else None,
        "synthetic_precision": float(tp / (tp + fp)) if tp + fp else None,
        "real_recall": float(tn / (tn + fp)) if tn + fp else None,
        "fake_recall": float(tp / (tp + fn)) if tp + fn else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "raw_confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["Real", "Fake"], digits=4, zero_division=0
        ) if count else "Undefined: no samples in this population.",
    }


def format_evaluation_summary(metrics: Dict[str, Any], title: str = "Evaluation Report") -> str:
    """Format real operating thresholds and gracefully display undefined metrics."""
    def fmt(value, scale=1.0, digits=4):
        return "undefined" if value is None else f"{value * scale:.{digits}f}"

    cm = metrics["confusion_matrix"]
    lines = [
        f"================ {title} ================",
        f"Samples:      {metrics.get('sample_count', 'unknown')}",
        f"AUC-ROC:      {fmt(metrics['auc_roc'])}",
        f"EER (%):      {fmt(metrics['eer'], 100, 2)} (descriptive ROC statistic)",
        f"Accuracy (%): {fmt(metrics['accuracy'], 100, 2)} (Threshold: {fmt(metrics.get('threshold'))})",
        "",
        "Confusion Matrix:",
        f"  True Real  (TN): {cm['tn']:<5} | False Fake (FP): {cm['fp']:<5}",
        f"  False Real (FN): {cm['fn']:<5} | True Fake  (TP): {cm['tp']:<5}",
        "",
        "Detailed Classification Report:",
        metrics["classification_report"],
        "=" * (len(title) + 34),
    ]
    return "\n".join(lines)


def evaluate_observability_subsets(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    confidences: np.ndarray,
    obs_threshold: float = 1.0,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Report the actual subset, including empty/single-class populations."""
    labels, probs = _binary_arrays(y_true, y_probs)
    confs = np.asarray(confidences, dtype=float)
    if confs.ndim != 2 or len(confs) != len(labels) or not np.all(np.isfinite(confs)):
        raise ValueError("Confidences must be a finite [samples, modalities] array.")
    if not np.isfinite(obs_threshold):
        raise ValueError("Observability threshold must be finite.")
    obs_mask = np.sum(confs, axis=1) >= obs_threshold
    count = int(np.sum(obs_mask))
    return {
        "overall": evaluate_predictions(labels, probs, threshold),
        "observable": evaluate_predictions(labels[obs_mask], probs[obs_mask], threshold),
        "obs_threshold": float(obs_threshold),
        "decision_threshold": float(threshold),
        "num_observable": count,
        "num_indeterminate": len(labels) - count,
        "total_samples": len(labels),
        "coverage_percent": 100.0 * count / len(labels) if len(labels) else None,
    }
