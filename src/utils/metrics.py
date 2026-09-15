"""Evaluation metrics: AUC-ROC, Equal Error Rate (EER), Accuracy, Confusion Matrix."""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, classification_report


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) where FPR == FNR, along with the optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find threshold where FPR and FNR intersect
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5

    return eer, eer_threshold


def compute_youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Computes Youden's J statistic (J = Sensitivity + Specificity - 1) and optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_j = float(j_scores[best_idx])
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_j, best_thresh


def compute_optimal_accuracy_threshold(
    y_true: np.ndarray, y_probs: np.ndarray, num_steps: int = 200
) -> Tuple[float, float]:
    """Finds decision threshold maximizing empirical classification accuracy: tau* = argmax_tau Accuracy(tau)."""
    thresholds = np.linspace(0.1, 0.9, num_steps)
    best_acc = -1.0
    best_thresh = 0.5
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        acc = float(accuracy_score(y_true, preds))
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(th)
    return best_acc, best_thresh


def evaluate_predictions(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Computes comprehensive metrics on model probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    # AUC-ROC
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc = 0.5

    # Equal Error Rate (EER)
    try:
        eer, eer_thresh = compute_eer(y_true, y_probs)
    except Exception:
        eer, eer_thresh = 0.5, 0.5

    # Accuracy at given threshold
    acc = float(accuracy_score(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Classification report
    report_str = classification_report(
        y_true, y_pred, target_names=["Real", "Fake"], digits=4, zero_division=0
    )

    return {
        "auc_roc": auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "accuracy": acc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "raw_confusion_matrix": cm.tolist(),
        "classification_report": report_str,
    }


def format_evaluation_summary(metrics: Dict[str, Any], title: str = "Evaluation Report") -> str:
    """Formats metrics dictionary into a readable markdown/text summary."""
    cm = metrics["confusion_matrix"]
    lines = [
        f"================ {title} ================",
        f"AUC-ROC:      {metrics['auc_roc']:.4f}",
        f"EER:          {metrics['eer'] * 100:.2f}% (Threshold: {metrics['eer_threshold']:.4f})",
        f"Accuracy:     {metrics['accuracy'] * 100:.2f}% (Threshold: 0.50)",
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
) -> Dict[str, Any]:
    """Evaluates both full population and observable subset."""
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    confidences = np.asarray(confidences, dtype=float)

    observability_scores = np.sum(confidences, axis=1)
    obs_mask = observability_scores >= obs_threshold

    overall_metrics = evaluate_predictions(y_true, y_probs)

    num_obs = int(np.sum(obs_mask))
    num_total = len(y_true)

    if num_obs >= 10 and len(np.unique(y_true[obs_mask])) > 1:
        obs_metrics = evaluate_predictions(y_true[obs_mask], y_probs[obs_mask])
    else:
        obs_metrics = {
            "auc_roc": overall_metrics["auc_roc"],
            "eer": overall_metrics["eer"],
            "eer_threshold": overall_metrics["eer_threshold"],
            "accuracy": overall_metrics["accuracy"],
            "confusion_matrix": overall_metrics["confusion_matrix"],
            "classification_report": "Insufficient sample variance in observable subset.",
        }

    return {
        "overall": overall_metrics,
        "observable": obs_metrics,
        "obs_threshold": obs_threshold,
        "num_observable": num_obs,
        "num_indeterminate": num_total - num_obs,
        "total_samples": num_total,
    }

