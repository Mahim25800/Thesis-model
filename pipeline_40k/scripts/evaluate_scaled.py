"""Evaluates 100% Pure Physics 5-Token Transformer with 3-Way Forensic Decision Policy."""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

# Ensure pipeline root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cross_gen_gated import TransformerPhysicsCrossGenHead
from src.data.dataset import PhysicsFeatureDataset
from src.utils.metrics import (
    evaluate_predictions,
    evaluate_observability_subsets,
    format_evaluation_summary,
    compute_eer,
    compute_youden_threshold,
    compute_optimal_accuracy_threshold,
)


def fit_platt_scaling(val_logits: np.ndarray, val_labels: np.ndarray):
    """Fits Platt scaling logistic regression: P(Fake | z) = sigma(a * z + b)."""
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(val_logits.reshape(-1, 1), val_labels)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return lr, a, b


def evaluate_selective_abstention(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    confidences: np.ndarray,
    tau_low: float = 0.40,
    tau_high: float = 0.60,
    obs_threshold: float = 1.0,
):
    """Evaluates 3-way selective classification with rejection option:

    Class 1 (Authentic): P(Fake) < tau_low AND O >= obs_threshold
    Class 2 (Synthetic): P(Fake) >= tau_high AND O >= obs_threshold
    Class 3 (Indeterminate): O < obs_threshold OR tau_low <= P(Fake) < tau_high
    """
    obs_scores = np.sum(confidences, axis=1)
    is_obs = obs_scores >= obs_threshold

    is_authentic = (y_probs < tau_low) & is_obs
    is_synthetic = (y_probs >= tau_high) & is_obs
    is_decided = is_authentic | is_synthetic
    is_abstained = ~is_decided

    total = len(y_true)
    num_decided = int(np.sum(is_decided))
    num_abstained = total - num_decided
    coverage = float(num_decided / total)

    if num_decided > 0:
        decided_preds = np.zeros(num_decided, dtype=int)
        decided_preds[is_synthetic[is_decided]] = 1
        decided_targets = y_true[is_decided]
        decided_acc = float(np.mean(decided_preds == decided_targets))

        # Separate authentic & synthetic accuracy
        auth_mask = is_authentic[is_decided]
        synth_mask = is_synthetic[is_decided]
        auth_acc = float(np.mean(decided_targets[auth_mask] == 0)) if np.sum(auth_mask) > 0 else 0.0
        synth_acc = float(np.mean(decided_targets[synth_mask] == 1)) if np.sum(synth_mask) > 0 else 0.0
    else:
        decided_acc = 0.0
        auth_acc = 0.0
        synth_acc = 0.0

    return {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "obs_threshold": obs_threshold,
        "total_samples": total,
        "num_decided": num_decided,
        "num_abstained": num_abstained,
        "coverage_percent": coverage * 100.0,
        "decided_accuracy": decided_acc * 100.0,
        "authentic_accuracy": auth_acc * 100.0,
        "synthetic_accuracy": synth_acc * 100.0,
    }


def evaluate_scaled_pipeline(
    checkpoint_path: Path,
    test_cache: Path,
    train_cache: Path = None,
    output_json: Path = None,
    obs_threshold: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("EVALUATING 100% PURE PHYSICS 5-TOKEN TRANSFORMER MODEL WITH 3-WAY FORENSIC DECISION POLICY")
    print("Derived Optical & Projective Invariants: Light Field SH, Corneal Specular, Normals, Shadows, VP")
    print("Physical Cross-Modal Coupling: Multi-Head Self-Attention over 5 Physical Tokens (Zero Pixel/FFT)")
    print(f"Device:           {device}")
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Test Cache:       {test_cache}")
    print(f"Primary Tau_obs:  {obs_threshold:.2f}")
    print("=" * 95)

    test_dataset = PhysicsFeatureDataset.from_cache(test_cache)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = TransformerPhysicsCrossGenHead(
        in_features=14,
        conf_dim=4,
        d_model=64,
        nhead=4,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    all_logits = []
    all_probs = []
    all_targets = []
    all_confs = []

    with torch.no_grad():
        for features, confidences, labels in test_loader:
            features = features.to(device)
            confidences = confidences.to(device)
            logits, _ = model(features, confidences)
            probs = torch.sigmoid(logits.squeeze())

            all_logits.extend(logits.squeeze().cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_confs.extend(confidences.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_logits = np.array(all_logits)
    y_probs = np.array(all_probs)
    confs = np.array(all_confs)

    # 1. Uncalibrated Static Evaluation (Threshold = 0.50)
    static_eval = evaluate_predictions(y_true, y_probs, threshold=0.50)
    print("\n" + format_evaluation_summary(static_eval, title="Static Decision Evaluation (Fixed Threshold Tau = 0.50)"))

    # 2. Cost-Sensitive Optimal EER Threshold Evaluation
    eer_val, eer_threshold = compute_eer(y_true, y_probs)
    optimal_eer_eval = evaluate_predictions(y_true, y_probs, threshold=eer_threshold)
    print("\n" + format_evaluation_summary(optimal_eer_eval, title=f"Cost-Sensitive Evaluation (Optimal EER Threshold Tau* = {eer_threshold:.4f})"))

    # 3. Youden's J Statistic Optimal Threshold Evaluation
    youden_j, youden_thresh = compute_youden_threshold(y_true, y_probs)
    youden_eval = evaluate_predictions(y_true, y_probs, threshold=youden_thresh)
    print("\n" + format_evaluation_summary(youden_eval, title=f"Youden's J Optimal Evaluation (J* = {youden_j:.4f}, Tau* = {youden_thresh:.4f})"))

    # 4. Platt Scaling Calibration: P(Fake | z) = sigma(a * z + b)
    if train_cache and train_cache.exists():
        print(f"\nFitting Platt Scaling calibration on training cache: {train_cache} ...")
        train_dataset = PhysicsFeatureDataset.from_cache(train_cache)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        train_logits = []
        train_targets = []
        with torch.no_grad():
            for f_tr, c_tr, l_tr in train_loader:
                lg, _ = model(f_tr.to(device), c_tr.to(device))
                train_logits.extend(lg.squeeze().cpu().numpy().tolist())
                train_targets.extend(l_tr.numpy().tolist())
        calib_lr, platt_a, platt_b = fit_platt_scaling(np.array(train_logits), np.array(train_targets))
    else:
        n_half = len(y_logits) // 2
        calib_lr, platt_a, platt_b = fit_platt_scaling(y_logits[:n_half], y_true[:n_half])

    calib_probs = calib_lr.predict_proba(y_logits.reshape(-1, 1))[:, 1]
    platt_eval_05 = evaluate_predictions(y_true, calib_probs, threshold=0.50)
    calib_j, calib_youden_thresh = compute_youden_threshold(y_true, calib_probs)
    platt_eval_youden = evaluate_predictions(y_true, calib_probs, threshold=calib_youden_thresh)

    print("\n" + "=" * 85)
    print(f"PLATT SCALING CALIBRATION PARAMETERS: a = {platt_a:.4f}, b = {platt_b:.4f}")
    print(f"Calibration formula: P(Fake | z) = sigma({platt_a:.4f} * z + {platt_b:.4f})")
    print(f"Platt Calibrated Accuracy (@ 0.50):                    {platt_eval_05['accuracy']*100:.2f}%")
    print(f"Platt Calibrated Accuracy (@ Youden Tau*={calib_youden_thresh:.4f}):   {platt_eval_youden['accuracy']*100:.2f}% (J*={calib_j:.4f})")
    print("=" * 85)

    # 5. Production 3-Way Selective Classification / Abstention Policy
    print("\n" + "=" * 90)
    print("3-WAY SELECTIVE CLASSIFICATION WITH ABSTENTION POLICY (Forensic Confidence Bands):")
    print(f"{'Band Margin':<14}{'Tau Low':<10}{'Tau High':<10}{'Coverage':<12}{'Abstain %':<12}{'Decided Acc':<14}{'Authentic Acc'}")
    print("-" * 90)

    abstain_results = {}
    margins = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    for m in margins:
        t_low = max(0.1, youden_thresh - m)
        t_high = min(0.9, youden_thresh + m)
        res = evaluate_selective_abstention(y_true, y_probs, confs, tau_low=t_low, tau_high=t_high, obs_threshold=obs_threshold)
        abstain_results[f"margin_{m:.2f}"] = res
        print(
            f"+/- {m:<10.2f}{t_low:<10.3f}{t_high:<10.3f}{res['coverage_percent']:<11.1f}%{100-res['coverage_percent']:<11.1f}%{res['decided_accuracy']:<13.2f}%{res['authentic_accuracy']:.2f}%"
        )
    print("=" * 90)

    # 6. Observability Threshold Sweep
    print("\n" + "=" * 85)
    print("PHYSICAL OBSERVABILITY THRESHOLD SWEEP (tau from 0.5 to 2.0 with Youden's J Optimization):")
    print(f"{'Tau':<8}{'Observable N':<16}{'Coverage':<12}{'AUC-ROC':<12}{'EER':<12}{'Youden Acc'}")
    print("-" * 85)

    sweep_results = {}
    tau_values = [0.5, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]

    for tau in tau_values:
        obs_mask = np.sum(confs, axis=1) >= tau
        n_obs = int(np.sum(obs_mask))
        pct = (n_obs / len(y_true)) * 100
        if n_obs > 10 and len(np.unique(y_true[obs_mask])) > 1:
            _, sub_y_th = compute_youden_threshold(y_true[obs_mask], y_probs[obs_mask])
            sub_eval = evaluate_predictions(y_true[obs_mask], y_probs[obs_mask], threshold=sub_y_th)
            auc = sub_eval["auc_roc"]
            eer = sub_eval["eer"] * 100
            acc = sub_eval["accuracy"] * 100
        else:
            auc, eer, acc = static_eval["auc_roc"], static_eval["eer"] * 100, static_eval["accuracy"] * 100

        sweep_results[f"tau_{tau:.1f}"] = {
            "tau": tau,
            "observable_count": n_obs,
            "coverage_percent": pct,
            "auc_roc": auc,
            "eer": eer,
            "accuracy": acc,
        }
        print(f"{tau:<8.1f}{n_obs:<16}{pct:<11.1f}%{auc:<12.4f}{eer:<11.2f}%{acc:<11.2f}%")

    print("=" * 85)

    full_output = {
        "static_threshold_0_50": static_eval,
        "optimal_eer_threshold": optimal_eer_eval,
        "youden_j_threshold": youden_eval,
        "platt_calibrated_0_50": platt_eval_05,
        "platt_calibrated_youden": platt_eval_youden,
        "selective_abstention_policy": abstain_results,
        "primary_obs_threshold": obs_threshold,
        "observability_sweep": sweep_results,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(full_output, f, indent=2)
        print(f"\nSaved comprehensive evaluation report to: {output_json}")

    return full_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer Scaled 40k Physics Deepfake Pipeline")
    parser.add_argument("--checkpoint", type=str, default="models/gated_cross_gen_40k_best.pt")
    parser.add_argument("--test-cache", type=str, default="data/cache_scaled/test_scaled_features.pt")
    parser.add_argument("--train-cache", type=str, default="data/cache_scaled/train_scaled_features.pt")
    parser.add_argument("--output-json", type=str, default="models/evaluation_scaled_40k.json")
    parser.add_argument("--obs-threshold", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if Path(args.checkpoint).exists() else PROJECT_ROOT / args.checkpoint
    test_cache = Path(args.test_cache) if Path(args.test_cache).exists() else PROJECT_ROOT / args.test_cache
    train_cache = (Path(args.train_cache) if Path(args.train_cache).exists() else PROJECT_ROOT / args.train_cache) if args.train_cache else None
    output_json = Path(args.output_json) if Path(args.output_json).is_absolute() or Path(args.output_json).parent.exists() else PROJECT_ROOT / args.output_json

    evaluate_scaled_pipeline(
        checkpoint_path,
        test_cache,
        train_cache=train_cache,
        output_json=output_json,
        obs_threshold=args.obs_threshold,
    )


if __name__ == "__main__":
    main()
