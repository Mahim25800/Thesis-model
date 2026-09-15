"""Forensic Shortcut Diagnostic & Source Breakdown Suite.

Conducts fine-grained multi-source audits to detect potential shortcuts,
modality collapse, or single-feature reliance hidden behind aggregated AUC metrics:

1. Per-Class & Per-Source Breakdown (Authentic vs Synthetic):
   - Specificity (True Negative Rate on Authentic Real photos)
   - Sensitivity / Recall (True Positive Rate on AI Synthetic images)
   - FPR & FNR analysis
   - Probability distribution quantiles (p10, p25, p50, p75, p90)

2. Modality Importance & Cross-Modal Coupling Audit:
   - Systematically ablates each physical token (conf -> 0) via learned mask tokens
   - Quantifies marginal AUC & Accuracy drop per modality
   - Assesses whether model relies on a single dominant token vs coupled physics

3. Observability Stratification & Rejection Coverage:
   - Observable (O >= 1.0) vs Indeterminate (O < 1.0) performance
   - 3-Way decision coverage and accuracy on decided cases

4. Feature-Level Separation & Cohen's d Effect Size:
   - Inspects all 14 physical dimensions for artificial distribution spikes
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cross_gen_gated import TransformerPhysicsCrossGenHead
from src.data.dataset import PhysicsFeatureDataset
from src.utils.metrics import (
    evaluate_predictions,
    evaluate_observability_subsets,
    compute_eer,
    compute_youden_threshold,
)


FEATURE_NAMES = [
    # Modality 1: Spherical Harmonics Illumination (5D)
    "SH_Mean_Multi_Source_Res",
    "SH_Max_Isolated_Res",
    "SH_Variance_Res_Field",
    "SH_Mode_Separation",
    "SH_Spatial_Coherence",
    # Modality 2: Corneal Specular Optics (4D)
    "Cornea_Horiz_Displacement",
    "Cornea_Vert_Displacement",
    "Cornea_Angular_Delta",
    "Cornea_Profile_Discrepancy",
    # Modality 3: Surface Normal Shading (3D)
    "Normal_Variance_Error",
    "Normal_Skewness_Error",
    "Normal_Kurtosis_Error",
    # Modality 4: Chromatic Shadow Drift (2D)
    "Shadow_Penumbra_Mean_Drift",
    "Shadow_Peak_Gradient_Drift",
]

MODALITY_NAMES = [
    ("Token 1: Illumination SH (5D)", 0, slice(0, 5)),
    ("Token 2: Corneal/Specular (4D)", 1, slice(5, 9)),
    ("Token 3: Surface Normals (3D)", 2, slice(9, 12)),
    ("Token 4: Chromatic Shadow (2D)", 3, slice(12, 14)),
]


def run_shortcut_diagnostics(
    checkpoint_path: Path,
    test_cache: Path,
    val_cache: Path = None,
    output_report_path: Path = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("PHYSICS-BASED MODEL SHORTCUT DIAGNOSTIC & PER-SOURCE FORENSIC AUDIT")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Test Cache:  {test_cache}")
    print(f"Device:      {device}")
    print("=" * 95)

    test_dataset = PhysicsFeatureDataset.from_cache(test_cache)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

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
    all_feats = []

    with torch.no_grad():
        for features, confidences, labels in test_loader:
            feats_d = features.to(device)
            confs_d = confidences.to(device)
            logits, _ = model(feats_d, confs_d)
            probs = torch.sigmoid(logits.squeeze())

            all_logits.extend(logits.squeeze().cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_confs.extend(confidences.cpu().numpy().tolist())
            all_feats.extend(features.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    confs = np.array(all_confs)
    feats = np.array(all_feats)

    # ---------------------------------------------------------
    # 1. Overall & Source-Level Breakdown
    # ---------------------------------------------------------
    base_eval = evaluate_predictions(y_true, y_probs, threshold=0.50)
    eer_val, eer_th = compute_eer(y_true, y_probs)
    youden_j, youden_th = compute_youden_threshold(y_true, y_probs)

    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)

    real_probs = y_probs[real_mask]
    fake_probs = y_probs[fake_mask]

    # Specificity & Sensitivity
    tn = int(np.sum(real_probs < 0.50))
    fp = int(np.sum(real_probs >= 0.50))
    fn = int(np.sum(fake_probs < 0.50))
    tp = int(np.sum(fake_probs >= 0.50))

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print("\n" + "=" * 80)
    print("1. PER-SOURCE & CLASS DISCRIMINATION BREAKDOWN (@ Tau = 0.50)")
    print("=" * 80)
    print(f"Overall Test AUC-ROC:       {base_eval['auc_roc']:.4f}")
    print(f"Overall Accuracy:           {base_eval['accuracy'] * 100:.2f}%")
    print(f"Equal Error Rate (EER):     {eer_val * 100:.2f}% (at Tau* = {eer_th:.4f})")
    print(f"Youden J* Index:            {youden_j:.4f} (Optimal Tau* = {youden_th:.4f})")
    print("-" * 80)
    print(f"Authentic Real Photos (COCO Val):")
    print(f"  Total Samples:            {len(real_probs):,}")
    print(f"  Accuracy (Specificity):   {specificity * 100:.2f}% ({tn:,} / {len(real_probs):,})")
    print(f"  False Positive Rate:      {(1.0 - specificity) * 100:.2f}% ({fp:,} false alarms)")
    print(f"  P(Fake) Quantiles:        p10={np.percentile(real_probs, 10):.3f} | p25={np.percentile(real_probs, 25):.3f} | Median={np.median(real_probs):.3f} | p75={np.percentile(real_probs, 75):.3f} | p90={np.percentile(real_probs, 90):.3f}")
    print("-" * 80)
    print(f"Synthetic AI Images (Modern Generators):")
    print(f"  Total Samples:            {len(fake_probs):,}")
    print(f"  Accuracy (Recall/Sens):   {sensitivity * 100:.2f}% ({tp:,} / {len(fake_probs):,})")
    print(f"  False Negative Rate:      {(1.0 - sensitivity) * 100:.2f}% ({fn:,} missed fakes)")
    print(f"  P(Fake) Quantiles:        p10={np.percentile(fake_probs, 10):.3f} | p25={np.percentile(fake_probs, 25):.3f} | Median={np.median(fake_probs):.3f} | p75={np.percentile(fake_probs, 75):.3f} | p90={np.percentile(fake_probs, 90):.3f}")

    # ---------------------------------------------------------
    # 2. Modality Ablation & Shortcut Risk Audit
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. MODALITY ABLATION AUDIT (Gating out each physical token)")
    print("=" * 80)
    print(f"{'Ablated Modality':<32}{'Test AUC':<12}{'Delta AUC':<13}{'Accuracy':<12}{'EER':<10}")
    print("-" * 80)

    modality_ablation_results = {}
    for mod_name, mod_idx, _ in MODALITY_NAMES:
        ablated_probs = []
        with torch.no_grad():
            for features, confidences, labels in test_loader:
                confs_ab = confidences.clone()
                confs_ab[:, mod_idx] = 0.0  # Force confidence to 0 -> learned mask token
                logits, _ = model(features.to(device), confs_ab.to(device))
                probs = torch.sigmoid(logits.squeeze())
                ablated_probs.extend(probs.cpu().numpy().tolist())

        res = evaluate_predictions(y_true, np.array(ablated_probs))
        delta_auc = res["auc_roc"] - base_eval["auc_roc"]
        modality_ablation_results[mod_name] = {
            "auc_roc": float(res["auc_roc"]),
            "delta_auc": float(delta_auc),
            "accuracy": float(res["accuracy"]),
            "eer": float(res["eer"]),
        }
        print(f"{mod_name:<32}{res['auc_roc']:<12.4f}{delta_auc:<+13.4f}{res['accuracy']*100:<11.2f}%{res['eer']*100:<9.2f}%")

    # ---------------------------------------------------------
    # 3. Observability Stratification & 3-Way Forensic Policy
    # ---------------------------------------------------------
    obs_res = evaluate_observability_subsets(y_true, y_probs, confs, obs_threshold=1.0)
    print("\n" + "=" * 80)
    print("3. OBSERVABILITY SUBSET PERFORMANCE (Threshold O >= 1.00)")
    print("=" * 80)
    print(f"Total Samples:        {obs_res['total_samples']:,}")
    print(f"Observable Subset:    {obs_res['num_observable']:,} ({obs_res['num_observable']/obs_res['total_samples']*100:.1f}%)")
    print(f"  Observable AUC:     {obs_res['observable']['auc_roc']:.4f} (vs {base_eval['auc_roc']:.4f} overall)")
    print(f"  Observable Acc:     {obs_res['observable']['accuracy']*100:.2f}% (vs {base_eval['accuracy']*100:.2f}% overall)")
    print(f"  Observable EER:     {obs_res['observable']['eer']*100:.2f}% (vs {base_eval['eer']*100:.2f}% overall)")
    print(f"Indeterminate Subset: {obs_res['num_indeterminate']:,} ({obs_res['num_indeterminate']/obs_res['total_samples']*100:.1f}%)")

    # ---------------------------------------------------------
    # 4. Feature-Level Separation & Cohen's d Effect Sizes
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. FEATURE EFFECT SIZE & PHYSICAL SEPARATION (Cohen's d)")
    print("=" * 80)
    print(f"{'Feature Dimension':<32}{'Real Mean':<12}{'Fake Mean':<12}{'Cohen d':<12}{'Separation'}")
    print("-" * 80)

    feature_stats = {}
    for f_idx, f_name in enumerate(FEATURE_NAMES):
        real_vals = feats[real_mask, f_idx]
        fake_vals = feats[fake_mask, f_idx]

        m_r, s_r = float(np.mean(real_vals)), float(np.std(real_vals))
        m_f, s_f = float(np.mean(fake_vals)), float(np.std(fake_vals))

        pooled_std = np.sqrt((s_r**2 + s_f**2) / 2.0) + 1e-7
        cohen_d = float((m_f - m_r) / pooled_std)

        # Classification of effect size
        abs_d = abs(cohen_d)
        if abs_d > 0.8:
            tag = "High (Check shortcut)"
        elif abs_d > 0.5:
            tag = "Moderate signal"
        elif abs_d > 0.2:
            tag = "Small signal"
        else:
            tag = "Subtle / Shared"

        feature_stats[f_name] = {
            "real_mean": m_r,
            "real_std": s_r,
            "fake_mean": m_f,
            "fake_std": s_f,
            "cohen_d": cohen_d,
            "assessment": tag,
        }
        print(f"{f_name:<32}{m_r:<12.4f}{m_f:<12.4f}{cohen_d:<+12.3f}{tag}")

    report = {
        "overall_evaluation": base_eval,
        "optimal_eer": {"eer": eer_val, "threshold": eer_th},
        "youden_j": {"j_stat": youden_j, "threshold": youden_th},
        "source_breakdown": {
            "real_samples": len(real_probs),
            "specificity": specificity,
            "false_positive_rate": 1.0 - specificity,
            "fake_samples": len(fake_probs),
            "sensitivity": sensitivity,
            "false_negative_rate": 1.0 - sensitivity,
        },
        "modality_ablation": modality_ablation_results,
        "observability_stratification": obs_res,
        "feature_effect_sizes": feature_stats,
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved complete diagnostic report to: {output_report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run Forensic Shortcut Diagnostics on Physics Deepfake Pipeline")
    parser.add_argument("--checkpoint", type=str, default="models/gated_cross_gen_40k_best.pt")
    parser.add_argument("--test-cache", type=str, default="data/cache_scaled/test_scaled_features.pt")
    parser.add_argument("--output-json", type=str, default="models/shortcut_diagnostics_report.json")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if Path(args.checkpoint).exists() else PROJECT_ROOT / args.checkpoint
    test_cache = Path(args.test_cache) if Path(args.test_cache).exists() else PROJECT_ROOT / args.test_cache
    output_json = Path(args.output_json) if Path(args.output_json).is_absolute() else PROJECT_ROOT / args.output_json

    run_shortcut_diagnostics(
        checkpoint_path=checkpoint_path,
        test_cache=test_cache,
        output_report_path=output_json,
    )


if __name__ == "__main__":
    main()
