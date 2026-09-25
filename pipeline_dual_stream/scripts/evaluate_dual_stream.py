"""Evaluation and Statistical Synergy Audit for Dual-Stream Hybrid Detector.
Computes ROC-AUC, Accuracy, stream ablations, and paired bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DualStreamCachedDataset
from src.models.hybrid_detector import DualStreamHybridDetector
from scripts.train_dual_stream import apply_standardizer


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_pred_hybrid: np.ndarray,
    y_pred_sem: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Union[float, List[float]]]:
    """Calculate paired bootstrap confidence interval for synergy delta (Hybrid - Semantic)."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    delta_scores = []
    hybrid_scores = []
    sem_scores = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, n_samples)
        # Ensure both classes exist in resample
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc_h = roc_auc_score(y_true[indices], y_pred_hybrid[indices])
        auc_s = roc_auc_score(y_true[indices], y_pred_sem[indices])
        hybrid_scores.append(auc_h)
        sem_scores.append(auc_s)
        delta_scores.append(auc_h - auc_s)

    delta_scores = np.sort(delta_scores)
    hybrid_scores = np.sort(hybrid_scores)
    sem_scores = np.sort(sem_scores)

    low_idx = int((alpha / 2) * len(delta_scores))
    high_idx = int((1 - alpha / 2) * len(delta_scores))

    return {
        "hybrid_auc_mean": float(np.mean(hybrid_scores)),
        "hybrid_auc_ci": [float(hybrid_scores[low_idx]), float(hybrid_scores[high_idx])],
        "semantic_auc_mean": float(np.mean(sem_scores)),
        "semantic_auc_ci": [float(sem_scores[low_idx]), float(sem_scores[high_idx])],
        "delta_synergy_mean": float(np.mean(delta_scores)),
        "delta_synergy_ci": [float(delta_scores[low_idx]), float(delta_scores[high_idx])],
        "p_value_synergy": float(np.mean(np.array(delta_scores) <= 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Stream Hybrid Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/models/dual_stream_v1/best_model.pt",
    )
    parser.add_argument("--physics-cache", type=str, required=True)
    parser.add_argument("--dinov2-cache", type=str, required=True)
    parser.add_argument(
        "--output-report",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/reports/evaluation_report.json",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {model_path}...")
    ckpt = torch.load(str(model_path), map_location=args.device, weights_only=False)
    standardizer = ckpt["standardizer"]
    config = ckpt.get("config", {})

    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        phys_dim=config.get("phys_dim", 64),
        proj_dim=config.get("proj_dim", 128),
        num_heads=config.get("num_heads", 4),
        dropout=0.0,
    ).to(args.device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loading evaluation dataset:\n  Physics: {args.physics_cache}\n  DINOv2:  {args.dinov2_cache}")
    dataset = DualStreamCachedDataset(
        physics_cache_path=args.physics_cache,
        dinov2_cache_path=args.dinov2_cache,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_labels = []
    all_final_probs = []
    all_sem_probs = []
    all_phys_probs = []
    all_alphas = []
    all_discrepancies = []

    print(f"Running inference across {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            labels = batch["label"].numpy()
            features = apply_standardizer(batch["physics_features"], standardizer).to(args.device)
            confidences = batch["physics_confidences"].to(args.device)
            dinov2_cls = batch["dinov2_cls"].to(args.device)
            dinov2_reg = batch["dinov2_regional"].to(args.device)

            out = model(
                physics_features=features,
                physics_confidences=confidences,
                dinov2_cls=dinov2_cls,
                dinov2_regional=dinov2_reg,
            )

            all_labels.extend(labels)
            all_final_probs.extend(out["prob_final"].cpu().numpy())
            all_sem_probs.extend(out["prob_semantic"].cpu().numpy())
            all_phys_probs.extend(out["prob_physics"].cpu().numpy())
            all_alphas.extend(out["alpha"].cpu().numpy())
            all_discrepancies.extend(out["quad_discrepancy"].cpu().numpy())

    y_true = np.array(all_labels)
    y_final = np.array(all_final_probs)
    y_sem = np.array(all_sem_probs)
    y_phys = np.array(all_phys_probs)
    alphas = np.array(all_alphas)
    discrepancies = np.array(all_discrepancies)

    # Point estimates
    auc_final = roc_auc_score(y_true, y_final)
    auc_sem = roc_auc_score(y_true, y_sem)
    auc_phys = roc_auc_score(y_true, y_phys)

    acc_final = accuracy_score(y_true, y_final >= 0.5)
    acc_sem = accuracy_score(y_true, y_sem >= 0.5)
    acc_phys = accuracy_score(y_true, y_phys >= 0.5)

    print(f"\nRunning {args.bootstrap_repeats} paired bootstrap iterations...")
    bootstrap_res = bootstrap_auc_ci(
        y_true=y_true,
        y_pred_hybrid=y_final,
        y_pred_sem=y_sem,
        n_bootstraps=args.bootstrap_repeats,
    )

    report = {
        "sample_count": len(y_true),
        "metrics": {
            "hybrid": {"roc_auc": float(auc_final), "accuracy": float(acc_final)},
            "semantic_dinov2_alone": {"roc_auc": float(auc_sem), "accuracy": float(acc_sem)},
            "physics_alone": {"roc_auc": float(auc_phys), "accuracy": float(acc_phys)},
            "synergy_delta": float(auc_final - auc_sem),
        },
        "bootstrap_analysis": bootstrap_res,
        "trust_gating": {
            "mean_alpha": float(np.mean(alphas)),
            "std_alpha": float(np.std(alphas)),
            "median_alpha": float(np.median(alphas)),
        },
        "spatial_discrepancy_by_quadrant": {
            "top_left": float(np.mean(discrepancies[:, 0])),
            "top_right": float(np.mean(discrepancies[:, 1])),
            "bottom_left": float(np.mean(discrepancies[:, 2])),
            "bottom_right": float(np.mean(discrepancies[:, 3])),
        },
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("EVALUATION & SYNERGY REPORT")
    print("=" * 65)
    print(f"Total Samples Evaluated:      {len(y_true)}")
    print(f"Dual-Stream Hybrid ROC-AUC:   {auc_final:.4f}  (95% CI: [{bootstrap_res['hybrid_auc_ci'][0]:.4f}, {bootstrap_res['hybrid_auc_ci'][1]:.4f}])")
    print(f"DINOv2 Semantic Standalone:   {auc_sem:.4f}  (95% CI: [{bootstrap_res['semantic_auc_ci'][0]:.4f}, {bootstrap_res['semantic_auc_ci'][1]:.4f}])")
    print(f"Regional Physics Standalone:  {auc_phys:.4f}")
    print(f"Synergy Delta (Gain):         +{report['metrics']['synergy_delta']:.4f} (95% CI: [{bootstrap_res['delta_synergy_ci'][0]:.4f}, {bootstrap_res['delta_synergy_ci'][1]:.4f}])")
    print(f"Statistical Significance (p): {bootstrap_res['p_value_synergy']:.5f}")
    print(f"Mean Trust Gate alpha:        {report['trust_gating']['mean_alpha']:.2f}")
    print("=" * 65)
    print(f"Report saved to: {output_report}\n")


if __name__ == "__main__":
    main()
