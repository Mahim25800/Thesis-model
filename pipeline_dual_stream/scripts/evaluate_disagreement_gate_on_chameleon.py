"""Evaluate Disagreement-Exposed Gate on Chameleon Benchmark.

Computes exact empirical metrics:
- Fused vs Standalone (Semantic, Physics, Joint) Accuracy & AUC
- Mean Alpha on Semantic False Alarms vs Semantic Hits
- Disagreement Resolution & Rescue Rates
- Temperature distribution
Saves raw JSON to reports/disagreement_gate_chameleon_eval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.hybrid_detector import DualStreamHybridDetector


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        prop = in_bin.mean()
        if prop > 0:
            acc = labels[in_bin].mean()
            conf = probs[in_bin].mean()
            ece += np.abs(acc - conf) * prop
    return float(ece)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gate on Chameleon")
    parser.add_argument("--checkpoint", type=str, default="models/universal_v5_disagreement_gate/best_model.pt")
    parser.add_argument("--gate_mode", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_json", type=str, default="reports/disagreement_gate_chameleon_eval.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = ROOT_DIR / args.checkpoint
    print("=" * 80)
    print("EVALUATING DUAL-STREAM MODEL ON HELD-OUT CHAMELEON BENCHMARK")
    print(f"Checkpoint: {ckpt_path.relative_to(ROOT_DIR)}")
    print(f"Device: {device}")
    print("=" * 80)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gate_mode = args.gate_mode or ckpt.get("gate_mode", "v4_disagreement")
    standardizer = ckpt["standardizer"]

    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode=gate_mode,
        dropout=0.15,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cham_path = ROOT_DIR / "data" / "universal_v4" / "chameleon" / "chameleon_cache.pt"
    print(f"Loading Chameleon cache: {cham_path}...")
    d_cham = torch.load(cham_path, map_location="cpu", weights_only=True)

    phys_f = d_cham["physics_features"]
    phys_c = d_cham["physics_confidences"]
    dino_cls = d_cham["dinov2_cls"]
    dino_reg = d_cham["dinov2_regional"]
    labels = d_cham["labels"].numpy()
    filenames = d_cham["filenames"]
    n_samples = len(labels)

    mean = torch.tensor(standardizer["mean"], dtype=phys_f.dtype)
    scale = torch.tensor(standardizer["scale"], dtype=phys_f.dtype)

    all_final, all_sem, all_phys, all_joint, all_alphas, all_temps = [], [], [], [], [], []

    print(f"Running inference on {n_samples:,} Chameleon images...")
    with torch.no_grad():
        for i in range(0, n_samples, args.batch_size):
            b_pf = ((phys_f[i:i+args.batch_size] - mean.unsqueeze(0)) / scale.unsqueeze(0)).to(device)
            b_pc = phys_c[i:i+args.batch_size].to(device)
            b_dc = dino_cls[i:i+args.batch_size].to(device)
            b_dr = dino_reg[i:i+args.batch_size].to(device)

            out = model(
                physics_features=b_pf,
                physics_confidences=b_pc,
                dinov2_cls=b_dc,
                dinov2_regional=b_dr,
            )

            all_final.extend(out["prob_final"].cpu().numpy().tolist())
            all_sem.extend(out["prob_semantic"].cpu().numpy().tolist())
            all_phys.extend(out["prob_physics"].cpu().numpy().tolist())
            all_joint.extend(torch.sigmoid(out["joint_logits"]).cpu().numpy().tolist())
            all_alphas.extend(out["alpha"].cpu().numpy().tolist())
            all_temps.extend(out["temperature"].cpu().numpy().tolist())

    y = np.array(labels)
    p_final = np.array(all_final)
    p_sem = np.array(all_sem)
    p_phys = np.array(all_phys)
    p_joint = np.array(all_joint)
    alpha = np.array(all_alphas)
    temp = np.array(all_temps)

    pred_final = (p_final >= 0.5).astype(int)
    pred_sem = (p_sem >= 0.5).astype(int)
    pred_phys = (p_phys >= 0.5).astype(int)
    pred_joint = (p_joint >= 0.5).astype(int)

    # Core Metrics
    fused_acc = float(accuracy_score(y, pred_final))
    fused_auc = float(roc_auc_score(y, p_final))
    fused_ece = float(compute_ece(p_final, y))

    sem_acc = float(accuracy_score(y, pred_sem))
    sem_auc = float(roc_auc_score(y, p_sem))

    phys_acc = float(accuracy_score(y, pred_phys))
    phys_auc = float(roc_auc_score(y, p_phys))

    joint_acc = float(accuracy_score(y, pred_joint))
    joint_auc = float(roc_auc_score(y, p_joint))

    # Error analysis
    disagree = (pred_sem != pred_phys)
    sem_wrong_phys_right = (pred_sem != y) & (pred_phys == y)
    sem_right_phys_wrong = (pred_sem == y) & (pred_phys != y)
    sem_fa = (y == 0) & (pred_sem == 1) # Semantic False Alarms

    rescue_count = int((sem_wrong_phys_right & (pred_final == y)).sum())
    rescue_rate = float(rescue_count / max(1, sem_wrong_phys_right.sum()))

    fa_rescued = int((sem_fa & (pred_final == 0)).sum())
    fa_rescue_rate = float(fa_rescued / max(1, sem_fa.sum()))

    results = {
        "checkpoint": str(ckpt_path.relative_to(ROOT_DIR)),
        "gate_mode": gate_mode,
        "total_samples": int(n_samples),
        "fused_metrics": {
            "accuracy": fused_acc,
            "auc": fused_auc,
            "ece": fused_ece,
            "mean_alpha": float(alpha.mean()),
            "std_alpha": float(alpha.std()),
            "mean_temp": float(temp.mean()),
        },
        "stream_baselines": {
            "semantic": {"accuracy": sem_acc, "auc": sem_auc},
            "physics": {"accuracy": phys_acc, "auc": phys_auc},
            "joint": {"accuracy": joint_acc, "auc": joint_auc},
        },
        "disagreement_dynamics": {
            "total_disagreements": int(disagree.sum()),
            "sem_wrong_phys_right_count": int(sem_wrong_phys_right.sum()),
            "sem_wrong_phys_right_mean_alpha": float(alpha[sem_wrong_phys_right].mean()) if sem_wrong_phys_right.any() else 0.0,
            "sem_wrong_phys_right_rescued": rescue_count,
            "sem_wrong_phys_right_rescue_rate": rescue_rate,
            "semantic_false_alarms_count": int(sem_fa.sum()),
            "semantic_false_alarms_mean_alpha": float(alpha[sem_fa].mean()) if sem_fa.any() else 0.0,
            "semantic_false_alarms_rescued": fa_rescued,
            "semantic_false_alarms_rescue_rate": fa_rescue_rate,
            "sem_right_phys_wrong_count": int(sem_right_phys_wrong.sum()),
            "sem_right_phys_wrong_mean_alpha": float(alpha[sem_right_phys_wrong].mean()) if sem_right_phys_wrong.any() else 0.0,
            "sem_right_phys_wrong_preserved_acc": float((pred_final[sem_right_phys_wrong] == y[sem_right_phys_wrong]).mean()) if sem_right_phys_wrong.any() else 0.0,
        },
    }

    out_file = ROOT_DIR / args.out_json
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("CHAMELEON EVALUATION RESULTS")
    print("=" * 80)
    print(f"Fused Accuracy       : {fused_acc * 100:.2f}% (Baseline v5: 69.75%)")
    print(f"Fused ROC-AUC        : {fused_auc:.4f}   (Baseline v5: 0.7521)")
    print(f"Joint Stream Alone   : {joint_acc * 100:.2f}% | AUC: {joint_auc:.4f}")
    print(f"Semantic Alone       : {sem_acc * 100:.2f}% | AUC: {sem_auc:.4f}")
    print(f"Physics Alone        : {phys_acc * 100:.2f}% | AUC: {phys_auc:.4f}")
    print(f"Mean Gate Alpha      : {alpha.mean():.4f} (Semantic FA Alpha: {alpha[sem_fa].mean():.4f})")
    print(f"Semantic FA Rescued  : {fa_rescued} / {sem_fa.sum()} ({fa_rescue_rate * 100:.2f}%)")
    print(f"Results saved to     : {out_file.relative_to(ROOT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
