"""Comprehensive statistical error analysis of Chameleon predictions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd


def main():
    pred_path = ROOT_DIR / "reports" / "chameleon_per_image_predictions.parquet"
    df = pd.read_parquet(pred_path)

    real = df[df["label"] == 0.0]
    tn = real[real["prob_final"] < 0.5]
    fp = real[real["prob_final"] >= 0.5]

    print("=" * 80)
    print("STATISTICAL COMPARISON: TRUE NEGATIVES (Correct Real) vs FALSE POSITIVES (Misclassified Real)")
    print("=" * 80)
    print(f"Count: True Negatives = {len(tn)} ({len(tn)/len(real)*100:.1f}%), False Positives = {len(fp)} ({len(fp)/len(real)*100:.1f}%)")
    print(f"Mean DINOv2 Fake Prob:  TN = {tn['prob_semantic'].mean():.4f}, FP = {fp['prob_semantic'].mean():.4f}")
    print(f"Mean Physics Fake Prob: TN = {tn['prob_physics'].mean():.4f}, FP = {fp['prob_physics'].mean():.4f}")
    print(f"Mean Gate Alpha:        TN = {tn['alpha'].mean():.4f}, FP = {fp['alpha'].mean():.4f}")

    fp_phys_below_half = (fp["prob_physics"] < 0.5).sum()
    fp_sem_above_half = (fp["prob_semantic"] >= 0.5).sum()
    print(f"\nIn False Positives:")
    print(f"  Physics stream correctly predicted REAL (<0.50) in: {fp_phys_below_half}/{len(fp)} ({fp_phys_below_half/len(fp)*100:.1f}%)")
    print(f"  DINOv2 stream drove the false positive (>=0.50) in:  {fp_sem_above_half}/{len(fp)} ({fp_sem_above_half/len(fp)*100:.1f}%)")

    # False Negatives
    fake = df[df["label"] == 1.0]
    tp = fake[fake["prob_final"] >= 0.5]
    fn = fake[fake["prob_final"] < 0.5]

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: TRUE POSITIVES (Detected AI) vs FALSE NEGATIVES (Undetected AI)")
    print("=" * 80)
    print(f"Count: True Positives = {len(tp)} ({len(tp)/len(fake)*100:.1f}%), False Negatives = {len(fn)} ({len(fn)/len(fake)*100:.1f}%)")
    print(f"Mean DINOv2 Fake Prob:  TP = {tp['prob_semantic'].mean():.4f}, FN = {fn['prob_semantic'].mean():.4f}")
    print(f"Mean Physics Fake Prob: TP = {tp['prob_physics'].mean():.4f}, FN = {fn['prob_physics'].mean():.4f}")
    print(f"Mean Gate Alpha:        TP = {tp['alpha'].mean():.4f}, FN = {fn['alpha'].mean():.4f}")

    fn_phys_above_half = (fn["prob_physics"] >= 0.5).sum()
    fn_sem_below_half = (fn["prob_semantic"] < 0.5).sum()
    print(f"\nIn False Negatives:")
    print(f"  Physics stream correctly flagged AI (>=0.50) in:    {fn_phys_above_half}/{len(fn)} ({fn_phys_above_half/len(fn)*100:.1f}%)")
    print(f"  DINOv2 stream was fooled (<0.50) in:                {fn_sem_below_half}/{len(fn)} ({fn_sem_below_half/len(fn)*100:.1f}%)")

    # Disagreement & Synergy Analysis
    print("\n" + "=" * 80)
    print("STREAM COMPLEMENTARITY & DISAGREEMENT DYNAMICS ACROSS ALL 26,033 SAMPLES")
    print("=" * 80)
    # Both agree
    agree_mask = (df["prob_semantic"] >= 0.5) == (df["prob_physics"] >= 0.5)
    disagree_mask = ~agree_mask
    print(f"Streams Agree:    {agree_mask.sum()} / {len(df)} ({agree_mask.mean()*100:.1f}%)")
    print(f"Streams Disagree: {disagree_mask.sum()} / {len(df)} ({disagree_mask.mean()*100:.1f}%)")

    disagree_df = df[disagree_mask]
    disagree_acc_hybrid = ( (disagree_df["prob_final"] >= 0.5) == (disagree_df["label"] == 1.0) ).mean()
    disagree_acc_dino = ( (disagree_df["prob_semantic"] >= 0.5) == (disagree_df["label"] == 1.0) ).mean()
    disagree_acc_phys = ( (disagree_df["prob_physics"] >= 0.5) == (disagree_df["label"] == 1.0) ).mean()
    print(f"Accuracy on Disagreement Set ({len(disagree_df)} images):")
    print(f"  Hybrid Model: {disagree_acc_hybrid*100:.2f}%")
    print(f"  DINOv2:       {disagree_acc_dino*100:.2f}%")
    print(f"  Physics:      {disagree_acc_phys*100:.2f}%")
    print(f"  Hybrid gain over DINOv2 on disagreement: {(disagree_acc_hybrid - disagree_acc_dino)*100:+.2f}%")


if __name__ == "__main__":
    main()
