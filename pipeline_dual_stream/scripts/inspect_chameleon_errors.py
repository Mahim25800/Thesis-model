"""Detailed post-hoc analysis of Chameleon classification errors (False Positives & False Negatives).
Inspects metadata, visual features, stream disagreement, and failure modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def analyze_errors():
    pred_path = ROOT_DIR / "reports" / "chameleon_per_image_predictions.parquet"
    if not pred_path.is_file():
        print(f"Error: {pred_path} does not exist.")
        return

    df = pd.read_parquet(pred_path)
    real_dir = ROOT_DIR / "data" / "Chameleon" / "Chameleon" / "test" / "0_real"
    fake_dir = ROOT_DIR / "data" / "Chameleon" / "Chameleon" / "test" / "1_fake"

    # 1. Overall stats
    real_mask = (df["label"] == 0.0)
    fake_mask = (df["label"] == 1.0)
    n_real = int(real_mask.sum())
    n_fake = int(fake_mask.sum())

    fp_mask = real_mask & (df["prob_final"] >= 0.5)
    fn_mask = fake_mask & (df["prob_final"] < 0.5)
    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())

    fpr = n_fp / n_real
    fnr = n_fn / n_fake

    print("=" * 80)
    print("CHAMELEON ERROR DISTRIBUTION SUMMARY")
    print("=" * 80)
    print(f"Total Real Images: {n_real} | False Positives: {n_fp} (FPR: {fpr*100:.2f}%)")
    print(f"Total Fake Images: {n_fake} | False Negatives: {n_fn} (FNR: {fnr*100:.2f}%)")
    print(f"Overall Accuracy: {(1.0 - (n_fp + n_fn)/(n_real + n_fake))*100:.2f}%")

    # Disagreement Breakdown
    # Cases where Physics was Real (<0.5) but DINOv2 was Fake (>=0.5) among real photos
    fp_phys_correct = real_mask & (df["prob_physics"] < 0.5) & (df["prob_semantic"] >= 0.5)
    print(f"\nAmong {n_fp} Real False Positives:")
    print(f"  Physics was CORRECT (<0.50 fake) on: {fp_phys_correct.sum()} ({fp_phys_correct.sum()/n_fp*100:.1f}%)")
    print(f"  DINOv2 drove the error (semantic >= 0.50): {(real_mask & (df['prob_semantic'] >= 0.5)).sum()}")

    # Cases where Physics was Fake (>=0.5) but DINOv2 was Fooled (<0.5) among fake images
    fn_phys_correct = fake_mask & (df["prob_physics"] >= 0.5) & (df["prob_semantic"] < 0.5)
    print(f"\nAmong {n_fn} Fake False Negatives (AI images that fooled the model):")
    print(f"  Physics was CORRECT (>=0.50 fake) on: {fn_phys_correct.sum()} ({fn_phys_correct.sum()/n_fn*100:.1f}%)")
    print(f"  DINOv2 was FOOLED (semantic < 0.50): {(fake_mask & (df['prob_semantic'] < 0.5)).sum()}")

    # 2. Top False Positives
    print("\n" + "=" * 80)
    print("TOP 10 WORST FALSE POSITIVES (Real photos flagged as Fake)")
    print("=" * 80)
    top_fp = df[real_mask].sort_values("prob_final", ascending=False).head(10)

    for i, (_, row) in enumerate(top_fp.iterrows(), 1):
        fn = row["filename"]
        p = real_dir / fn
        with Image.open(p) as img:
            w, h = img.size
            arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        print(f"[{i}] {fn}")
        print(f"    Prob Final: {row['prob_final']*100:.3f}% | DINOv2: {row['prob_semantic']*100:.3f}% | Physics: {row['prob_physics']*100:.3f}% | Gate Alpha: {row['alpha']:.3f}")
        print(f"    Size: {w}x{h} (Aspect: {w/h:.2f}) | Sharpness (Laplacian Var): {lap_var:.1f} | Mean Lum: {gray.mean():.1f}")

    # 3. Top False Negatives
    print("\n" + "=" * 80)
    print("TOP 10 WORST FALSE NEGATIVES (Fake images flagged as Real)")
    print("=" * 80)
    top_fn = df[fake_mask].sort_values("prob_final", ascending=True).head(10)

    for i, (_, row) in enumerate(top_fn.iterrows(), 1):
        fn = row["filename"]
        p = fake_dir / fn
        with Image.open(p) as img:
            w, h = img.size
            arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        print(f"[{i}] {fn}")
        print(f"    Prob Final: {row['prob_final']*100:.3f}% | DINOv2: {row['prob_semantic']*100:.3f}% | Physics: {row['prob_physics']*100:.3f}% | Gate Alpha: {row['alpha']:.3f}")
        print(f"    Size: {w}x{h} (Aspect: {w/h:.2f}) | Sharpness (Laplacian Var): {lap_var:.1f} | Mean Lum: {gray.mean():.1f}")


if __name__ == "__main__":
    analyze_errors()
