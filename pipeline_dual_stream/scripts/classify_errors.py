"""Classify and inspect the semantic content and visual characteristics of Chameleon error images."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image


def main():
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    df = pd.read_parquet(ROOT_DIR / "reports" / "chameleon_per_image_predictions.parquet")
    real_dir = ROOT_DIR / "data" / "Chameleon" / "Chameleon" / "test" / "0_real"
    fake_dir = ROOT_DIR / "data" / "Chameleon" / "Chameleon" / "test" / "1_fake"

    # Top False Positives
    top_fp = df[df["label"] == 0.0].sort_values("prob_final", ascending=False).head(15)

    print("=" * 80)
    print("CHARACTERIZING TOP 15 FALSE POSITIVES (Real Images Misclassified as AI)")
    print("=" * 80)
    for i, (_, row) in enumerate(top_fp.iterrows(), 1):
        fn = row["filename"]
        p = real_dir / fn
        with Image.open(p) as img:
            w, h = img.size
            t = preprocess(img.convert("RGB")).unsqueeze(0)
            arr = np.array(img.convert("RGB"))

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        with torch.no_grad():
            out = model(t)
            probs = torch.softmax(out, dim=1)[0]
            top3_prob, top3_catid = torch.topk(probs, 3)

        preds = [f"{categories[c]} ({pr*100:.1f}%)" for pr, c in zip(top3_prob, top3_catid)]
        p_final = row["prob_final"] * 100
        p_sem = row["prob_semantic"] * 100
        p_phys = row["prob_physics"] * 100
        alpha = row["alpha"]

        print(f"[{i}] {fn}")
        print(f"    Predictions: Final={p_final:.2f}%, DINOv2={p_sem:.2f}%, Physics={p_phys:.2f}%, Gate Alpha={alpha:.3f}")
        print(f"    Dimensions: {w}x{h} | Sharpness (Laplacian Var): {lap_var:.1f} | Mean Lum: {gray.mean():.1f}")
        print(f"    Scene Category: {', '.join(preds)}")

    # Top False Negatives
    top_fn = df[df["label"] == 1.0].sort_values("prob_final", ascending=True).head(15)

    print("\n" + "=" * 80)
    print("CHARACTERIZING TOP 15 FALSE NEGATIVES (AI Images Misclassified as Real)")
    print("=" * 80)
    for i, (_, row) in enumerate(top_fn.iterrows(), 1):
        fn = row["filename"]
        p = fake_dir / fn
        with Image.open(p) as img:
            w, h = img.size
            t = preprocess(img.convert("RGB")).unsqueeze(0)
            arr = np.array(img.convert("RGB"))

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        with torch.no_grad():
            out = model(t)
            probs = torch.softmax(out, dim=1)[0]
            top3_prob, top3_catid = torch.topk(probs, 3)

        preds = [f"{categories[c]} ({pr*100:.1f}%)" for pr, c in zip(top3_prob, top3_catid)]
        p_final = row["prob_final"] * 100
        p_sem = row["prob_semantic"] * 100
        p_phys = row["prob_physics"] * 100
        alpha = row["alpha"]

        print(f"[{i}] {fn}")
        print(f"    Predictions: Final={p_final:.2f}%, DINOv2={p_sem:.2f}%, Physics={p_phys:.2f}%, Gate Alpha={alpha:.3f}")
        print(f"    Dimensions: {w}x{h} | Sharpness (Laplacian Var): {lap_var:.1f} | Mean Lum: {gray.mean():.1f}")
        print(f"    Scene Category: {', '.join(preds)}")


if __name__ == "__main__":
    main()
