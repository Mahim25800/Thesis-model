"""Generate and save per-image predictions for Chameleon using Universal v5 Calibrated."""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch

from src.models.hybrid_detector import DualStreamHybridDetector


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = ROOT_DIR / "models" / "universal_v5_calibrated" / "best_model.pt"
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    standardizer = ckpt["standardizer"]

    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode=ckpt.get("gate_mode", "v3_calibrated"),
        dropout=0.15,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cham_path = ROOT_DIR / "data" / "universal_v4" / "chameleon" / "chameleon_cache.pt"
    print(f"Loading Chameleon cache ({cham_path})...")
    d_cham = torch.load(cham_path, map_location="cpu", weights_only=True)

    phys_f = d_cham["physics_features"]
    phys_c = d_cham["physics_confidences"]
    dino_cls = d_cham["dinov2_cls"]
    dino_reg = d_cham["dinov2_regional"]
    labels = d_cham["labels"]
    filenames = d_cham["filenames"]

    mean = torch.tensor(standardizer["mean"], dtype=phys_f.dtype)
    scale = torch.tensor(standardizer["scale"], dtype=phys_f.dtype)

    batch_size = 256
    n = len(labels)
    all_final, all_sem, all_phys, all_alphas, all_temps = [], [], [], [], []

    print(f"Evaluating {n:,} images on {device}...")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_pf = ((phys_f[i:i+batch_size] - mean.unsqueeze(0)) / scale.unsqueeze(0)).to(device)
            b_pc = phys_c[i:i+batch_size].to(device)
            b_dc = dino_cls[i:i+batch_size].to(device)
            b_dr = dino_reg[i:i+batch_size].to(device)

            out = model(
                physics_features=b_pf,
                physics_confidences=b_pc,
                dinov2_cls=b_dc,
                dinov2_regional=b_dr,
            )

            all_final.extend(out["prob_final"].cpu().numpy().tolist())
            all_sem.extend(out["prob_semantic"].cpu().numpy().tolist())
            all_phys.extend(out["prob_physics"].cpu().numpy().tolist())
            all_alphas.extend(out["alpha"].cpu().numpy().tolist())
            all_temps.extend(out["temperature"].cpu().numpy().tolist())

    df = pd.DataFrame({
        "filename": filenames,
        "label": labels.numpy() if hasattr(labels, "numpy") else labels,
        "prob_final": all_final,
        "prob_semantic": all_sem,
        "prob_physics": all_phys,
        "alpha": all_alphas,
        "temperature": all_temps,
    })

    out_pq = ROOT_DIR / "reports" / "chameleon_v5_calibrated_predictions.parquet"
    df.to_parquet(out_pq)
    print(f"Saved {len(df):,} predictions to {out_pq.relative_to(ROOT_DIR)}")

    # Summary
    y = df["label"].values
    pred = (df["prob_final"].values >= 0.5).astype(int)
    acc = (pred == y).mean()
    print(f"Chameleon Accuracy: {acc*100:.2f}% | Mean Temp: {df['temperature'].mean():.3f} | Min Temp: {df['temperature'].min():.3f} | Max Temp: {df['temperature'].max():.3f}")


if __name__ == "__main__":
    main()
