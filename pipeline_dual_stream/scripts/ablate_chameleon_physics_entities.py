"""Per-Entity and Per-Region Physical Ablation & Attribution on Chameleon Real False Positives.

Identifies the exact physical entity driving false alarms on real photos:
- Entity 0: Illumination (Spherical Harmonics, dims 0:5)
- Entity 1: Specular Reflections & Optics (dims 5:9)
- Entity 2: Surface Normals (DSINE, dims 9:12)
- Entity 3: Chromatic Shadow / Color Consistency (dims 12:14)

Ablates one entity/region at a time by zeroing confidence and features (triggering missing tokens),
and measures Delta P_phys and Delta P_final on real images falsely flagged as AI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.models.hybrid_detector import DualStreamHybridDetector

ENTITY_NAMES = [
    "Entity 0: Illumination (SH Lighting)",
    "Entity 1: Specular & Optics Blur",
    "Entity 2: DSINE Surface Normals",
    "Entity 3: Chromatic Shadow Consistency",
]
ENTITY_DIMS = (5, 4, 3, 2)
ENTITY_RANGES = [
    (0, 5),    # Illumination
    (5, 9),    # Specular
    (9, 12),   # Normals
    (12, 14),  # Shadow
]
REGION_NAMES = [
    "Region 0: Global",
    "Region 1: Top-Left",
    "Region 2: Top-Right",
    "Region 3: Bottom-Left",
    "Region 4: Bottom-Right",
]


def apply_standardizer(features: torch.Tensor, standardizer: Dict[str, list]) -> torch.Tensor:
    mean = torch.tensor(standardizer["mean"], dtype=features.dtype, device=features.device)
    scale = torch.tensor(standardizer["scale"], dtype=features.dtype, device=features.device)
    return (features - mean.unsqueeze(0)) / scale.unsqueeze(0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("PER-ENTITY AND PER-REGION PHYSICAL ABLATION ON CHAMELEON FALSE POSITIVES")
    print("=" * 80)
    print(f"Device: {device}")

    # 1. Load Model Checkpoint (v5 Calibrated)
    ckpt_path = ROOT_DIR / "models" / "universal_v5_calibrated" / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ROOT_DIR / "models" / "universal_v4" / "best_model.pt"
    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    standardizer = ckpt["standardizer"]

    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode=ckpt.get("gate_mode", "v3_calibrated"),
        dropout=0.15,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Load Chameleon Cache & Predictions
    cham_path = ROOT_DIR / "data" / "universal_v4" / "chameleon" / "chameleon_cache.pt"
    print(f"Loading Chameleon cache ({cham_path.name})...")
    d_cham = torch.load(cham_path, map_location="cpu", weights_only=True)

    all_phys_f = d_cham["physics_features"]      # [26033, 5, 14]
    all_phys_c = d_cham["physics_confidences"]   # [26033, 5, 4]
    all_dino_cls = d_cham["dinov2_cls"]          # [26033, 768]
    all_dino_reg = d_cham["dinov2_regional"]     # [26033, 5, 768]
    all_labels = d_cham["labels"].numpy()
    all_filenames = d_cham["filenames"]

    # 3. Filter for Real False Positives
    # Real photos where either Final prediction >= 0.5 OR Physics prediction >= 0.5
    # First get baseline predictions for all real photos
    real_mask = all_labels == 0
    real_indices = np.where(real_mask)[0]
    num_reals = len(real_indices)
    print(f"Total real photos in Chameleon: {num_reals:,}")

    # Load previously computed predictions to rapidly select false positives
    pred_path = ROOT_DIR / "reports" / "chameleon_v5_calibrated_predictions.parquet"
    if not pred_path.exists():
        pred_path = ROOT_DIR / "reports" / "chameleon_per_image_predictions.parquet"
    df_preds = pd.read_parquet(pred_path)

    df_real = df_preds[df_preds["label"] == 0].copy()
    fp_final_mask = df_real["prob_final"] >= 0.5
    fp_phys_mask = df_real["prob_physics"] >= 0.5
    fp_both_mask = fp_final_mask & fp_phys_mask

    print(f"Real False Positives (Final >= 0.5):   {fp_final_mask.sum():,} ({fp_final_mask.mean()*100:.2f}%)")
    print(f"Real False Positives (Physics >= 0.5): {fp_phys_mask.sum():,} ({fp_phys_mask.mean()*100:.2f}%)")
    print(f"Real False Positives (Both >= 0.5):    {fp_both_mask.sum():,} ({fp_both_mask.mean()*100:.2f}%)")

    # Focus analysis on the population of false positives
    # Subsets to ablate:
    # A) All Real False Positives (Final >= 0.5)
    # B) Top 200 Worst False Positives (Ranked by highest prob_final)
    # C) Physics-Specific False Positives (Physics >= 0.5)
    target_idx_all_fp = df_real[fp_final_mask].index.values
    target_idx_top200 = df_real.sort_values(by="prob_final", ascending=False).head(200).index.values
    target_idx_phys_fp = df_real[fp_phys_mask].index.values

    subsets = {
        "Top 200 Extreme False Positives": target_idx_top200,
        "All Real False Positives (Final >= 0.5)": target_idx_all_fp,
        "Physics False Positives (Physics >= 0.5)": target_idx_phys_fp,
    }

    results = {}

    for subset_name, indices in subsets.items():
        print(f"\n" + "=" * 80)
        print(f"RUNNING ABLATION ON: {subset_name} (N = {len(indices):,})")
        print("=" * 80)

        sub_f = all_phys_f[indices]
        sub_c = all_phys_c[indices]
        sub_dc = all_dino_cls[indices]
        sub_dr = all_dino_reg[indices]

        # Compute Baseline
        batch_size = 256
        n_samples = len(indices)

        def run_eval(f_t, c_t):
            p_final_list, p_phys_list = [], []
            with torch.no_grad():
                for b_start in range(0, n_samples, batch_size):
                    b_f = apply_standardizer(f_t[b_start : b_start + batch_size], standardizer).to(device)
                    b_c = c_t[b_start : b_start + batch_size].to(device)
                    b_dc = sub_dc[b_start : b_start + batch_size].to(device)
                    b_dr = sub_dr[b_start : b_start + batch_size].to(device)

                    out = model(
                        physics_features=b_f,
                        physics_confidences=b_c,
                        dinov2_cls=b_dc,
                        dinov2_regional=b_dr,
                    )
                    p_final_list.extend(out["prob_final"].cpu().numpy().tolist())
                    p_phys_list.extend(out["prob_physics"].cpu().numpy().tolist())
            return np.array(p_final_list), np.array(p_phys_list)

        base_final, base_phys = run_eval(sub_f, sub_c)
        print(f"  Baseline Mean Prob Final:   {base_final.mean():.4f}")
        print(f"  Baseline Mean Prob Physics: {base_phys.mean():.4f}")

        # 1. Ablate Each Physical Entity
        entity_deltas_phys = {}
        entity_deltas_final = {}
        print(f"\n  --- Per-Entity Ablation (Neutralizing One Physical Cue) ---")
        print(f"  {'Entity':<42} {'Mean Delta P_phys':<20} {'Mean Delta P_final':<20} {'Max Drop P_phys':<16}")
        print("  " + "-" * 95)

        for e_idx, (e_name, (f_start, f_end)) in enumerate(zip(ENTITY_NAMES, ENTITY_RANGES)):
            abl_f = sub_f.clone()
            abl_c = sub_c.clone()
            # Neutralize entity: zero out feature and confidence (activates missing token)
            abl_f[:, :, f_start:f_end] = 0.0
            abl_c[:, :, e_idx] = 0.0

            abl_final, abl_phys = run_eval(abl_f, abl_c)
            # Delta = Baseline - Ablated (positive means removing the cue reduced the fake score)
            delta_phys = base_phys - abl_phys
            delta_final = base_final - abl_final

            entity_deltas_phys[e_name] = {
                "mean_delta": float(delta_phys.mean()),
                "std_delta": float(delta_phys.std()),
                "median_delta": float(np.median(delta_phys)),
                "max_drop": float(delta_phys.max()),
                "pct_reduced": float((delta_phys > 0.01).mean() * 100),
            }
            entity_deltas_final[e_name] = {
                "mean_delta": float(delta_final.mean()),
                "std_delta": float(delta_final.std()),
                "median_delta": float(np.median(delta_final)),
                "max_drop": float(delta_final.max()),
                "pct_reduced": float((delta_final > 0.01).mean() * 100),
            }

            print(f"  {e_name:<42} {delta_phys.mean():<+20.4f} {delta_final.mean():<+20.4f} {delta_phys.max():<16.4f}")

        # 2. Ablate Each Spatial Region
        region_deltas_phys = {}
        region_deltas_final = {}
        print(f"\n  --- Per-Region Ablation (Neutralizing One Spatial Quadrant) ---")
        print(f"  {'Region':<30} {'Mean Delta P_phys':<20} {'Mean Delta P_final':<20} {'Max Drop P_phys':<16}")
        print("  " + "-" * 85)

        for r_idx, r_name in enumerate(REGION_NAMES):
            abl_f = sub_f.clone()
            abl_c = sub_c.clone()
            # Neutralize spatial region
            abl_f[:, r_idx, :] = 0.0
            abl_c[:, r_idx, :] = 0.0

            abl_final, abl_phys = run_eval(abl_f, abl_c)
            delta_phys = base_phys - abl_phys
            delta_final = base_final - abl_final

            region_deltas_phys[r_name] = {
                "mean_delta": float(delta_phys.mean()),
                "std_delta": float(delta_phys.std()),
                "median_delta": float(np.median(delta_phys)),
                "max_drop": float(delta_phys.max()),
            }
            region_deltas_final[r_name] = {
                "mean_delta": float(delta_final.mean()),
                "std_delta": float(delta_final.std()),
                "median_delta": float(np.median(delta_final)),
                "max_drop": float(delta_final.max()),
            }

            print(f"  {r_name:<30} {delta_phys.mean():<+20.4f} {delta_final.mean():<+20.4f} {delta_phys.max():<16.4f}")

        results[subset_name] = {
            "count": n_samples,
            "baseline_mean_p_final": float(base_final.mean()),
            "baseline_mean_p_phys": float(base_phys.mean()),
            "entity_ablation_physics": entity_deltas_phys,
            "entity_ablation_final": entity_deltas_final,
            "region_ablation_physics": region_deltas_phys,
            "region_ablation_final": region_deltas_final,
        }

    # 4. Detailed Top-10 Image Inspection
    print("\n" + "=" * 80)
    print("DETAILED CASE INSPECTION: TOP 10 WORST FALSE POSITIVES WITH PHYSICAL ATTRIBUTION")
    print("=" * 80)

    top10_idx = target_idx_top200[:10]
    top10_records = []

    real_img_dir = ROOT_DIR / "data" / "Chameleon" / "Chameleon" / "test" / "0_real"

    for rank, idx in enumerate(top10_idx, 1):
        fname = all_filenames[idx]
        img_path = real_img_dir / fname

        img_w, img_h = -1, -1
        if img_path.exists():
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                pass

        sample_f = all_phys_f[idx : idx + 1]
        sample_c = all_phys_c[idx : idx + 1]
        sample_dc = all_dino_cls[idx : idx + 1]
        sample_dr = all_dino_reg[idx : idx + 1]

        # Base prediction
        with torch.no_grad():
            b_f = apply_standardizer(sample_f, standardizer).to(device)
            b_c = sample_c.to(device)
            b_dc = sample_dc.to(device)
            b_dr = sample_dr.to(device)
            b_out = model(
                physics_features=b_f,
                physics_confidences=b_c,
                dinov2_cls=b_dc,
                dinov2_regional=b_dr,
            )

        b_p_final = float(b_out["prob_final"].cpu().item())
        b_p_sem = float(b_out["prob_semantic"].cpu().item())
        b_p_phys = float(b_out["prob_physics"].cpu().item())
        b_alpha = float(b_out["alpha"].cpu().item())

        # Test entity drops on this specific image
        e_drops = {}
        for e_idx, (e_name, (f_start, f_end)) in enumerate(zip(ENTITY_NAMES, ENTITY_RANGES)):
            test_f = sample_f.clone()
            test_c = sample_c.clone()
            test_f[:, :, f_start:f_end] = 0.0
            test_c[:, :, e_idx] = 0.0
            with torch.no_grad():
                o = model(
                    physics_features=apply_standardizer(test_f, standardizer).to(device),
                    physics_confidences=test_c.to(device),
                    dinov2_cls=b_dc,
                    dinov2_regional=b_dr,
                )
            p_p = float(o["prob_physics"].cpu().item())
            e_drops[e_name.split(":")[1].strip()] = round(b_p_phys - p_p, 4)

        # Identify primary culprit
        sorted_drops = sorted(e_drops.items(), key=lambda x: x[1], reverse=True)
        top_culprit = sorted_drops[0]

        top10_records.append({
            "rank": rank,
            "filename": fname,
            "resolution": f"{img_w}x{img_h}" if img_w > 0 else "unknown",
            "prob_final": b_p_final,
            "prob_semantic": b_p_sem,
            "prob_physics": b_p_phys,
            "alpha": b_alpha,
            "primary_culprit_entity": top_culprit[0],
            "primary_drop": top_culprit[1],
            "all_entity_drops": e_drops,
        })

        print(f"Rank #{rank:02d} | File: {fname[:32]}... ({img_w}x{img_h})")
        print(f"  P_final={b_p_final:.4f} | P_sem={b_p_sem:.4f} | P_phys={b_p_phys:.4f} | alpha={b_alpha:.2f}")
        print(f"  Top Culprit Entity: {top_culprit[0]} (Drop Delta P_phys: {top_culprit[1]:+.4f})")
        print(f"  All Entity Deltas: {e_drops}")
        print("-" * 80)

    # Save complete JSON report
    report_out = ROOT_DIR / "reports" / "chameleon_physical_entity_ablation.json"
    with open(report_out, "w") as f:
        json.dump({
            "ablation_subsets": results,
            "top_10_worst_false_positives": top10_records,
        }, f, indent=2)

    print(f"\n[OK] Ablation analysis complete! Full report saved to {report_out.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
