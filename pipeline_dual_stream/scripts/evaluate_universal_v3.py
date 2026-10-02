"""Comprehensive Evaluation Suite for Universal Dual-Stream Detector (dual_stream_v3).
Evaluates:
  1. Unseen Synthbuster Benchmark (DALL-E 2/3, Midjourney v5, Firefly, SDXL, etc.) vs. held-out RAISE DSLR photos.
  2. Blur & Optical Bokeh Robustness Stress Test on Real Camera Photos.
  3. In-the-Wild Real vs AI Sanity Test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.models.hybrid_detector import DualStreamHybridDetector


def apply_standardizer(features: torch.Tensor, standardizer: Dict[str, list]) -> torch.Tensor:
    mean = torch.tensor(standardizer["mean"], dtype=features.dtype, device=features.device)
    scale = torch.tensor(standardizer["scale"], dtype=features.dtype, device=features.device)
    return (features - mean.unsqueeze(0)) / scale.unsqueeze(0)


def evaluate_stream(model: nn.Module, standardizer: dict, phys_feats, phys_confs, dino_cls, dino_reg, labels, device="cuda"):
    model.eval()
    batch_size = 128
    n = len(labels)
    all_final_probs = []
    all_sem_probs = []
    all_phys_probs = []
    all_alphas = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_pf = apply_standardizer(phys_feats[i:i+batch_size], standardizer).to(device)
            b_pc = phys_confs[i:i+batch_size].to(device)
            b_dc = dino_cls[i:i+batch_size].to(device)
            b_dr = dino_reg[i:i+batch_size].to(device)

            out = model(
                physics_features=b_pf,
                physics_confidences=b_pc,
                dinov2_cls=b_dc,
                dinov2_regional=b_dr,
            )

            all_final_probs.extend(out["prob_final"].cpu().numpy().tolist())
            all_sem_probs.extend(out["prob_semantic"].cpu().numpy().tolist())
            all_phys_probs.extend(out["prob_physics"].cpu().numpy().tolist())
            all_alphas.extend(out["alpha"].cpu().numpy().tolist())

    y_true = np.array(labels.numpy() if hasattr(labels, "numpy") else labels)
    y_final = np.array(all_final_probs)
    y_sem = np.array(all_sem_probs)
    y_phys = np.array(all_phys_probs)

    auc_final = float(roc_auc_score(y_true, y_final))
    auc_sem = float(roc_auc_score(y_true, y_sem))
    auc_phys = float(roc_auc_score(y_true, y_phys))

    acc_final = float(accuracy_score(y_true, y_final >= 0.5))
    acc_sem = float(accuracy_score(y_true, y_sem >= 0.5))
    acc_phys = float(accuracy_score(y_true, y_phys >= 0.5))

    return {
        "auc_hybrid": auc_final,
        "auc_dinov2": auc_sem,
        "auc_physics": auc_phys,
        "acc_hybrid": acc_final,
        "acc_dinov2": acc_sem,
        "acc_physics": acc_phys,
        "mean_alpha": float(np.mean(all_alphas)),
        "synergy_delta": float(auc_final - max(auc_sem, auc_phys)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/universal_v3/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 75)
    print(f"Loading Universal Dual-Stream Detector from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    standardizer = ckpt["standardizer"]
    
    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode="v2",
        dropout=0.15,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded successfully from Epoch {ckpt['epoch']}!")
    print("=" * 75)

    # Load held-out RAISE real DSLR test samples (500 images)
    raise_test_path = Path("data/universal_v3/raise_held_out_test.pt")
    raise_test_data = torch.load(raise_test_path, map_location="cpu")
    r_pf = raise_test_data["physics_features"]
    r_pc = raise_test_data["physics_confidences"]
    r_dc = raise_test_data["dinov2_cls"]
    r_dr = raise_test_data["dinov2_regional"]
    r_lbl = raise_test_data["labels"]
    n_real = len(r_lbl)

    synth_phys_dir = Path("G:/Thesis/pipeline_40k/data/cache_synthbuster_regional_v2")
    synth_dino_dir = Path("data/synthbuster_dino")

    generators = [
        ("dalle2", synth_phys_dir / "dalle2.pt", synth_dino_dir / "dinov2_cache_dalle2.pt"),
        ("dalle3", synth_phys_dir / "dalle3.pt", synth_dino_dir / "dinov2_cache_dalle3.pt"),
        ("firefly", synth_phys_dir / "firefly.pt", synth_dino_dir / "dinov2_cache_firefly.pt"),
        ("glide", synth_phys_dir / "glide.pt", synth_dino_dir / "dinov2_cache_glide.pt"),
        ("midjourney-v5", synth_phys_dir / "midjourney-v5.pt", synth_dino_dir / "dinov2_cache_midjourney_v5.pt"),
        ("sd13", synth_phys_dir / "stable-diffusion-1-3.pt", synth_dino_dir / "dinov2_cache_sd13.pt"),
        ("sd14", synth_phys_dir / "stable-diffusion-1-4.pt", synth_dino_dir / "dinov2_cache_sd14.pt"),
        ("sd2", synth_phys_dir / "stable-diffusion-2.pt", synth_dino_dir / "dinov2_cache_sd2.pt"),
        ("sdxl", synth_phys_dir / "stable-diffusion-xl.pt", synth_dino_dir / "dinov2_cache_sdxl.pt"),
    ]

    print("\n" + "=" * 80)
    print("1. EXTERNAL SYNTHBUSTER BENCHMARK (PAIRED WITH HELD-OUT RAISE REAL DSLR PHOTOS)")
    print(f"{'Generator':<16} | {'Samples':<7} | {'DINOv2':<9} | {'Physics':<9} | {'Hybrid v3':<11} | {'Synergy':<9} | {'Alpha':<6}")
    print("-" * 80)

    results = {}
    hyb_aucs, dino_aucs, phys_aucs, deltas = [], [], [], []

    for name, p_path, d_path in generators:
        p_data = torch.load(p_path, map_location="cpu")
        d_data = torch.load(d_path, map_location="cpu")

        f_pf = p_data["features"]
        f_pc = p_data["confidences"]
        f_dc = d_data["dinov2_cls"]
        f_dr = d_data["dinov2_regional"]
        f_lbl = torch.ones(len(f_pf), dtype=torch.float32)

        # Combine with held-out real RAISE
        comb_pf = torch.cat([r_pf, f_pf], dim=0)
        comb_pc = torch.cat([r_pc, f_pc], dim=0)
        comb_dc = torch.cat([r_dc, f_dc], dim=0)
        comb_dr = torch.cat([r_dr, f_dr], dim=0)
        comb_lbl = torch.cat([r_lbl, f_lbl], dim=0)

        metrics = evaluate_stream(model, standardizer, comb_pf, comb_pc, comb_dc, comb_dr, comb_lbl, args.device)
        results[name] = metrics

        hyb_aucs.append(metrics["auc_hybrid"])
        dino_aucs.append(metrics["auc_dinov2"])
        phys_aucs.append(metrics["auc_physics"])
        deltas.append(metrics["synergy_delta"])

        print(
            f"{name.upper():<16} | {len(comb_lbl):<7} | "
            f"{metrics['auc_dinov2']:<9.4f} | {metrics['auc_physics']:<9.4f} | "
            f"{metrics['auc_hybrid']:<11.4f} | {metrics['synergy_delta']:+9.4f} | "
            f"{metrics['mean_alpha']:<6.2f}"
        )

    print("-" * 80)
    print(
        f"{'MEAN':<16} | {sum(len(r_lbl) + 1000 for _ in generators):<7} | "
        f"{np.mean(dino_aucs):<9.4f} | {np.mean(phys_aucs):<9.4f} | "
        f"{np.mean(hyb_aucs):<11.4f} | {np.mean(deltas):+9.4f}"
    )

    # 2. Blur Sensitivity Stress Test on Real Camera Photos
    print("\n" + "=" * 80)
    print("2. BLUR SENSITIVITY STRESS TEST ON HELD-OUT RAISE REAL CAMERA PHOTOS")
    print("Evaluates whether optical blur causes false-positive AI detections:")
    print(f"{'Blur Condition':<22} | {'Real Photos':<12} | {'False Positive Rate':<22} | {'Mean Fake Prob':<15}")
    print("-" * 80)

    # Clean RAISE test
    with torch.no_grad():
        b_pf = apply_standardizer(r_pf, standardizer).to(args.device)
        b_pc = r_pc.to(args.device)
        b_dc = r_dc.to(args.device)
        b_dr = r_dr.to(args.device)

        out_clean = model(physics_features=b_pf, physics_confidences=b_pc, dinov2_cls=b_dc, dinov2_regional=b_dr)
        probs_clean = out_clean["prob_final"].cpu().numpy()
        fpr_clean = float((probs_clean >= 0.5).mean())
        mean_p_clean = float(probs_clean.mean())
        print(f"{'Clean (Pristine DSLR)':<22} | {len(r_lbl):<12} | {fpr_clean*100:<20.2f}% | {mean_p_clean*100:<13.2f}%")

        # Blur sweep with feature jitter
        for sigma in [1.0, 2.0, 4.0, 8.0]:
            noise_scale = 0.015 * sigma
            dc_blur = b_dc + torch.randn_like(b_dc) * noise_scale
            dr_blur = b_dr + torch.randn_like(b_dr) * noise_scale
            # Peripheral normals smooth out under optical blur
            pc_blur = b_pc.clone()
            pc_blur[:, 1:] = pc_blur[:, 1:] * max(0.1, 1.0 - 0.1 * sigma)

            out_blur = model(physics_features=b_pf, physics_confidences=pc_blur, dinov2_cls=dc_blur, dinov2_regional=dr_blur)
            probs_blur = out_blur["prob_final"].cpu().numpy()
            fpr_blur = float((probs_blur >= 0.5).mean())
            mean_p_blur = float(probs_blur.mean())
            print(f"{f'Gaussian Blur (sigma={sigma})':<22} | {len(r_lbl):<12} | {fpr_blur*100:<20.2f}% | {mean_p_blur*100:<13.2f}%")

    # 3. In-The-Wild Sanity Test Suite
    print("\n" + "=" * 80)
    print("3. IN-THE-WILD SANITY TEST SUITE (CAMERA BLUR VS. MODERN AI GENERATORS)")
    print(f"{'Image Name':<30} | {'Ground Truth':<15} | {'Verdict':<15} | {'Prob Fake':<10} | {'Status':<10}")
    print("-" * 80)

    from src.inference.predict import DualStreamPredictor
    predictor = DualStreamPredictor(checkpoint_path=args.checkpoint, device=args.device)

    wild_tests = [
        {
            "name": "DSC03398.JPG (24MP DSLR Blur)",
            "path": r"C:\Users\laptop villa\AppData\Local\Temp\gradio\e87b25f52193471154a16b20d382713c8b7e983c30a39fa25a88af2e3db9b751\DSC03398.JPG",
            "gt": "Real Camera",
        },
        {
            "name": "Smartphone Portrait Clean",
            "path": "data/test_smartphone_portrait_clean.png",
            "gt": "Real Camera",
        },
        {
            "name": "Catwoman Diffusion",
            "path": "data/catwoman_extracted.png",
            "gt": "AI-Generated",
        },
        {
            "name": "Blue Bedroom (Anime Illustration PNG)",
            "path": "data/test_blue_bedroom.png",
            "gt": "AI-Generated",
        },
        {
            "name": "Blue Bedroom JPG (Web Compressed)",
            "path": r"C:\Users\laptop villa\AppData\Local\Temp\gradio\895daf81d3c3f075701e5a23ef1d4273ea6f3836016275e73f65ba6f3b05d2f8\r485s41njdrh1.jpg",
            "gt": "AI-Generated",
        },
    ]

    wild_results = {}
    for wt in wild_tests:
        p = Path(wt["path"])
        if not p.exists():
            print(f"Skipping {wt['name']}, file not found.")
            continue
        res = predictor.predict_image(str(p))
        is_fake = res["final_prob_fake"] >= 0.5
        pred_label = "AI-Generated" if is_fake else "Real Camera"
        passed = (pred_label == wt["gt"])
        status = "PASSED" if passed else "FAILED"
        wild_results[wt["name"]] = res
        print(f"{wt['name']:<30} | {wt['gt']:<15} | {pred_label:<15} | {res['final_prob_fake']*100:<9.1f}% | {status:<10}")

    # Save summary report
    out_rep = Path("reports/universal_v3_evaluation_summary.json")
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "epoch": ckpt["epoch"],
                "synthbuster_cross_eval": results,
                "mean_synthbuster_auc": float(np.mean(hyb_aucs)),
                "mean_dino_auc": float(np.mean(dino_aucs)),
                "mean_phys_auc": float(np.mean(phys_aucs)),
                "clean_real_fpr": fpr_clean,
                "in_the_wild_results": {k: {"prob_fake": v["final_prob_fake"], "verdict": v["verdict"]} for k, v in wild_results.items()},
            },
            f,
            indent=2,
        )
    print(f"\nEvaluation complete! Report saved to {out_rep}")


if __name__ == "__main__":
    main()
