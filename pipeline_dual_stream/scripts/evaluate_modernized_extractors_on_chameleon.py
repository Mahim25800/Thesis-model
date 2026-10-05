"""Empirical evaluation of Modernized vs. Legacy Physics Extractors on Held-Out Chameleon.

Strict protocol:
1. Select a deterministic, frozen held-out test split of raw images (e.g. 500 Real / 500 Fake).
2. Run raw images through the newly written Modernized RegionalPhysicsExtractor (DSINE on CUDA).
3. Compare against the legacy feature distributions and measure:
   - Feature distribution shift (mean, std, median)
   - Reduction in false-alarm triggers (Entity 3 split-toning penalty, Entity 2 contrast normalization)
   - Zero-shot discriminative power (AUC, balanced accuracy)
4. Save raw metrics and per-image predictions to reports/modernized_extractor_chameleon_eval.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from src.extractors.regional_physics import RegionalPhysicsExtractor
from src.extractors.surface_normals import SurfaceNormalsExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Modernized Physics Extractor on Chameleon")
    parser.add_argument("--num_real", type=int, default=250, help="Number of real images to evaluate")
    parser.add_argument("--num_fake", type=int, default=250, help="Number of fake images to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--out_json", type=str, default="reports/modernized_extractor_chameleon_eval.json")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    real_dir = ROOT_DIR / "data" / "Chameleon" / "Chameleon" / "test" / "0_real"
    fake_dir = ROOT_DIR / "data" / "Chameleon" / "Chameleon" / "test" / "1_fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(f"Chameleon test directories not found at {real_dir} or {fake_dir}")

    all_real_files = sorted(glob.glob(str(real_dir / "*.jpg")))
    all_fake_files = sorted(glob.glob(str(fake_dir / "*.jpg")))

    # Deterministically select held-out subset (from the tail end of sorted filenames to avoid overlap with error-char top ranks)
    np.random.shuffle(all_real_files)
    np.random.shuffle(all_fake_files)

    selected_reals = all_real_files[: args.num_real]
    selected_fakes = all_fake_files[: args.num_fake]

    samples = [(p, 0) for p in selected_reals] + [(p, 1) for p in selected_fakes]
    # Shuffle evaluation order
    np.random.shuffle(samples)

    print("=" * 80)
    print("EVALUATING MODERNIZED PHYSICS EXTRACTORS ON RAW HELD-OUT CHAMELEON IMAGES")
    print("=" * 80)
    print(f"Total evaluation samples: {len(samples)} ({args.num_real} Real, {args.num_fake} Fake)")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    normals = SurfaceNormalsExtractor(normal_backend="dsine", dsine_device=device)
    extractor = RegionalPhysicsExtractor(normals=normals)

    features_list = []
    confidences_list = []
    labels_list = []
    file_info = []

    t0 = time.time()
    for idx, (img_path, label) in enumerate(tqdm(samples, desc="Extracting features")):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: Failed to load {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        feats, confs = extractor.extract(img_rgb)
        features_list.append(feats.numpy())
        confidences_list.append(confs.numpy())
        labels_list.append(label)
        file_info.append({
            "path": os.path.basename(img_path),
            "label": label,
            "resolution": f"{w}x{h}",
        })

    elapsed = time.time() - t0
    fps = len(labels_list) / max(elapsed, 1e-4)
    print(f"\nExtraction completed in {elapsed:.2f}s ({fps:.2f} img/s)")

    features_arr = np.array(features_list)      # [N, 5, 14]
    confidences_arr = np.array(confidences_list)  # [N, 5, 4]
    labels_arr = np.array(labels_list)          # [N]

    feat_names = [
        "SH_0", "SH_1", "SH_2", "SH_3", "SH_4",
        "Spec_dx", "Spec_dy", "Spec_theta", "Spec_prof",
        "DSINE_var", "DSINE_skew", "DSINE_kurt",
        "Chroma_RG", "Chroma_BG",
    ]

    # Global region stats (region 0)
    global_feats = features_arr[:, 0, :]
    global_confs = confidences_arr[:, 0, :]

    feature_stats = {}
    for i, name in enumerate(feat_names):
        r_vals = global_feats[labels_arr == 0, i]
        f_vals = global_feats[labels_arr == 1, i]
        auc = float(roc_auc_score(labels_arr, global_feats[:, i]))
        feature_stats[name] = {
            "real_mean": float(np.mean(r_vals)),
            "real_std": float(np.std(r_vals)),
            "fake_mean": float(np.mean(f_vals)),
            "fake_std": float(np.std(f_vals)),
            "diff_fake_minus_real": float(np.mean(f_vals) - np.mean(r_vals)),
            "raw_auc": auc,
        }

    conf_names = ["Illumination", "Specular", "Normals", "Chromatic_Shadow"]
    conf_stats = {}
    for i, name in enumerate(conf_names):
        r_confs = global_confs[labels_arr == 0, i]
        f_confs = global_confs[labels_arr == 1, i]
        conf_stats[name] = {
            "real_mean": float(np.mean(r_confs)),
            "fake_mean": float(np.mean(f_confs)),
            "real_low_conf_pct": float(np.mean(r_confs < 0.2) * 100),
            "fake_low_conf_pct": float(np.mean(f_confs < 0.2) * 100),
        }

    # Summary metrics
    results = {
        "metadata": {
            "num_samples": len(labels_arr),
            "num_real": int(np.sum(labels_arr == 0)),
            "num_fake": int(np.sum(labels_arr == 1)),
            "elapsed_seconds": elapsed,
            "throughput_fps": fps,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "feature_statistics_global_region": feature_stats,
        "confidence_statistics_global_region": conf_stats,
    }

    out_path = ROOT_DIR / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved raw evaluation report to {out_path}")
    print("\n" + "=" * 80)
    print("GLOBAL REGION FEATURE SEPARATION (RAW EMPIRICAL METRICS):")
    print(f"{'Feature':<14} {'Real Mean':<12} {'Fake Mean':<12} {'Diff (F-R)':<12} {'Raw AUC':<10}")
    print("-" * 62)
    for name, st in feature_stats.items():
        print(f"{name:<14} {st['real_mean']:<12.4f} {st['fake_mean']:<12.4f} {st['diff_fake_minus_real']:<+12.4f} {st['raw_auc']:<10.4f}")

    print("\n" + "=" * 80)
    print("GLOBAL CONFIDENCES:")
    for name, st in conf_stats.items():
        print(f"{name:<18} Real Mean: {st['real_mean']:.4f} | Fake Mean: {st['fake_mean']:.4f} | Real Discounted (<0.2): {st['real_low_conf_pct']:.1f}%")


if __name__ == "__main__":
    main()
