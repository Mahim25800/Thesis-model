"""Metadata & Shortcut Correlation Audit for DINOv2 vs Physics Stream.
Computes Pearson, Spearman, and within-class partial correlations with:
- Image resolution (width, height, total pixels)
- Aspect ratio
- File size and bytes-per-pixel
- JPEG Quantization statistics
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from scipy import stats
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.hybrid_detector import DualStreamHybridDetector

def run_audit():
    print("Loading model and caches...")
    checkpoint = torch.load("G:/Thesis/pipeline_dual_stream/models/dual_stream_v1/best_model.pt", map_location="cuda")
    model = DualStreamHybridDetector().cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load features
    st = json.load(open("G:/Thesis/pipeline_dual_stream/models/dual_stream_v1/standardizer.json"))
    mean = torch.tensor(st["mean"], dtype=torch.float32).cuda()
    scale = torch.tensor(st["scale"], dtype=torch.float32).cuda()

    phys_cache = torch.load("G:/Thesis/pipeline_40k/data/cache_regional_dsine_v2/test_unseen_features.pt", map_location="cuda")
    features = ((phys_cache["features"] - mean.unsqueeze(0)) / scale.unsqueeze(0)).cuda()
    confs = phys_cache["confidences"].cuda()
    labels = phys_cache["labels"].cpu().numpy()

    sem_cache = torch.load("G:/Thesis/pipeline_dual_stream/data/dinov2_cache_test_unseen.pt", map_location="cuda")
    sem_cls = sem_cache["dinov2_cls"].cuda()
    sem_reg = sem_cache["dinov2_regional"].cuda()

    with torch.no_grad():
        out = model(physics_features=features, physics_confidences=confs, dinov2_cls=sem_cls, dinov2_regional=sem_reg)
        prob_sem = out["prob_semantic"].cpu().numpy()
        prob_phys = out["prob_physics"].cpu().numpy()
        prob_final = out["prob_final"].cpu().numpy()

    print(f"Computed predictions for {len(labels)} samples.")

    # Extract metadata for all 8000 images
    widths, heights, aspect_ratios, total_pixels, file_sizes, bytes_per_px, jpeg_mean_qs = [], [], [], [], [], [], []

    manifest_path = "G:/Thesis/pipeline_40k/data/cache_regional_dsine_v2/test_unseen_manifest.jsonl"
    with open(manifest_path) as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            img_path = Path("G:/Thesis/pipeline_40k") / data["relative_path"]
            w = data["width"]
            h = data["height"]
            ar = w / max(h, 1)
            tp = w * h
            fsize = os.path.getsize(img_path)
            bpp = fsize / max(tp, 1)
            
            q_val = 0.0
            try:
                with Image.open(img_path) as im:
                    if hasattr(im, "quantization") and im.quantization and 0 in im.quantization:
                        q_val = float(np.mean(im.quantization[0]))
            except Exception:
                pass
                
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(ar)
            total_pixels.append(tp)
            file_sizes.append(fsize)
            bytes_per_px.append(bpp)
            jpeg_mean_qs.append(q_val)

    metadata_dict = {
        "Width": np.array(widths),
        "Height": np.array(heights),
        "Aspect Ratio": np.array(aspect_ratios),
        "Total Pixels (Res)": np.array(total_pixels),
        "File Size (Bytes)": np.array(file_sizes),
        "Bytes per Pixel": np.array(bytes_per_px),
        "JPEG Quant Table Mean": np.array(jpeg_mean_qs)
    }

    results = {}

    print("\n" + "="*80)
    print("METADATA & SHORTCUT CORRELATION AUDIT (N=8,000 Unseen Test Samples)")
    print("="*80)
    print(f"{'Metadata Feature':<24} | {'DINOv2 r':<10} | {'DINOv2 rho':<12} | {'Physics rho':<12} | {'Label rho':<12}")
    print("-" * 80)

    for name, meta in metadata_dict.items():
        r_sem, p_r_sem = stats.pearsonr(meta, prob_sem)
        rho_sem, p_rho_sem = stats.spearmanr(meta, prob_sem)
        rho_phys, p_rho_phys = stats.spearmanr(meta, prob_phys)
        rho_lbl, p_rho_lbl = stats.spearmanr(meta, labels)
        results[name] = {
            "dinov2_pearson_r": float(r_sem),
            "dinov2_pearson_p": float(p_r_sem),
            "dinov2_spearman_rho": float(rho_sem),
            "dinov2_spearman_p": float(p_rho_sem),
            "physics_spearman_rho": float(rho_phys),
            "physics_spearman_p": float(p_rho_phys),
            "label_spearman_rho": float(rho_lbl),
            "label_spearman_p": float(p_rho_lbl)
        }
        print(f"{name:<24} | {r_sem:+.4f}     | {rho_sem:+.4f}       | {rho_phys:+.4f}        | {rho_lbl:+.4f}")

    print("\n" + "="*80)
    print("WITHIN-CLASS PARTIAL CORRELATIONS (Controlling for True Label)")
    print("="*80)
    print(f"{'Metadata Feature':<24} | {'Real DINOv2 r':<14} | {'Real Phys r':<12} | {'Fake DINOv2 r':<14} | {'Fake Phys r':<12}")
    print("-" * 80)

    real_mask = (labels == 0)
    fake_mask = (labels == 1)
    within_class = {}

    for name, meta in metadata_dict.items():
        r_sem_real, _ = stats.pearsonr(meta[real_mask], prob_sem[real_mask])
        r_phys_real, _ = stats.pearsonr(meta[real_mask], prob_phys[real_mask])
        r_sem_fake, _ = stats.pearsonr(meta[fake_mask], prob_sem[fake_mask])
        r_phys_fake, _ = stats.pearsonr(meta[fake_mask], prob_phys[fake_mask])
        within_class[name] = {
            "real_dinov2_r": float(r_sem_real),
            "real_phys_r": float(r_phys_real),
            "fake_dinov2_r": float(r_sem_fake),
            "fake_phys_r": float(r_phys_fake)
        }
        print(f"{name:<24} | {r_sem_real:+.4f}         | {r_phys_real:+.4f}       | {r_sem_fake:+.4f}         | {r_phys_fake:+.4f}")

    # Save to JSON
    report_path = ROOT_DIR / "reports" / "metadata_shortcut_audit.json"
    with open(report_path, "w") as f:
        json.dump({"overall_correlations": results, "within_class_correlations": within_class}, f, indent=2)
    print(f"\nAudit saved to {report_path}")

if __name__ == "__main__":
    run_audit()
