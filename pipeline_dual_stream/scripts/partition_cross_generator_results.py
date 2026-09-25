"""Partitioned Cross-Generator Generalization Benchmark.
Separates genuinely novel architectures (ADM, BigGAN, VQDM, GLIDE, Midjourney)
from the SD-family architectures (SDv4, SDv5, Wukong) to provide full transparency.
"""

import json
from pathlib import Path
import numpy as np

reports_dir = Path("G:/Thesis/pipeline_dual_stream/reports")

novel_gens = ["adm", "biggan", "vqdm", "glide", "midjourney"]
sd_family = ["sdv4", "sdv5", "wukong"]

def summarize_group(names, group_name):
    phys_aucs = []
    dino_aucs = []
    hybrid_aucs = []
    deltas = []
    total_samples = 0
    
    print(f"=== {group_name} ===")
    print(f"{'Generator':<15} | {'Samples':<8} | {'Physics':<10} | {'DINOv2':<10} | {'Hybrid':<10} | {'Synergy Gain':<12}")
    print("-" * 75)
    
    for g in names:
        p = reports_dir / f"{g}_evaluation.json"
        with open(p) as f:
            d = json.load(f)
        n = d["sample_count"]
        total_samples += n
        phys = d["metrics"]["physics_alone"]["roc_auc"]
        dino = d["metrics"]["semantic_dinov2_alone"]["roc_auc"]
        hyb = d["metrics"]["hybrid"]["roc_auc"]
        delta = d["metrics"]["synergy_delta"]
        
        phys_aucs.append(phys)
        dino_aucs.append(dino)
        hybrid_aucs.append(hyb)
        deltas.append(delta)
        
        print(f"{g:<15} | {n:<8} | {phys:<10.4f} | {dino:<10.4f} | {hyb:<10.4f} | {delta:+10.4f}")
        
    print("-" * 75)
    print(f"{'MEAN':<15} | {total_samples:<8} | {np.mean(phys_aucs):<10.4f} | {np.mean(dino_aucs):<10.4f} | {np.mean(hybrid_aucs):<10.4f} | {np.mean(deltas):+10.4f}\n")
    return {
        "mean_physics": float(np.mean(phys_aucs)),
        "mean_dinov2": float(np.mean(dino_aucs)),
        "mean_hybrid": float(np.mean(hybrid_aucs)),
        "mean_delta": float(np.mean(deltas)),
        "total_samples": total_samples
    }

if __name__ == "__main__":
    res_novel = summarize_group(novel_gens, "1. GENUINELY NOVEL ARCHITECTURES (Non-SD Family)")
    res_sd = summarize_group(sd_family, "2. SD-FAMILY ARCHITECTURES (Latent Diffusion Overlap)")
    
    summary = {
        "novel_architectures": res_novel,
        "sd_family": res_sd
    }
    with open(reports_dir / "partitioned_cross_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
