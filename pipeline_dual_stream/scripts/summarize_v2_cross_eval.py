import json
from pathlib import Path
import numpy as np

rep_dir = Path("G:/Thesis/pipeline_dual_stream/reports")
novel_gens = ["adm", "biggan", "vqdm", "glide", "midjourney"]
sd_family = ["sdv4", "sdv5", "wukong"]

def summarize_subset(names, title):
    hyb_list, dino_list, phys_list, deltas = [], [], [], []
    total_samples = 0
    print(f"\n=== {title} ===")
    print(f"{'Generator':<12} | {'Samples':<7} | {'DINOv2':<9} | {'Physics':<9} | {'Dual-Stream v2':<14} | {'Synergy':<9} | {'Alpha':<6}")
    print("-" * 75)
    for g in names:
        p = rep_dir / f"v2_{g}_eval.json"
        with open(p) as f:
            d = json.load(f)
        n = d["sample_count"]
        total_samples += n
        dino = d["metrics"]["semantic_dinov2_alone"]["roc_auc"]
        phys = d["metrics"]["physics_alone"]["roc_auc"]
        hyb = d["metrics"]["hybrid"]["roc_auc"]
        delta = d["metrics"]["synergy_delta"]
        alpha = d["trust_gating"]["mean_alpha"]
        
        hyb_list.append(hyb)
        dino_list.append(dino)
        phys_list.append(phys)
        deltas.append(delta)
        print(f"{g.upper():<12} | {n:<7} | {dino:<9.4f} | {phys:<9.4f} | {hyb:<14.4f} | {delta:+9.4f} | {alpha:<6.2f}")
    print("-" * 75)
    print(f"{'MEAN':<12} | {total_samples:<7} | {np.mean(dino_list):<9.4f} | {np.mean(phys_list):<9.4f} | {np.mean(hyb_list):<14.4f} | {np.mean(deltas):+9.4f}")
    return {
        "mean_dino": float(np.mean(dino_list)),
        "mean_phys": float(np.mean(phys_list)),
        "mean_hyb": float(np.mean(hyb_list)),
        "mean_delta": float(np.mean(deltas)),
        "total_samples": total_samples,
    }

novel_res = summarize_subset(novel_gens, "1. GENUINELY NOVEL ARCHITECTURES (Non-SD Family)")
sd_res = summarize_subset(sd_family, "2. SD-FAMILY ARCHITECTURES (Latent Diffusion Overlap)")
all_res = summarize_subset(novel_gens + sd_family, "3. OVERALL (ALL 8 GENERATOR DOMAINS, 24,000 SAMPLES)")

summary = {
    "novel_architectures": novel_res,
    "sd_family": sd_res,
    "overall": all_res,
}

with open(rep_dir / "v2_partitioned_cross_eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSummary saved to reports/v2_partitioned_cross_eval_summary.json")
