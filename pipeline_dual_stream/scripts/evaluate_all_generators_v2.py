import subprocess
import sys
from pathlib import Path

ROOT = Path("G:/Thesis/pipeline_dual_stream")
PYTHON = r"G:\Thesis\.venv\Scripts\python.exe"

generators = [
    {
        "name": "adm",
        "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_regional_v2/adm_features.pt",
        "dino": "data/dinov2_cache_adm.pt",
        "report": "reports/v2_adm_eval.json",
    },
    {
        "name": "biggan",
        "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_regional_v2/biggan_features.pt",
        "dino": "data/dinov2_cache_biggan.pt",
        "report": "reports/v2_biggan_eval.json",
    },
    {
        "name": "vqdm",
        "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_regional_v2/vqdm_features.pt",
        "dino": "data/dinov2_cache_vqdm.pt",
        "report": "reports/v2_vqdm_eval.json",
    },
    {
        "name": "wukong",
        "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_regional_v2/wukong_features.pt",
        "dino": "data/dinov2_cache_wukong.pt",
        "report": "reports/v2_wukong_eval.json",
    },
    {
        "name": "glide",
        "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_confirmatory_disjoint_regional_v2/glide_features.pt",
        "dino": "data/dinov2_cache_glide.pt",
        "report": "reports/v2_glide_eval.json",
    },
    {
        "name": "midjourney",
        "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_confirmatory_disjoint_regional_v2/midjourney_features.pt",
        "dino": "data/dinov2_cache_midjourney.pt",
        "report": "reports/v2_midjourney_eval.json",
    },
    {
        "name": "sdv4",
        "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_confirmatory_disjoint_regional_v2/sdv4_features.pt",
        "dino": "data/dinov2_cache_sdv4.pt",
        "report": "reports/v2_sdv4_eval.json",
    },
    {
        "name": "sdv5",
        "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_confirmatory_disjoint_regional_v2/sdv5_features.pt",
        "dino": "data/dinov2_cache_sdv5.pt",
        "report": "reports/v2_sdv5_eval.json",
    },
]

print("Starting Batch Cross-Generator Evaluation for Dual Stream v2...")
for g in generators:
    print(f"\n---> Evaluating {g['name'].upper()}...")
    cmd = [
        PYTHON,
        "scripts/evaluate_dual_stream.py",
        "--model-path", "models/dual_stream_v2/best_model.pt",
        "--physics-cache", g["phys"],
        "--dinov2-cache", g["dino"],
        "--output-report", g["report"],
        "--bootstrap-repeats", "200",
        "--device", "cuda",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error on {g['name']}: {res.stderr}")
    else:
        print(f"Completed {g['name']}.")

print("\nAll 8 generator evaluations completed!")
