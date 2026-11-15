"""Create publication-ready summary figures from frozen evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/figures"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def mean_external(report: dict) -> float:
    return float(np.mean([row["binary"]["auc_roc"] for row in report["generators"].values()]))


def main() -> None:
    v4_internal = load("models/dsine_multi_entity_v4/evaluation.json")["binary"]
    v6_internal = load("models/generator_invariant_v6/evaluation_internal.json")["binary"]
    v14_internal = load("models/physics_ensemble_v14/evaluation_internal.json")["binary"]
    v4_external = load("models/dsine_multi_entity_v4/external_synthbuster/evaluation.json")
    v6_external = load("models/generator_invariant_v6/external_synthbuster/evaluation.json")
    v14_external = load("models/physics_ensemble_v14/external_synthbuster/evaluation.json")
    ablation = load("models/generator_invariant_v6/external_synthbuster/ablation.json")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "figure.dpi": 160, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    blue, teal, orange, gray = "#3568A8", "#1B8A7A", "#D77A2B", "#A7ADB4"

    labels = ["Global\nDSINE v4", "Invariant\nregional v6", "Physics\nensemble"]
    internal = [v4_internal["auc_roc"], v6_internal["auc_roc"], v14_internal["auc_roc"]]
    external = [mean_external(v4_external), mean_external(v6_external), mean_external(v14_external)]
    x = np.arange(3)
    width = 0.34
    axes[0].bar(x - width / 2, internal, width, color=blue, label="Internal unseen")
    axes[0].bar(x + width / 2, external, width, color=teal, label="Synthbuster mean")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.45, 0.95)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("A. Generalization across model stages", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for bars in axes[0].containers:
        axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    generators = list(v14_external["generators"])
    short = {
        "dalle2": "DALL-E 2", "dalle3": "DALL-E 3", "firefly": "Firefly", "glide": "GLIDE",
        "midjourney-v5": "Midjourney", "stable-diffusion-1-3": "SD 1.3",
        "stable-diffusion-1-4": "SD 1.4", "stable-diffusion-2": "SD 2", "stable-diffusion-xl": "SDXL",
    }
    baseline = [v4_external["generators"][name]["binary"]["auc_roc"] for name in generators]
    candidate = [v14_external["generators"][name]["binary"]["auc_roc"] for name in generators]
    y = np.arange(len(generators))
    axes[1].hlines(y, baseline, candidate, color=gray, linewidth=2)
    axes[1].scatter(baseline, y, color=orange, s=25, label="v4")
    axes[1].scatter(candidate, y, color=teal, s=25, label="v14 ensemble")
    axes[1].set_yticks(y, [short[name] for name in generators])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.48, 0.86)
    axes[1].set_xlabel("ROC-AUC")
    axes[1].set_title("B. Every Synthbuster generator improves", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    axes[1].axvline(0.5, color="#666666", linewidth=0.8, linestyle="--")

    preferred = ["full", "global_only", "without_illumination", "without_specular_optics",
                 "without_surface_normals", "without_chromatic_shadows"]
    rows = ablation.get("ablations", ablation.get("variants", ablation))
    values = []
    for key in preferred:
        row = rows[key]
        if isinstance(row, dict) and "mean_auc" in row:
            values.append(row["mean_auc"])
        elif isinstance(row, dict) and "generators" in row:
            values.append(float(np.mean([item["auc_roc"] for item in row["generators"].values()])))
        else:
            values.append(float(row))
    ablation_labels = ["Full", "Global only", "No illumination", "No specular", "No normals", "No chromatic"]
    colors = [teal if value >= values[0] else orange for value in values]
    axes[2].barh(np.arange(len(values)), values, color=colors)
    axes[2].set_yticks(np.arange(len(values)), ablation_labels)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0.48, 0.69)
    axes[2].set_xlabel("Mean Synthbuster ROC-AUC")
    axes[2].set_title("C. Regional and normal cues are critical", loc="left", weight="bold")
    axes[2].axvline(0.5, color="#666666", linewidth=0.8, linestyle="--")
    for index, value in enumerate(values):
        axes[2].text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=7)

    fig.suptitle("Regional multi-physics detector: frozen cross-generator evidence", fontsize=13, weight="bold")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "regional_multi_physics_results.png", bbox_inches="tight")
    fig.savefig(OUTPUT / "regional_multi_physics_results.pdf", bbox_inches="tight")

    confirmation = load("models/physics_ensemble_v14/genimage_confirmatory/evaluation.json")
    confirm_models = confirmation["models"]
    v6_confirm = confirm_models["generator_invariant_v6"]
    v14_confirm = confirm_models["physics_ensemble_v14"]
    confirm_domains = confirmation["domains"]
    display_domains = {"glide": "GLIDE", "midjourney": "Midjourney", "sdv4": "SD 1.4", "sdv5": "SD 1.5"}

    confirm_fig, confirm_axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    benchmark_labels = ["Synthbuster\n9 generators", "GenImage confirmation\n4 generators"]
    v6_auc = [mean_external(v6_external), v6_confirm["summary"]["mean_auc"]]
    v14_auc = [mean_external(v14_external), v14_confirm["summary"]["mean_auc"]]
    bx = np.arange(2)
    confirm_axes[0].bar(bx - width / 2, v6_auc, width, color=blue, label="Invariant v6")
    confirm_axes[0].bar(bx + width / 2, v14_auc, width, color=teal, label="V14 ensemble")
    confirm_axes[0].set_xticks(bx, benchmark_labels)
    confirm_axes[0].set_ylim(0.50, 0.75)
    confirm_axes[0].set_ylabel("Mean ROC-AUC")
    confirm_axes[0].set_title("A. Transfer depends on the benchmark", loc="left", weight="bold")
    confirm_axes[0].legend(frameon=False, fontsize=8)
    for bars in confirm_axes[0].containers:
        confirm_axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    cy = np.arange(len(confirm_domains))
    v6_domain_auc = [v6_confirm["generators"][name]["auc_roc"] for name in confirm_domains]
    v14_domain_auc = [v14_confirm["generators"][name]["auc_roc"] for name in confirm_domains]
    confirm_axes[1].hlines(cy, v6_domain_auc, v14_domain_auc, color=gray, linewidth=2)
    confirm_axes[1].scatter(v6_domain_auc, cy, color=blue, s=32, label="Invariant v6")
    confirm_axes[1].scatter(v14_domain_auc, cy, color=teal, s=32, label="V14 ensemble")
    confirm_axes[1].set_yticks(cy, [display_domains[name] for name in confirm_domains])
    confirm_axes[1].invert_yaxis()
    confirm_axes[1].set_xlim(0.50, 0.80)
    confirm_axes[1].set_xlabel("ROC-AUC")
    confirm_axes[1].set_title("B. Frozen GenImage confirmation", loc="left", weight="bold")
    confirm_axes[1].axvline(0.5, color="#666666", linewidth=0.8, linestyle="--")
    confirm_axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    confirm_fig.suptitle("Cross-dataset confirmation reveals the generalization limit", fontsize=13, weight="bold")
    confirm_fig.savefig(OUTPUT / "cross_dataset_confirmation.png", bbox_inches="tight")
    confirm_fig.savefig(OUTPUT / "cross_dataset_confirmation.pdf", bbox_inches="tight")
    print(OUTPUT / "regional_multi_physics_results.png")
    print(OUTPUT / "cross_dataset_confirmation.png")


if __name__ == "__main__":
    main()
