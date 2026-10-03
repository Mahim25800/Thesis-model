"""Empirical Out-Of-Distribution (OOD) Distance Analysis in DINOv2 Embedding Space.

Evaluates Option 2:
1. Computes training set centroid and distribution geometry in DINOv2 768-d space.
2. Quantifies distance of held-out in-distribution sets vs. Chameleon (26,033 images).
3. Evaluates correlation between OOD distance and DINOv2 overconfidence/errors.
4. Simulates OOD-aware trust modulation on the gating network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent


def compute_distances(embeddings: torch.Tensor, centroid: torch.Tensor, cov_inv_diag: torch.Tensor = None) -> Dict[str, np.ndarray]:
    """Compute cosine, euclidean, and regularized diagonal Mahalanobis distances to a centroid."""
    # L2 normalized cosine distance
    emb_norm = F.normalize(embeddings, p=2, dim=-1)
    cen_norm = F.normalize(centroid.unsqueeze(0), p=2, dim=-1)
    cos_sim = (emb_norm * cen_norm).sum(dim=-1).clamp(-1.0, 1.0)
    cos_dist = (1.0 - cos_sim).cpu().numpy()

    # Euclidean distance
    diff = embeddings - centroid.unsqueeze(0)
    euc_dist = torch.norm(diff, p=2, dim=-1).cpu().numpy()

    # Diagonal Mahalanobis distance
    res = {
        "cos_dist": cos_dist,
        "euc_dist": euc_dist,
    }
    if cov_inv_diag is not None:
        maha_diag = torch.sqrt(torch.sum((diff ** 2) * cov_inv_diag.unsqueeze(0), dim=-1))
        res["maha_dist"] = maha_diag.cpu().numpy()

    return res


def get_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    print("=" * 80)
    print("OOD DISTANCE ANALYSIS: DINOV2 EMBEDDING SPACE (768-D)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")

    # 1. Load Universal Train Corpus
    train_path = ROOT_DIR / "data" / "universal_v4" / "universal_train_corpus.pt"
    print(f"\n[1/5] Loading Universal Training Corpus ({train_path.name})...")
    train_data = torch.load(train_path, map_location="cpu")
    train_cls = train_data["dinov2_cls"].to(device)  # [50100, 768]
    train_labels = train_data["labels"].numpy()
    num_train = train_cls.shape[0]
    print(f"  Training samples: {num_train:,} | Dim: {train_cls.shape[1]}")

    # Compute training centroids and diagonal covariance
    centroid_global = train_cls.mean(dim=0)
    centroid_real = train_cls[train_labels == 0].mean(dim=0)
    centroid_fake = train_cls[train_labels == 1].mean(dim=0)

    # Diagonal variance for Mahalanobis
    train_var = train_cls.var(dim=0) + 1e-5
    cov_inv_diag = 1.0 / train_var

    print("  Global & class centroids computed.")

    # Compute training baseline distances
    train_dists = compute_distances(train_cls, centroid_global, cov_inv_diag)
    train_stats = {k: get_stats(v) for k, v in train_dists.items()}

    print(f"  Train Baseline Cosine Distance: Mean = {train_stats['cos_dist']['mean']:.4f} "
          f"+/- {train_stats['cos_dist']['std']:.4f} (p95 = {train_stats['cos_dist']['p95']:.4f})")
    print(f"  Train Baseline Euclidean Distance: Mean = {train_stats['euc_dist']['mean']:.4f} "
          f"+/- {train_stats['euc_dist']['std']:.4f} (p95 = {train_stats['euc_dist']['p95']:.4f})")

    # 2. Load Evaluation Benchmarks
    print("\n[2/5] Loading Test Benchmarks...")
    benchmarks = {}

    # RAISE held out (Real)
    raise_path = ROOT_DIR / "data" / "universal_v4" / "raise_held_out_test.pt"
    if raise_path.exists():
        d_raise = torch.load(raise_path, map_location="cpu")
        benchmarks["RAISE_DSLR_Real"] = {
            "cls": d_raise["dinov2_cls"].to(device),
            "labels": d_raise["labels"].numpy(),
            "desc": "In-Distribution Real (DSLR Camera)",
        }

    # CelebA held out (Real)
    celeba_path = ROOT_DIR / "data" / "universal_v4" / "celeba_portraits_test.pt"
    if celeba_path.exists():
        d_celeba = torch.load(celeba_path, map_location="cpu")
        benchmarks["CelebA_Portraits_Real"] = {
            "cls": d_celeba["dinov2_cls"].to(device),
            "labels": d_celeba["labels"].numpy(),
            "desc": "In-Distribution Real (Portraits)",
        }

    # Glide held out (Unseen Fake)
    glide_path = ROOT_DIR / "data" / "dinov2_cache_glide.pt"
    if glide_path.exists():
        d_glide = torch.load(glide_path, map_location="cpu")
        benchmarks["OpenAI_Glide_Fake"] = {
            "cls": d_glide["dinov2_cls"].to(device),
            "labels": d_glide["labels"].numpy(),
            "desc": "Unseen Generator AI (Glide)",
        }

    # Chameleon (Full, Real, Fake)
    cham_path = ROOT_DIR / "data" / "universal_v4" / "chameleon" / "chameleon_cache.pt"
    print(f"  Loading Chameleon Cache ({cham_path.name})...")
    d_cham = torch.load(cham_path, map_location="cpu")
    cham_cls = d_cham["dinov2_cls"].to(device)
    cham_labels = np.array(d_cham["labels"].numpy() if hasattr(d_cham["labels"], "numpy") else d_cham["labels"])
    cham_filenames = d_cham["filenames"]

    benchmarks["Chameleon_All"] = {
        "cls": cham_cls,
        "labels": cham_labels,
        "desc": "Chameleon Benchmark (All 26,033)",
    }
    benchmarks["Chameleon_Real"] = {
        "cls": cham_cls[cham_labels == 0],
        "labels": cham_labels[cham_labels == 0],
        "desc": "Chameleon Real (14,863)",
    }
    benchmarks["Chameleon_Fake"] = {
        "cls": cham_cls[cham_labels == 1],
        "labels": cham_labels[cham_labels == 1],
        "desc": "Chameleon Fake (11,170)",
    }

    # 3. Compute Cross-Dataset Distance Profiles
    print("\n[3/5] Computing Cross-Dataset Distance Metrics...")
    results_datasets = {}

    print(f"\n{'Dataset':<26} {'N':<8} {'Cos Dist (Mean +/- Std)':<25} {'Euc Dist (Mean)':<16} {'KS Test p-val':<14} {'OOD Ratio vs Train':<16}")
    print("-" * 110)

    # Add train baseline to table
    train_cos_mean = train_stats["cos_dist"]["mean"]
    print(f"{'Universal Train Corpus':<26} {num_train:<8} {train_stats['cos_dist']['mean']:.4f} +/- {train_stats['cos_dist']['std']:.4f}        {train_stats['euc_dist']['mean']:.4f}         {'1.0000 (ref)':<14} {'1.000x':<16}")

    for name, data in benchmarks.items():
        cls_t = data["cls"]
        dists = compute_distances(cls_t, centroid_global, cov_inv_diag)
        st = {k: get_stats(v) for k, v in dists.items()}

        # KS test against training cosine distance distribution
        # Sample 5000 points from each for fast, stable computation
        s_train = np.random.choice(train_dists["cos_dist"], size=min(5000, len(train_dists["cos_dist"])), replace=False)
        s_test = np.random.choice(dists["cos_dist"], size=min(5000, len(dists["cos_dist"])), replace=False)
        ks_res = stats.ks_2samp(s_train, s_test)

        cos_mean = st["cos_dist"]["mean"]
        ratio = cos_mean / train_cos_mean

        results_datasets[name] = {
            "count": int(cls_t.shape[0]),
            "description": data["desc"],
            "stats": st,
            "ks_statistic": float(ks_res.statistic),
            "ks_pvalue": float(ks_res.pvalue),
            "ood_ratio_vs_train": float(ratio),
        }

        print(f"{name:<26} {cls_t.shape[0]:<8} {cos_mean:.4f} +/- {st['cos_dist']['std']:.4f}        {st['euc_dist']['mean']:.4f}         {ks_res.pvalue:<14.2e} {f'{ratio:.3f}x':<16}")

    # 4. Chameleon Error Analysis vs OOD Distance
    print("\n[4/5] Correlating OOD Distance with DINOv2 Overconfidence & Errors...")
    pred_path = ROOT_DIR / "reports" / "chameleon_per_image_predictions.parquet"
    df_preds = pd.read_parquet(pred_path)

    # Attach distances to chameleon predictions
    cham_dists = compute_distances(cham_cls, centroid_global, cov_inv_diag)
    df_preds["cos_dist"] = cham_dists["cos_dist"]
    df_preds["euc_dist"] = cham_dists["euc_dist"]
    df_preds["maha_dist"] = cham_dists["maha_dist"]

    # Distance to class centroids
    d_to_real = compute_distances(cham_cls, centroid_real)["cos_dist"]
    d_to_fake = compute_distances(cham_cls, centroid_fake)["cos_dist"]
    df_preds["dist_to_train_real"] = d_to_real
    df_preds["dist_to_train_fake"] = d_to_fake

    # Define prediction categories
    df_preds["pred_sem"] = (df_preds["prob_semantic"] >= 0.5).astype(int)
    df_preds["pred_phys"] = (df_preds["prob_physics"] >= 0.5).astype(int)
    df_preds["pred_final"] = (df_preds["prob_final"] >= 0.5).astype(int)

    # DINOv2 error types
    sem_correct = df_preds["pred_sem"] == df_preds["label"]
    sem_fp = (df_preds["pred_sem"] == 1) & (df_preds["label"] == 0)  # Real flagged Fake
    sem_fn = (df_preds["pred_sem"] == 0) & (df_preds["label"] == 1)  # Fake flagged Real

    # Extreme overconfidence subsets
    sem_fp_extreme = sem_fp & (df_preds["prob_semantic"] > 0.999)
    sem_fn_extreme = sem_fn & (df_preds["prob_semantic"] < 0.001)

    # Physics-rescued candidates
    fp_phys_rescues = sem_fp & (df_preds["pred_phys"] == 0)  # 1,953 cases
    fn_phys_rescues = sem_fn & (df_preds["pred_phys"] == 1)  # 1,446 cases

    error_dist_stats = {
        "DINOv2_Correct": get_stats(df_preds.loc[sem_correct, "cos_dist"].values),
        "DINOv2_False_Positives": get_stats(df_preds.loc[sem_fp, "cos_dist"].values),
        "DINOv2_False_Positives_Extreme (>0.999)": get_stats(df_preds.loc[sem_fp_extreme, "cos_dist"].values),
        "DINOv2_False_Negatives": get_stats(df_preds.loc[sem_fn, "cos_dist"].values),
        "DINOv2_False_Negatives_Extreme (<0.001)": get_stats(df_preds.loc[sem_fn_extreme, "cos_dist"].values),
        "False_Positives_Physics_Correct": get_stats(df_preds.loc[fp_phys_rescues, "cos_dist"].values),
        "False_Negatives_Physics_Correct": get_stats(df_preds.loc[fn_phys_rescues, "cos_dist"].values),
    }

    print(f"\n{'Category':<42} {'Count':<8} {'Cos Dist (Mean +/- Std)':<26} {'Median (p50)':<14} {'p90':<10}")
    print("-" * 105)
    counts = {
        "DINOv2_Correct": sem_correct.sum(),
        "DINOv2_False_Positives": sem_fp.sum(),
        "DINOv2_False_Positives_Extreme (>0.999)": sem_fp_extreme.sum(),
        "DINOv2_False_Negatives": sem_fn.sum(),
        "DINOv2_False_Negatives_Extreme (<0.001)": sem_fn_extreme.sum(),
        "False_Positives_Physics_Correct": fp_phys_rescues.sum(),
        "False_Negatives_Physics_Correct": fn_phys_rescues.sum(),
    }
    for cat, st in error_dist_stats.items():
        cnt = counts[cat]
        print(f"{cat:<42} {cnt:<8} {st['mean']:.4f} +/- {st['std']:.4f}         {st['median']:.4f}         {st['p90']:.4f}")

    # Stat tests: Is FP extreme distance significantly higher than correct?
    ttest_fp = stats.ttest_ind(df_preds.loc[sem_fp_extreme, "cos_dist"], df_preds.loc[sem_correct, "cos_dist"])
    print(f"\nT-Test (Extreme FP vs Correct Cosine Dist): t = {ttest_fp.statistic:.3f}, p = {ttest_fp.pvalue:.2e}")

    # 5. OOD-Aware Dynamic Gating Simulation
    print("\n[5/5] Simulating OOD-Aware Trust Modulation (Option 2)...")
    # Base model results
    y_true = df_preds["label"].values
    base_acc_hybrid = accuracy_score(y_true, df_preds["pred_final"])
    base_auc_hybrid = roc_auc_score(y_true, df_preds["prob_final"])
    base_acc_dino = accuracy_score(y_true, df_preds["pred_sem"])
    base_auc_dino = roc_auc_score(y_true, df_preds["prob_semantic"])

    print(f"  Baseline DINOv2 Alone:  Accuracy = {base_acc_dino * 100:.2f}% | AUC = {base_auc_dino:.4f}")
    print(f"  Baseline Hybrid v4:     Accuracy = {base_acc_hybrid * 100:.2f}% | AUC = {base_auc_hybrid:.4f}")

    # We simulate how gating adjusts when provided an OOD penalty:
    # If d_OOD > train_threshold (e.g. p75 or p90 of train), discount alpha towards physics
    # alpha_new = alpha * sigmoid(-gamma * (d_cos - threshold) / train_std)
    # Or linear blending:
    train_p90 = train_stats["cos_dist"]["p90"]
    train_std = train_stats["cos_dist"]["std"]

    # Reconstruct logits
    # p = sigmoid(z) -> z = log(p / (1 - p))
    eps = 1e-6
    p_sem = np.clip(df_preds["prob_semantic"].values, eps, 1.0 - eps)
    p_phys = np.clip(df_preds["prob_physics"].values, eps, 1.0 - eps)
    z_sem = np.log(p_sem / (1.0 - p_sem))
    z_phys = np.log(p_phys / (1.0 - p_phys))
    base_alpha = df_preds["alpha"].values

    simulation_results = []
    print(f"\n  Sweeping OOD attenuation factor gamma on Chameleon:")
    print(f"  {'Gamma (attenuation)':<22} {'Mean Alpha':<12} {'Accuracy (%)':<15} {'AUC':<12} {'Delta Acc vs Base':<18} {'Disagreement Acc (%)':<22}")
    print("-" * 105)

    disagree_mask = df_preds["pred_sem"] != df_preds["pred_phys"]

    for gamma in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        # Compute dynamic penalty based on distance exceeding training p90
        excess = np.maximum(0.0, (df_preds["cos_dist"].values - train_p90) / train_std)
        # Attenuation factor in [0, 1]
        attenuation = np.exp(-gamma * excess)
        adj_alpha = base_alpha * attenuation

        # Fused logits
        z_fused = adj_alpha * z_sem + (1.0 - adj_alpha) * z_phys
        p_fused = 1.0 / (1.0 + np.exp(-z_fused))
        pred_fused = (p_fused >= 0.5).astype(int)

        sim_acc = accuracy_score(y_true, pred_fused)
        sim_auc = roc_auc_score(y_true, p_fused)
        dis_acc = accuracy_score(y_true[disagree_mask], pred_fused[disagree_mask])
        delta_acc = (sim_acc - base_acc_hybrid) * 100

        print(f"  gamma = {gamma:<14.1f} {adj_alpha.mean():<12.3f} {sim_acc * 100:<15.2f} {sim_auc:<12.4f} {f'{delta_acc:+.2f}%':<18} {dis_acc * 100:<22.2f}")

        simulation_results.append({
            "gamma": gamma,
            "mean_alpha": float(adj_alpha.mean()),
            "accuracy": float(sim_acc),
            "auc": float(sim_auc),
            "delta_acc": float(delta_acc),
            "disagreement_accuracy": float(dis_acc),
        })

    # Save complete JSON analysis
    output_data = {
        "train_baseline": train_stats,
        "datasets": results_datasets,
        "chameleon_error_distance_correlations": error_dist_stats,
        "statistical_tests": {
            "extreme_fp_vs_correct_ttest": {
                "statistic": float(ttest_fp.statistic),
                "pvalue": float(ttest_fp.pvalue),
            }
        },
        "ood_gating_simulation": simulation_results,
    }

    out_file = ROOT_DIR / "reports" / "chameleon_ood_distance_analysis.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[OK] Analysis complete! Results saved to {out_file.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
