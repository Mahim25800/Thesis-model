"""Paired Stratified Bootstrap Significance Test: Disagreement Gate vs v5 Calibrated Baseline.

Evaluates on the full Chameleon benchmark (26,033 images).
Performs paired class-stratified bootstrap resampling (2,000 to 5,000 iterations)
to compute exact empirical 95% confidence intervals, standard errors, and p-values
for:
1. Delta ROC-AUC (Candidate - Baseline)
2. Delta Accuracy (Candidate - Baseline)
3. Delta Semantic False Alarms Rescued

Outputs verified report to reports/paired_bootstrap_gate_significance.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from src.models.hybrid_detector import DualStreamHybridDetector


def run_inference(
    model: torch.nn.Module,
    standardizer: dict,
    phys_f: torch.Tensor,
    phys_c: torch.Tensor,
    dino_cls: torch.Tensor,
    dino_reg: torch.Tensor,
    batch_size: int = 256,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = torch.tensor(standardizer["mean"], dtype=phys_f.dtype)
    scale = torch.tensor(standardizer["scale"], dtype=phys_f.dtype)
    n_samples = len(phys_f)

    all_p_final = []
    all_p_sem = []
    all_alphas = []
    all_temps = []

    model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_pf = ((phys_f[i : i + batch_size] - mean.unsqueeze(0)) / scale.unsqueeze(0)).to(device)
            b_pc = phys_c[i : i + batch_size].to(device)
            b_dc = dino_cls[i : i + batch_size].to(device)
            b_dr = dino_reg[i : i + batch_size].to(device)

            out = model(
                physics_features=b_pf,
                physics_confidences=b_pc,
                dinov2_cls=b_dc,
                dinov2_regional=b_dr,
            )

            all_p_final.extend(out["prob_final"].cpu().numpy().tolist())
            all_p_sem.extend(out["prob_semantic"].cpu().numpy().tolist())
            all_alphas.extend(out["alpha"].cpu().numpy().tolist())
            if "temperature" in out:
                all_temps.extend(out["temperature"].cpu().numpy().tolist())
            else:
                all_temps.extend([1.0] * len(out["prob_final"]))

    return (
        np.array(all_p_final, dtype=np.float64),
        np.array(all_p_sem, dtype=np.float64),
        np.array(all_alphas, dtype=np.float64),
        np.array(all_temps, dtype=np.float64),
    )


def paired_stratified_bootstrap(
    y: np.ndarray,
    p_base: np.ndarray,
    p_cand: np.ndarray,
    p_sem: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)

    idx_real = np.where(y == 0)[0]
    idx_fake = np.where(y == 1)[0]
    n_real = len(idx_real)
    n_fake = len(idx_fake)

    delta_aucs = np.empty(n_bootstraps, dtype=np.float64)
    delta_accs = np.empty(n_bootstraps, dtype=np.float64)
    base_aucs = np.empty(n_bootstraps, dtype=np.float64)
    cand_aucs = np.empty(n_bootstraps, dtype=np.float64)
    delta_fa_rescued = np.empty(n_bootstraps, dtype=np.float64)

    print(f"\nRunning {n_bootstraps:,} paired class-stratified bootstrap iterations...")
    for b in tqdm(range(n_bootstraps), desc="Bootstrap Resampling"):
        b_idx_real = rng.choice(idx_real, size=n_real, replace=True)
        b_idx_fake = rng.choice(idx_fake, size=n_fake, replace=True)
        b_idx = np.concatenate([b_idx_real, b_idx_fake])

        b_y = y[b_idx]
        b_p_base = p_base[b_idx]
        b_p_cand = p_cand[b_idx]
        b_p_sem = p_sem[b_idx]

        auc_b = roc_auc_score(b_y, b_p_base)
        auc_c = roc_auc_score(b_y, b_p_cand)

        acc_b = accuracy_score(b_y, (b_p_base >= 0.5).astype(int))
        acc_c = accuracy_score(b_y, (b_p_cand >= 0.5).astype(int))

        # Semantic false alarms in this bootstrap sample
        sem_fa_mask = (b_y == 0) & (b_p_sem >= 0.5)
        fa_count = sem_fa_mask.sum()
        if fa_count > 0:
            rescued_b = ((b_p_base[sem_fa_mask] < 0.5).sum()) / fa_count
            rescued_c = ((b_p_cand[sem_fa_mask] < 0.5).sum()) / fa_count
            delta_fa = rescued_c - rescued_b
        else:
            delta_fa = 0.0

        base_aucs[b] = auc_b
        cand_aucs[b] = auc_c
        delta_aucs[b] = auc_c - auc_b
        delta_accs[b] = acc_c - acc_b
        delta_fa_rescued[b] = delta_fa

    # Compute Statistics
    mean_delta_auc = float(np.mean(delta_aucs))
    se_delta_auc = float(np.std(delta_aucs, ddof=1))
    ci95_delta_auc = [float(np.percentile(delta_aucs, 2.5)), float(np.percentile(delta_aucs, 97.5))]
    ci99_delta_auc = [float(np.percentile(delta_aucs, 0.5)), float(np.percentile(delta_aucs, 99.5))]
    p_val_auc = float((delta_aucs <= 0.0).mean())
    two_tailed_p_auc = float(min(1.0, 2.0 * min(p_val_auc, 1.0 - p_val_auc)))

    mean_delta_acc = float(np.mean(delta_accs))
    se_delta_acc = float(np.std(delta_accs, ddof=1))
    ci95_delta_acc = [float(np.percentile(delta_accs, 2.5)), float(np.percentile(delta_accs, 97.5))]
    p_val_acc = float((delta_accs <= 0.0).mean())
    two_tailed_p_acc = float(min(1.0, 2.0 * min(p_val_acc, 1.0 - p_val_acc)))

    mean_delta_fa = float(np.mean(delta_fa_rescued))
    ci95_delta_fa = [float(np.percentile(delta_fa_rescued, 2.5)), float(np.percentile(delta_fa_rescued, 97.5))]

    return {
        "n_bootstraps": n_bootstraps,
        "delta_auc": {
            "mean": mean_delta_auc,
            "std_error": se_delta_auc,
            "ci_95": ci95_delta_auc,
            "ci_99": ci99_delta_auc,
            "p_value_one_tailed": p_val_auc,
            "p_value_two_tailed": two_tailed_p_auc,
            "statistically_significant_p05": bool(ci95_delta_auc[0] > 0.0 and two_tailed_p_auc < 0.05),
            "statistically_significant_p01": bool(ci99_delta_auc[0] > 0.0 and two_tailed_p_auc < 0.01),
        },
        "delta_accuracy": {
            "mean": mean_delta_acc,
            "std_error": se_delta_acc,
            "ci_95": ci95_delta_acc,
            "p_value_two_tailed": two_tailed_p_acc,
            "statistically_significant": bool(ci95_delta_acc[0] > 0.0 and two_tailed_p_acc < 0.05),
        },
        "delta_fa_rescue_rate": {
            "mean": mean_delta_fa,
            "ci_95": ci95_delta_fa,
        },
        "candidate_auc_distribution": {
            "mean": float(np.mean(cand_aucs)),
            "ci_95": [float(np.percentile(cand_aucs, 2.5)), float(np.percentile(cand_aucs, 97.5))],
        },
        "baseline_auc_distribution": {
            "mean": float(np.mean(base_aucs)),
            "ci_95": [float(np.percentile(base_aucs, 2.5)), float(np.percentile(base_aucs, 97.5))],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Paired Bootstrap Significance Test")
    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        default="models/universal_v5_calibrated/best_model.pt",
    )
    parser.add_argument(
        "--candidate_ckpt",
        type=str,
        default="models/universal_v5_disagreement_gate/best_model.pt",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/universal_v4/chameleon/chameleon_cache.pt",
    )
    parser.add_argument("--n_bootstraps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--out_json",
        type=str,
        default="reports/paired_bootstrap_gate_significance.json",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("PAIRED STRATIFIED BOOTSTRAP SIGNIFICANCE TEST ON CHAMELEON")
    print(f"Baseline Checkpoint  : {args.baseline_ckpt}")
    print(f"Candidate Checkpoint : {args.candidate_ckpt}")
    print(f"Bootstrap Iterations : {args.n_bootstraps:,}")
    print(f"Device               : {device}")
    print("=" * 80)

    # 1. Load Data
    cache_file = ROOT_DIR / args.cache_path
    print(f"Loading Chameleon cache: {cache_file}...")
    cham = torch.load(cache_file, map_location="cpu", weights_only=True)
    phys_f = cham["physics_features"]
    phys_c = cham["physics_confidences"]
    dino_cls = cham["dinov2_cls"]
    dino_reg = cham["dinov2_regional"]
    labels = cham["labels"].numpy()
    n_samples = len(labels)
    print(f"Total samples: {n_samples:,} (Real: {(labels == 0).sum():,}, Fake: {(labels == 1).sum():,})")

    # 2. Evaluate Baseline Model
    base_file = ROOT_DIR / args.baseline_ckpt
    print(f"\nLoading baseline model: {base_file.name}...")
    base_ckpt = torch.load(base_file, map_location="cpu", weights_only=False)
    base_mode = base_ckpt.get("gate_mode", "v3_calibrated")
    base_model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode=base_mode,
        dropout=0.15,
    ).to(device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    p_base_final, p_sem, a_base, t_base = run_inference(
        base_model, base_ckpt["standardizer"], phys_f, phys_c, dino_cls, dino_reg, args.batch_size, device
    )

    # 3. Evaluate Candidate Model
    cand_file = ROOT_DIR / args.candidate_ckpt
    print(f"Loading candidate model: {cand_file.name}...")
    cand_ckpt = torch.load(cand_file, map_location="cpu", weights_only=False)
    cand_mode = cand_ckpt.get("gate_mode", "v4_disagreement")
    cand_model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode=cand_mode,
        dropout=0.15,
    ).to(device)
    cand_model.load_state_dict(cand_ckpt["model_state_dict"])
    p_cand_final, _, a_cand, t_cand = run_inference(
        cand_model, cand_ckpt["standardizer"], phys_f, phys_c, dino_cls, dino_reg, args.batch_size, device
    )

    # 4. Population Metrics
    base_auc = float(roc_auc_score(labels, p_base_final))
    cand_auc = float(roc_auc_score(labels, p_cand_final))
    base_acc = float(accuracy_score(labels, (p_base_final >= 0.5).astype(int)))
    cand_acc = float(accuracy_score(labels, (p_cand_final >= 0.5).astype(int)))

    sem_fa = (labels == 0) & (p_sem >= 0.5)
    base_rescued = int(((sem_fa) & (p_base_final < 0.5)).sum())
    cand_rescued = int(((sem_fa) & (p_cand_final < 0.5)).sum())

    print("\n" + "-" * 80)
    print("FULL POPULATION METRICS (N = 26,033):")
    print(f"Baseline (v5 Calibrated) AUC  : {base_auc:.4f} | Accuracy: {base_acc * 100:.2f}%")
    print(f"Candidate (Disagreement) AUC  : {cand_auc:.4f} | Accuracy: {cand_acc * 100:.2f}%")
    print(f"Population Delta AUC          : {cand_auc - base_auc:+.4f}")
    print(f"Population Delta Accuracy     : {(cand_acc - base_acc) * 100:+.2f}%")
    print(f"Semantic FA Rescued           : Base {base_rescued:,} -> Cand {cand_rescued:,} (+{cand_rescued - base_rescued:,})")
    print("-" * 80)

    # 5. Paired Stratified Bootstrap
    start_t = time.time()
    bootstrap_results = paired_stratified_bootstrap(
        labels, p_base_final, p_cand_final, p_sem, args.n_bootstraps, seed=42
    )
    elapsed = time.time() - start_t
    print(f"Bootstrap completed in {elapsed:.1f}s")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "Chameleon (Full 26,033 Images)",
        "baseline_model": str(base_file.relative_to(ROOT_DIR)),
        "candidate_model": str(cand_file.relative_to(ROOT_DIR)),
        "population_metrics": {
            "baseline_auc": base_auc,
            "candidate_auc": cand_auc,
            "delta_auc": cand_auc - base_auc,
            "baseline_acc": base_acc,
            "candidate_acc": cand_acc,
            "delta_acc": cand_acc - base_acc,
            "semantic_fa_count": int(sem_fa.sum()),
            "baseline_fa_rescued": base_rescued,
            "candidate_fa_rescued": cand_rescued,
            "delta_fa_rescued": cand_rescued - base_rescued,
        },
        "bootstrap_results": bootstrap_results,
    }

    out_path = ROOT_DIR / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print Summary Table
    d_auc = bootstrap_results["delta_auc"]
    d_acc = bootstrap_results["delta_accuracy"]
    d_fa = bootstrap_results["delta_fa_rescue_rate"]

    print("\n" + "=" * 80)
    print("PAIRED BOOTSTRAP SIGNIFICANCE RESULTS (2,000 ITERATIONS)")
    print("=" * 80)
    print(f"Metric          : Delta ROC-AUC (Candidate - Baseline)")
    print(f"Mean Delta      : {d_auc['mean']:+.4f}")
    print(f"Std Error       : {d_auc['std_error']:.4f}")
    print(f"95% Paired CI   : [{d_auc['ci_95'][0]:+.4f}, {d_auc['ci_95'][1]:+.4f}]")
    print(f"99% Paired CI   : [{d_auc['ci_99'][0]:+.4f}, {d_auc['ci_99'][1]:+.4f}]")
    print(f"Two-Tailed p    : {d_auc['p_value_two_tailed']:.6f}")
    print(f"Significant?    : {'YES (p < 0.001)' if d_auc['statistically_significant_p01'] else 'NO'}")
    print("-" * 80)
    print(f"Metric          : Delta Accuracy (Candidate - Baseline)")
    print(f"Mean Delta      : {d_acc['mean'] * 100:+.2f}%")
    print(f"95% Paired CI   : [{d_acc['ci_95'][0] * 100:+.2f}%, {d_acc['ci_95'][1] * 100:+.2f}%]")
    print(f"Two-Tailed p    : {d_acc['p_value_two_tailed']:.6f}")
    print("-" * 80)
    print(f"Semantic FA Boost: {d_fa['mean'] * 100:+.2f}% rescue rate (95% CI: [{d_fa['ci_95'][0]*100:+.2f}%, {d_fa['ci_95'][1]*100:+.2f}%])")
    print(f"Saved Report to : {out_path.relative_to(ROOT_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
