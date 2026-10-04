"""Train Universal Dual-Stream Deepfake Detector with Cross-Modal Evidential Calibration (Tier 2).

Innovation:
Physics-Conditioned Evidential Temperature Scaling:
T(x) = 1.0 + Softplus(W_cal * [Delta_phys(x), obs_phys(x)] + b_cal)

Adapts DINOv2 foundation epistemic temperature dynamically conditioned on
orthogonal physical discrepancies (surface normals, spherical harmonics, shadow residuals).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.hybrid_detector import DualStreamHybridDetector


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) with equal-width binning (Guo et al., 2017)."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        prop = in_bin.mean()
        if prop > 0:
            acc = labels[in_bin].mean()
            conf = probs[in_bin].mean()
            ece += np.abs(acc - conf) * prop
    return float(ece)


class CachedDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        confidences: torch.Tensor,
        dinov2_cls: torch.Tensor,
        dinov2_regional: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.features = features
        self.confidences = confidences
        self.dinov2_cls = dinov2_cls
        self.dinov2_regional = dinov2_regional
        self.labels = labels.float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "physics_features": self.features[idx],
            "physics_confidences": self.confidences[idx],
            "dinov2_cls": self.dinov2_cls[idx],
            "dinov2_regional": self.dinov2_regional[idx],
            "label": self.labels[idx],
        }


def apply_standardizer(features: torch.Tensor, standardizer: Dict[str, list]) -> torch.Tensor:
    mean = torch.tensor(standardizer["mean"], dtype=features.dtype, device=features.device)
    scale = torch.tensor(standardizer["scale"], dtype=features.dtype, device=features.device)
    return (features - mean.unsqueeze(0)) / scale.unsqueeze(0)


def evidential_calibration_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    prior_weight: float = 0.05,
    overconf_weight: float = 0.20,
    brier_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Composite loss designed specifically for cross-modal evidential calibration."""
    labels = labels.float().reshape(-1)
    p_final = outputs["prob_final"]
    p_sem = outputs["prob_semantic"]
    p_phys = outputs["prob_physics"]
    logits_final = outputs["logits"]
    logits_sem = outputs["sem_logits"]
    T_x = outputs["temperature"]

    # 1. Primary Classification BCE
    loss_bce_final = F.binary_cross_entropy_with_logits(logits_final, labels)
    loss_bce_sem = F.binary_cross_entropy_with_logits(logits_sem, labels)

    # 2. Proper Scoring Rule: Brier Score on final probabilities
    loss_brier = F.mse_loss(p_final, labels)

    # 3. Discrepancy-conditioned overconfidence penalty:
    # When semantic stream is wrong (|p_sem - y| > 0.5) and physics disagrees,
    # penalize semantic confidence to drive temperature T up!
    labels_bool = (labels >= 0.5)
    sem_wrong = ((p_sem >= 0.5) != labels_bool).float()
    stream_disc = torch.abs(p_sem - p_phys)
    sem_conf = 2.0 * torch.abs(p_sem - 0.5)

    loss_overconf = torch.mean(sem_wrong * stream_disc * sem_conf.pow(2))

    # 4. Temperature prior regularization: pull T towards 1.0 when not needed
    loss_t_prior = torch.mean((T_x - 1.0).pow(2))

    total_loss = (
        loss_bce_final +
        0.30 * loss_bce_sem +
        brier_weight * loss_brier +
        overconf_weight * loss_overconf +
        prior_weight * loss_t_prior
    )

    return {
        "total_loss": total_loss,
        "loss_bce_final": loss_bce_final,
        "loss_brier": loss_brier,
        "loss_overconf": loss_overconf,
        "loss_t_prior": loss_t_prior,
        "mean_temperature": T_x.mean(),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    standardizer: Dict[str, list],
    device: str,
) -> Dict[str, float]:
    model.eval()
    all_labels, all_final_probs, all_sem_probs, all_phys_probs, all_temps, all_alphas = [], [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].numpy()
            features = apply_standardizer(batch["physics_features"], standardizer).to(device)
            confidences = batch["physics_confidences"].to(device)
            dinov2_cls = batch["dinov2_cls"].to(device)
            dinov2_reg = batch["dinov2_regional"].to(device)

            out = model(
                physics_features=features,
                physics_confidences=confidences,
                dinov2_cls=dinov2_cls,
                dinov2_regional=dinov2_reg,
            )

            all_labels.extend(labels.tolist())
            all_final_probs.extend(out["prob_final"].cpu().numpy().tolist())
            all_sem_probs.extend(out["prob_semantic"].cpu().numpy().tolist())
            all_phys_probs.extend(out["prob_physics"].cpu().numpy().tolist())
            all_temps.extend(out["temperature"].cpu().numpy().tolist())
            all_alphas.extend(out["alpha"].cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = (np.array(all_final_probs) >= 0.5).astype(int)
    probs = np.array(all_final_probs)

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, probs))
    ece = compute_ece(probs, y_true, n_bins=15)
    brier = float(np.mean((probs - y_true) ** 2))

    return {
        "accuracy": acc,
        "auc": auc,
        "ece": ece,
        "brier": brier,
        "mean_temp": float(np.mean(all_temps)),
        "mean_alpha": float(np.mean(all_alphas)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Tier 2 Evidential Calibration on Universal v4")
    parser.add_argument("--epochs", type=int, default=15, help="Calibration epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for calibration & gate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--v4_checkpoint", type=str, default="models/universal_v4/best_model.pt")
    parser.add_argument("--out_dir", type=str, default="models/universal_v5_calibrated")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("UNIVERSAL DUAL-STREAM V5: CROSS-MODAL EVIDENTIAL CALIBRATION (TIER 2)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Loading Base Universal v4 Checkpoint: {args.v4_checkpoint}")

    ckpt = torch.load(args.v4_checkpoint, map_location="cpu", weights_only=False)
    standardizer = ckpt["standardizer"]

    # 1. Initialize DualStreamHybridDetector with gate_mode="v3_calibrated"
    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode="v3_calibrated",
        dropout=0.15,
    ).to(device)

    # Load pretrained weights with strict=False (calib_net initialized fresh)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Base weights loaded successfully! Initialized fresh calibration weights: {missing}")

    # Freeze backbone streams: only train calib_net and fine-tune gate_net
    for param in model.semantic_stream.parameters():
        param.requires_grad = False
    for param in model.physics_stream.parameters():
        param.requires_grad = False
    for param in model.fusion_head.sem_proj.parameters():
        param.requires_grad = False
    for param in model.fusion_head.phys_proj.parameters():
        param.requires_grad = False
    for param in model.fusion_head.cross_attn.parameters():
        param.requires_grad = False
    for param in model.fusion_head.sem_head.parameters():
        param.requires_grad = False
    for param in model.fusion_head.joint_head.parameters():
        param.requires_grad = False

    # Train calib_net + gate network
    trainable_params = [
        {"params": model.fusion_head.calib_net.parameters(), "lr": args.lr},
        {"params": model.fusion_head.gate_sem_proj.parameters(), "lr": args.lr * 0.5},
        {"params": model.fusion_head.gate_phys_proj.parameters(), "lr": args.lr * 0.5},
        {"params": model.fusion_head.gate_net.parameters(), "lr": args.lr * 0.5},
    ]

    total_trainable = sum(p.numel() for group in trainable_params for p in group["params"] if p.requires_grad)
    print(f"Trainable calibration parameters: {total_trainable:,}")

    # 2. Load Universal Train Corpus
    corpus_path = ROOT_DIR / "data" / "universal_v4" / "universal_train_corpus.pt"
    print(f"\nLoading Training Corpus from {corpus_path}...")
    corpus = torch.load(corpus_path, map_location="cpu", weights_only=True)
    num_total = len(corpus["labels"])

    indices = np.arange(num_total)
    np.random.shuffle(indices)
    val_size = 5000
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    print(f"Dataset split: {len(train_idx):,} train samples | {len(val_idx):,} val samples")

    train_ds = CachedDataset(
        features=corpus["physics_features"][train_idx],
        confidences=corpus["physics_confidences"][train_idx],
        dinov2_cls=corpus["dinov2_cls"][train_idx],
        dinov2_regional=corpus["dinov2_regional"][train_idx],
        labels=corpus["labels"][train_idx],
    )
    val_ds = CachedDataset(
        features=corpus["physics_features"][val_idx],
        confidences=corpus["physics_confidences"][val_idx],
        dinov2_cls=corpus["dinov2_cls"][val_idx],
        dinov2_regional=corpus["dinov2_regional"][val_idx],
        labels=corpus["labels"][val_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device == "cuda"))

    # Initial Zero-Epoch Validation (Verifying baseline performance)
    print("\n[Zero-Epoch Validation: Pre-Calibration Baseline]")
    init_metrics = evaluate(model, val_loader, standardizer, device)
    print(f"  Init Accuracy: {init_metrics['accuracy']*100:.2f}% | AUC: {init_metrics['auc']:.4f} | ECE: {init_metrics['ece']:.4f} | T_mean: {init_metrics['mean_temp']:.3f}")

    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_score = -1.0
    best_metrics = None
    history = []

    print(f"\nBeginning {args.epochs} Epochs of Evidential Calibration Training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_temps = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            features = apply_standardizer(batch["physics_features"], standardizer).to(device)
            confidences = batch["physics_confidences"].to(device)
            dinov2_cls = batch["dinov2_cls"].to(device)
            dinov2_reg = batch["dinov2_regional"].to(device)
            labels = batch["label"].to(device)

            out = model(
                physics_features=features,
                physics_confidences=confidences,
                dinov2_cls=dinov2_cls,
                dinov2_regional=dinov2_reg,
            )

            loss_dict = evidential_calibration_loss(out, labels)
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss_dict["total_loss"].item())
            epoch_temps.append(loss_dict["mean_temperature"].item())

        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, standardizer, device)
        train_loss = float(np.mean(epoch_losses))
        train_temp = float(np.mean(epoch_temps))

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | T_train: {train_temp:.3f} | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% | Val AUC: {val_metrics['auc']:.4f} | "
            f"Val ECE: {val_metrics['ece']:.4f} | T_val: {val_metrics['mean_temp']:.3f} | alpha: {val_metrics['mean_alpha']:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_temp": train_temp,
            "val_metrics": val_metrics,
        })

        # Score balances high AUC and low ECE: Score = AUC - ECE
        calibrated_score = val_metrics["auc"] - val_metrics["ece"]
        if calibrated_score > best_val_score:
            best_val_score = calibrated_score
            best_metrics = val_metrics
            best_ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "standardizer": standardizer,
                "metrics": val_metrics,
                "gate_mode": "v3_calibrated",
            }, best_ckpt_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Calibration Training Complete in {elapsed:.1f}s ({elapsed/60:.2f} min)!")
    print(f"Best Model Saved to: {best_ckpt_path}")
    print(f"Best Val Metrics: Acc={best_metrics['accuracy']*100:.2f}%, AUC={best_metrics['auc']:.4f}, ECE={best_metrics['ece']:.4f}, Mean T={best_metrics['mean_temp']:.3f}")
    print("=" * 80)

    # Save training log
    report_path = ROOT_DIR / "reports" / "universal_v5_calibration_training_log.json"
    with open(report_path, "w") as f:
        json.dump({
            "epochs": args.epochs,
            "best_metrics": best_metrics,
            "history": history,
            "checkpoint": str(best_ckpt_path.relative_to(ROOT_DIR)),
        }, f, indent=2)


if __name__ == "__main__":
    main()
