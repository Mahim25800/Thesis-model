"""Train Disagreement-Exposed Fusion Gate for Dual-Stream Deepfake Detection.

Addresses the Root Cause:
In training corpora, DINOv2 is 99.4% accurate, creating a 97.7% conditional bias
where semantic is almost always correct whenever streams disagree. Consequently,
ERM forces the gate to trust semantic predictions almost unconditionally (alpha ~ 0.7-0.8),
which catastrophically fails on in-the-wild benchmarks (Chameleon) where semantic false alarms
are rampant (5,217 samples) but physics signals are authentic.

Solution: Genuinely Disagreement-Exposed Training
1. Injects synthetic semantic false alarms and epistemic noise into 35% of training samples
   while preserving true labels and clean regional physics signals.
2. Directly supervises dynamic trust allocation (alpha) using relative stream competence.
3. Conditions cross-modal evidential temperature scaling T(x) on spatial and confidence discrepancies.
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
    """Expected Calibration Error (ECE) with equal-width binning."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
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


def apply_disagreement_augmentation(
    dinov2_cls: torch.Tensor,
    dinov2_regional: torch.Tensor,
    labels: torch.Tensor,
    p_disagree: float = 0.35,
    swap_ratio: float = 0.50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies synthetic semantic disagreement augmentations to simulate OOD failure modes.
    
    1. Semantic Swapping: Replaces DINOv2 tokens with donor from opposite class (false alarm / miss).
    2. Semantic Noise/Dropout: Injects high Gaussian noise and feature dropout (epistemic uncertainty).
    
    Returns:
        aug_cls: [B, 768]
        aug_reg: [B, 5, 768]
        is_augmented: [B] boolean mask
    """
    batch_size = dinov2_cls.shape[0]
    device = dinov2_cls.device

    aug_cls = dinov2_cls.clone()
    aug_reg = dinov2_regional.clone()
    is_augmented = torch.zeros(batch_size, dtype=torch.bool, device=device)

    mask_aug = (torch.rand(batch_size, device=device) < p_disagree)
    if not mask_aug.any():
        return aug_cls, aug_reg, is_augmented

    aug_indices = torch.where(mask_aug)[0]
    is_augmented[aug_indices] = True

    swap_mask = (torch.rand(len(aug_indices), device=device) < swap_ratio)
    swap_indices = aug_indices[swap_mask]
    noise_indices = aug_indices[~swap_mask]

    # 1. Semantic Swapping
    if len(swap_indices) > 0:
        for idx in swap_indices:
            orig_lbl = labels[idx].item()
            opp_candidates = torch.where(labels != orig_lbl)[0]
            if len(opp_candidates) > 0:
                donor_idx = opp_candidates[torch.randint(len(opp_candidates), (1,)).item()]
                alpha_mix = torch.rand(1, device=device).item() * 0.25 + 0.75  # 0.75 - 1.0 donor
                aug_cls[idx] = alpha_mix * dinov2_cls[donor_idx] + (1.0 - alpha_mix) * dinov2_cls[idx]
                aug_reg[idx] = alpha_mix * dinov2_regional[donor_idx] + (1.0 - alpha_mix) * dinov2_regional[idx]
            else:
                noise_indices = torch.cat([noise_indices, idx.unsqueeze(0)])

    # 2. Semantic Feature Noise & Dropout
    if len(noise_indices) > 0:
        noise_cls = torch.randn_like(aug_cls[noise_indices]) * 1.5
        noise_reg = torch.randn_like(aug_reg[noise_indices]) * 1.5
        drop_mask_cls = (torch.rand_like(aug_cls[noise_indices]) > 0.4).float()
        drop_mask_reg = (torch.rand_like(aug_reg[noise_indices]) > 0.4).float()

        aug_cls[noise_indices] = (aug_cls[noise_indices] + noise_cls) * drop_mask_cls
        aug_reg[noise_indices] = (aug_reg[noise_indices] + noise_reg) * drop_mask_reg

    return aug_cls, aug_reg, is_augmented


def disagreement_exposed_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    gate_weight: float = 0.35,
    overconf_weight: float = 0.25,
    prior_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Loss explicitly supervising trust allocation under stream disagreement."""
    labels = labels.float().reshape(-1)
    p_final = outputs["prob_final"]
    logits_final = outputs["logits"]
    alpha = outputs["alpha"]
    T_x = outputs["temperature"]
    p_sem_grad = outputs["prob_semantic"]
    p_sem_detached = p_sem_grad.detach()
    p_phys = outputs["prob_physics"].detach()
    p_joint = torch.sigmoid(outputs["joint_logits"]).detach()

    # 1. Primary Classification Loss
    loss_bce_final = F.binary_cross_entropy_with_logits(logits_final, labels)

    # 2. Determine relative competence
    err_sem = torch.abs(p_sem_detached - labels)
    err_phys = torch.abs(p_phys - labels)
    err_joint = torch.abs(p_joint - labels)
    err_best_phys = torch.minimum(err_phys, err_joint)

    sem_correct = (err_sem < 0.50).float()
    phys_correct = (err_best_phys < 0.50).float()

    # Directional Target Alpha:
    alpha_target = torch.full_like(p_sem_detached, 0.50)
    # Direction 1: Both correct -> Semantic-dominant blend (0.72)
    both_correct = (sem_correct == 1.0) & (phys_correct == 1.0)
    alpha_target[both_correct] = 0.72
    # Direction 2: Semantic correct, Physics wrong -> Strong Semantic trust (0.88)
    sem_only = (sem_correct == 1.0) & (phys_correct == 0.0)
    alpha_target[sem_only] = 0.88
    # Direction 3: Physics correct, Semantic wrong (FA / Misses) -> Strong Physics trust (0.12)
    phys_only = (sem_correct == 0.0) & (phys_correct == 1.0)
    alpha_target[phys_only] = 0.12
    # Direction 4: Both wrong -> Fallback blend (0.40)
    both_wrong = (sem_correct == 0.0) & (phys_correct == 0.0)
    alpha_target[both_wrong] = 0.40

    pred_alpha = torch.clamp(alpha, 1e-6, 1.0 - 1e-6)
    loss_gate = F.binary_cross_entropy(pred_alpha, alpha_target)

    # 3. Discrepancy-conditioned overconfidence penalty
    labels_bool = (labels >= 0.5)
    sem_wrong = ((p_sem_detached >= 0.5) != labels_bool).float()
    stream_disc = torch.abs(p_sem_detached - p_phys)
    sem_conf_grad = 2.0 * torch.abs(p_sem_grad - 0.5)
    loss_overconf = torch.mean(sem_wrong * stream_disc * sem_conf_grad.pow(2))

    # 4. Temperature prior regularization (keeps T near 1.0 when not needed)
    loss_t_prior = torch.mean((T_x - 1.0).pow(2))

    total_loss = (
        loss_bce_final +
        gate_weight * loss_gate +
        overconf_weight * loss_overconf +
        prior_weight * loss_t_prior
    )

    return {
        "total_loss": total_loss,
        "loss_bce_final": loss_bce_final,
        "loss_gate": loss_gate,
        "loss_overconf": loss_overconf,
        "loss_t_prior": loss_t_prior,
        "mean_alpha": alpha.mean(),
        "mean_temp": T_x.mean(),
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
    parser = argparse.ArgumentParser(description="Train Disagreement-Exposed Gate on Universal Corpus")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate for gate & calib")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--p_disagree", type=float, default=0.35, help="Probability of synthetic disagreement")
    parser.add_argument("--gate_mode", type=str, default="v4_disagreement", help="Gate mode: v4_disagreement or v3_calibrated")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_checkpoint", type=str, default="models/universal_v5_calibrated/best_model.pt")
    parser.add_argument("--out_dir", type=str, default="models/universal_v5_disagreement_gate")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("DISAGREEMENT-EXPOSED FUSION GATE TRAINING")
    print(f"Gate Mode: {args.gate_mode} | Disagreement Rate: {args.p_disagree*100:.1f}%")
    print(f"Device: {device} | Base Checkpoint: {args.base_checkpoint}")
    print("=" * 80)

    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    standardizer = ckpt["standardizer"]

    # 1. Initialize DualStreamHybridDetector
    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode=args.gate_mode,
        dropout=0.15,
    ).to(device)

    # Load weights
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Base weights loaded successfully! Missing: {missing}, Unexpected: {unexpected}")

    # Freeze backbone streams: only train calib_net, gate networks, and joint scale
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

    # Trainable parameters (wrapped in list to prevent generator exhaustion)
    trainable_params = [
        {"params": list(model.fusion_head.calib_net.parameters()), "lr": args.lr},
        {"params": list(model.fusion_head.gate_sem_proj.parameters()), "lr": args.lr},
        {"params": list(model.fusion_head.gate_phys_proj.parameters()), "lr": args.lr},
        {"params": list(model.fusion_head.gate_net.parameters()), "lr": args.lr},
        {"params": [model.fusion_head.joint_scale], "lr": args.lr * 0.5},
    ]

    total_trainable = sum(p.numel() for group in trainable_params for p in group["params"] if p.requires_grad)
    print(f"Trainable Gate & Calibration Parameters: {total_trainable:,}")

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

    print(f"Dataset split: {len(train_idx):,} train samples | {len(val_idx):,} in-domain val samples")

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

    # Baseline Zero-Epoch Validation
    print("\n[Zero-Epoch Validation: Baseline]")
    init_metrics = evaluate(model, val_loader, standardizer, device)
    print(f"  Init Val Acc: {init_metrics['accuracy']*100:.2f}% | AUC: {init_metrics['auc']:.4f} | ECE: {init_metrics['ece']:.4f} | Alpha: {init_metrics['mean_alpha']:.3f} | T: {init_metrics['mean_temp']:.3f}")

    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_score = -1.0
    best_metrics = None
    history = []

    print(f"\nBeginning {args.epochs} Epochs of Disagreement-Exposed Gate Training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses, epoch_alphas, epoch_temps = [], [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            features = apply_standardizer(batch["physics_features"], standardizer).to(device)
            confidences = batch["physics_confidences"].to(device)
            dinov2_cls = batch["dinov2_cls"].to(device)
            dinov2_reg = batch["dinov2_regional"].to(device)
            labels = batch["label"].to(device)

            # Apply Synthetic Disagreement Augmentation
            aug_cls, aug_reg, is_augmented = apply_disagreement_augmentation(
                dinov2_cls=dinov2_cls,
                dinov2_regional=dinov2_reg,
                labels=labels,
                p_disagree=args.p_disagree,
            )

            out = model(
                physics_features=features,
                physics_confidences=confidences,
                dinov2_cls=aug_cls,
                dinov2_regional=aug_reg,
            )

            loss_dict = disagreement_exposed_loss(out, labels)
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss_dict["total_loss"].item())
            epoch_alphas.append(loss_dict["mean_alpha"].item())
            epoch_temps.append(loss_dict["mean_temp"].item())

        scheduler.step()

        # In-Domain Validation (evaluated on clean features)
        val_metrics = evaluate(model, val_loader, standardizer, device)
        train_loss = float(np.mean(epoch_losses))
        train_alpha = float(np.mean(epoch_alphas))
        train_temp = float(np.mean(epoch_temps))

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {train_loss:.4f} | Alpha_tr: {train_alpha:.3f} | T_tr: {train_temp:.3f} | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% | Val AUC: {val_metrics['auc']:.4f} | "
            f"Val ECE: {val_metrics['ece']:.4f} | Alpha_val: {val_metrics['mean_alpha']:.3f} | T_val: {val_metrics['mean_temp']:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_alpha": train_alpha,
            "train_temp": train_temp,
            "val_metrics": val_metrics,
        })

        # Save model based on validation score
        val_score = val_metrics["auc"] - val_metrics["ece"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_metrics = val_metrics
            best_ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "standardizer": standardizer,
                "metrics": val_metrics,
                "gate_mode": args.gate_mode,
            }, best_ckpt_path)

    # Save final model
    final_ckpt_path = out_dir / "final_model.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "standardizer": standardizer,
        "metrics": val_metrics,
        "gate_mode": args.gate_mode,
    }, final_ckpt_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Training Complete in {elapsed:.1f}s ({elapsed/60:.2f} min)!")
    print(f"Best Model Saved to: {best_ckpt_path}")
    print(f"Best Clean Val Metrics: Acc={best_metrics['accuracy']*100:.2f}%, AUC={best_metrics['auc']:.4f}, ECE={best_metrics['ece']:.4f}")
    print("=" * 80)

    # Save training log
    report_path = ROOT_DIR / "reports" / "disagreement_gate_training_log.json"
    with open(report_path, "w") as f:
        json.dump({
            "epochs": args.epochs,
            "gate_mode": args.gate_mode,
            "p_disagree": args.p_disagree,
            "best_metrics": best_metrics,
            "history": history,
            "checkpoint": str(best_ckpt_path.relative_to(ROOT_DIR)),
        }, f, indent=2)


if __name__ == "__main__":
    main()
