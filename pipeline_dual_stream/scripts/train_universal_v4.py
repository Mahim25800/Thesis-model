"""Training script for Universal Dual-Stream Deepfake Detector (dual_stream_v4).

CRITICAL METHODOLOGICAL UPGRADES FROM V3:
1. LEAKAGE-FREE CORPUS:
   Trained on universal_v4 corpus (zero exposure to Glide, SDv4, SDv5).
2. BALANCED PORTRAIT INTEGRATION:
   Trained with 1,600 authentic CelebA photographic portraits to neutralize the
   shortcut correlation between human facial features and synthetic classes.
3. SEMANTIC STREAM DROPOUT & CORRUPTION IN TRAINING:
   To prevent the Oracle Gate from collapsing to alpha = 1.0 (relying exclusively on DINOv2),
   we apply stochastic semantic dropout (p = 0.25). When semantic features are masked,
   DINOv2 prediction fails while Physics remains accurate, generating thousands of training
   samples where the Gate is supervised with alpha* -> 0.0 (trust Physics!).
4. DYNAMIC MULTIMODAL SYNERGY LOSS:
   End-to-end optimization of cross-attention and gating parameters with synergy preservation.
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


class UniversalAugmentedDatasetV4(Dataset):
    """Dataset with online stochastic feature perturbations:
    1. Stochastic Blur Jitter (DINOv2 token noise)
    2. Optical Bokeh quadrant normal masking
    3. Semantic Stream Dropout (crucial to teach Oracle Gate to trust Physics out-of-distribution!)
    """

    def __init__(
        self,
        corpus_path: Path,
        indices: np.ndarray,
        is_train: bool = True,
        blur_aug_prob: float = 0.30,
        mask_aug_prob: float = 0.20,
        sem_dropout_prob: float = 0.25,
    ):
        data = torch.load(corpus_path, map_location="cpu", weights_only=True)
        idx_t = torch.as_tensor(indices, dtype=torch.long)
        self.physics_features = data["physics_features"][idx_t]
        self.physics_confidences = data["physics_confidences"][idx_t]
        self.dinov2_cls = data["dinov2_cls"][idx_t]
        self.dinov2_regional = data["dinov2_regional"][idx_t]
        self.labels = data["labels"][idx_t].float()
        self.is_train = is_train
        self.blur_aug_prob = blur_aug_prob
        self.mask_aug_prob = mask_aug_prob
        self.sem_dropout_prob = sem_dropout_prob

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        phys_f = self.physics_features[idx].clone()
        phys_c = self.physics_confidences[idx].clone()
        dino_cls = self.dinov2_cls[idx].clone()
        dino_reg = self.dinov2_regional[idx].clone()
        lbl = self.labels[idx]

        is_sem_dropped = False
        if self.is_train:
            # 1. Semantic Stream Dropout: randomly blind DINOv2 to force Gate & Fusion to utilize Physics
            if random.random() < self.sem_dropout_prob:
                is_sem_dropped = True
                dino_cls = torch.randn_like(dino_cls) * 0.1
                dino_reg = torch.randn_like(dino_reg) * 0.1

            # 2. Stochastic Blur Jitter: perturb DINOv2 tokens along empirical optical blur variance
            elif random.random() < self.blur_aug_prob:
                noise_scale = random.uniform(0.02, 0.05)
                cls_noise = torch.randn_like(dino_cls) * noise_scale
                reg_noise = torch.randn_like(dino_reg) * noise_scale
                dino_cls = dino_cls + cls_noise
                dino_reg = dino_reg + reg_noise

            # 3. Physics Masking: simulate bokeh / shallow depth of field where peripheral normals are smooth
            if random.random() < self.mask_aug_prob:
                quad_idx = random.randint(1, 4)
                phys_c[quad_idx] = phys_c[quad_idx] * 0.1
                phys_f[quad_idx, :5] = 0.0

        return {
            "physics_features": phys_f,
            "physics_confidences": phys_c,
            "dinov2_cls": dino_cls,
            "dinov2_regional": dino_reg,
            "label": lbl,
            "sem_dropped": torch.tensor(1.0 if is_sem_dropped else 0.0, dtype=torch.float32),
        }


def fit_standardizer(features: torch.Tensor) -> Dict[str, list]:
    feat_np = features.numpy()
    mean = feat_np.mean(axis=0)
    std = np.maximum(feat_np.std(axis=0), 1e-6)
    return {"mean": mean.tolist(), "scale": std.tolist()}


def apply_standardizer(features: torch.Tensor, standardizer: Dict[str, list]) -> torch.Tensor:
    mean = torch.tensor(standardizer["mean"], dtype=features.dtype, device=features.device)
    scale = torch.tensor(standardizer["scale"], dtype=features.dtype, device=features.device)
    return (features - mean.unsqueeze(0)) / scale.unsqueeze(0)


def compute_v4_hybrid_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    sem_dropped: torch.Tensor,
    sem_weight: float = 0.20,
    phys_weight: float = 0.20,
    joint_weight: float = 0.20,
    gate_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Robust dual-stream loss with oracle gate supervision that does NOT collapse to alpha=1.0."""
    labels = labels.float().reshape(-1)

    loss_final = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
    loss_sem = F.binary_cross_entropy_with_logits(outputs["sem_logits"], labels)
    loss_phys = F.binary_cross_entropy_with_logits(outputs["phys_logits"], labels)
    loss_joint = F.binary_cross_entropy_with_logits(outputs["joint_logits"], labels)

    # Oracle Target Calculation:
    p_sem = torch.sigmoid(outputs["sem_logits"]).detach()
    p_phys = torch.sigmoid(outputs["phys_logits"]).detach()
    labels_bool = (labels >= 0.5)

    sem_correct = ((p_sem >= 0.5) == labels_bool).float()
    phys_correct = ((p_phys >= 0.5) == labels_bool).float()
    err_sem = torch.abs(p_sem - labels)
    err_phys = torch.abs(p_phys - labels)

    alpha_target = torch.full_like(p_sem, 0.50)
    # When semantic is correct and physics is wrong -> trust semantic (alpha -> 0.90)
    alpha_target[(sem_correct == 1.0) & (phys_correct == 0.0)] = 0.90
    # When physics is correct and semantic is wrong (or dropped) -> trust physics (alpha -> 0.10)
    alpha_target[(sem_correct == 0.0) & (phys_correct == 1.0)] = 0.10
    alpha_target[sem_dropped == 1.0] = 0.10

    # When both are correct, balance proportionally
    both_correct = (sem_correct == 1.0) & (phys_correct == 1.0) & (sem_dropped == 0.0)
    margin_diff = torch.clamp(err_phys - err_sem, -0.3, 0.3)
    alpha_target[both_correct] = 0.50 + margin_diff[both_correct] * 0.5

    pred_alpha = torch.clamp(outputs["alpha"], 1e-6, 1.0 - 1e-6)
    loss_gate = F.binary_cross_entropy(pred_alpha, alpha_target)

    # Anti-saturation entropy penalty on alpha: prevents alpha from collapsing to constant 1.0
    mean_alpha = pred_alpha.mean()
    loss_balance = 0.10 * (mean_alpha - 0.55).pow(2)

    total_loss = (
        loss_final +
        sem_weight * loss_sem +
        phys_weight * loss_phys +
        joint_weight * loss_joint +
        gate_weight * loss_gate +
        loss_balance
    )

    return {
        "total_loss": total_loss,
        "loss_final": loss_final,
        "loss_sem": loss_sem,
        "loss_phys": loss_phys,
        "loss_joint": loss_joint,
        "loss_gate": loss_gate,
        "alpha_target_mean": alpha_target.mean(),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    standardizer: Dict[str, list],
    device: str,
) -> Dict[str, float]:
    model.eval()
    all_labels, all_final_probs, all_sem_probs, all_phys_probs, all_alphas = [], [], [], [], []

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

            all_labels.extend(labels)
            all_final_probs.extend(out["prob_final"].cpu().numpy())
            all_sem_probs.extend(out["prob_semantic"].cpu().numpy())
            all_phys_probs.extend(out["prob_physics"].cpu().numpy())
            all_alphas.extend(out["alpha"].cpu().numpy())

    y_true = np.array(all_labels)
    y_final = np.array(all_final_probs)
    y_sem = np.array(all_sem_probs)
    y_phys = np.array(all_phys_probs)

    auc_final = roc_auc_score(y_true, y_final)
    auc_sem = roc_auc_score(y_true, y_sem)
    auc_phys = roc_auc_score(y_true, y_phys)

    acc_final = accuracy_score(y_true, y_final >= 0.5)
    acc_sem = accuracy_score(y_true, y_sem >= 0.5)
    acc_phys = accuracy_score(y_true, y_phys >= 0.5)

    return {
        "auc_final": float(auc_final),
        "auc_semantic": float(auc_sem),
        "auc_physics": float(auc_phys),
        "acc_final": float(acc_final),
        "acc_semantic": float(acc_sem),
        "acc_physics": float(acc_phys),
        "mean_alpha": float(np.mean(all_alphas)),
        "synergy_gain": float(auc_final - max(auc_sem, auc_phys)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Universal Dual-Stream Deepfake Detector (v4)")
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/data/universal_v4/universal_train_corpus.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/models/universal_v4",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("Training Leakage-Free Universal Dual-Stream Detector (universal_v4)")
    print(f"Corpus:     {args.corpus_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device:     {args.device}")
    print("=" * 75)

    corpus = torch.load(args.corpus_path, map_location="cpu", weights_only=True)
    total_len = len(corpus["labels"])
    indices = np.arange(total_len)
    np.random.shuffle(indices)

    val_count = int(total_len * args.val_ratio)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    print(f"Dataset partitioned: {len(train_idx)} train samples, {len(val_idx)} validation samples.")

    # Standardizer
    train_phys_features = corpus["physics_features"][train_idx]
    standardizer = fit_standardizer(train_phys_features)
    with open(output_dir / "standardizer.json", "w", encoding="utf-8") as f:
        json.dump(standardizer, f, indent=2)
    print("Standardizer fitted on training partition and saved.")

    # Datasets
    train_dataset = UniversalAugmentedDatasetV4(Path(args.corpus_path), train_idx, is_train=True)
    val_dataset = UniversalAugmentedDatasetV4(Path(args.corpus_path), val_idx, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # Initialize model
    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode="v2",
        dropout=0.15,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_score = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nStarting Training with Semantic Stream Dropout & Multimodal Synergy Supervision...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_gate_losses = []

        progress = (epoch - 1) / max(1, args.epochs - 1)
        sem_weight = 0.20 * (1.0 - progress) + 0.05 * progress
        phys_weight = 0.25 * (1.0 - progress) + 0.10 * progress
        joint_weight = 0.20 * (1.0 - progress) + 0.15 * progress
        gate_weight = 0.15 * (1.0 - progress) + 0.35 * progress

        for batch in train_loader:
            labels = batch["label"].to(args.device)
            sem_dropped = batch["sem_dropped"].to(args.device)
            features = apply_standardizer(batch["physics_features"], standardizer).to(args.device)
            confidences = batch["physics_confidences"].to(args.device)
            dinov2_cls = batch["dinov2_cls"].to(args.device)
            dinov2_reg = batch["dinov2_regional"].to(args.device)

            optimizer.zero_grad()
            out = model(
                physics_features=features,
                physics_confidences=confidences,
                dinov2_cls=dinov2_cls,
                dinov2_regional=dinov2_reg,
            )

            loss_dict = compute_v4_hybrid_loss(
                out,
                labels,
                sem_dropped,
                sem_weight=sem_weight,
                phys_weight=phys_weight,
                joint_weight=joint_weight,
                gate_weight=gate_weight,
            )
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss_dict["total_loss"].item())
            epoch_gate_losses.append(loss_dict["loss_gate"].item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        gate_loss = float(np.mean(epoch_gate_losses))

        # Evaluate on validation
        val_metrics = evaluate(model, val_loader, standardizer, args.device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = train_loss
        val_metrics["gate_loss"] = gate_loss
        history.append(val_metrics)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_loss:.4f} (Gate: {gate_loss:.4f}) | "
            f"Hybrid AUC: {val_metrics['auc_final']:.4f} | "
            f"DINOv2 AUC: {val_metrics['auc_semantic']:.4f} | "
            f"Physics AUC: {val_metrics['auc_physics']:.4f} | "
            f"Synergy: {val_metrics['synergy_gain']:+.4f} | "
            f"Alpha: {val_metrics['mean_alpha']:.2f}",
            flush=True,
        )

        composite_score = val_metrics["auc_final"] + 0.3 * val_metrics["auc_physics"] + 0.2 * val_metrics["synergy_gain"]
        if composite_score > best_val_score:
            best_val_score = composite_score
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "standardizer": standardizer,
                    "config": vars(args),
                },
                output_dir / "best_model.pt",
            )
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Hybrid AUC: {val_metrics['auc_final']:.4f}, Physics: {val_metrics['auc_physics']:.4f}, Synergy: {val_metrics['synergy_gain']:+.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (Best Epoch: {best_epoch}).")
                break

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 75)
    print(f"Universal v4 Model Training Complete! Best Epoch: {best_epoch}")
    print(f"Saved to: {output_dir / 'best_model.pt'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
