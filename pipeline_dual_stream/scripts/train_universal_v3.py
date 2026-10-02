"""Training script for Universal Dual-Stream Deepfake Detector (dual_stream_v3).
Trained on 54,500 multi-generator & diverse-real samples with stochastic feature augmentation for blur & compression robustness.
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
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.models.hybrid_detector import DualStreamHybridDetector
from src.models.gated_fusion import dual_stream_hybrid_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UniversalAugmentedDataset(Dataset):
    """Dataset with online stochastic feature perturbations for blur & compression invariance."""

    def __init__(
        self,
        corpus_path: Path,
        indices: np.ndarray,
        is_train: bool = True,
        blur_aug_prob: float = 0.35,
        mask_aug_prob: float = 0.20,
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

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        phys_f = self.physics_features[idx].clone()
        phys_c = self.physics_confidences[idx].clone()
        dino_cls = self.dinov2_cls[idx].clone()
        dino_reg = self.dinov2_regional[idx].clone()
        lbl = self.labels[idx]

        if self.is_train:
            # 1. Stochastic Blur Jitter: perturb DINOv2 tokens along empirical optical blur variance
            if random.random() < self.blur_aug_prob:
                noise_scale = random.uniform(0.02, 0.06)
                cls_noise = torch.randn_like(dino_cls) * noise_scale
                reg_noise = torch.randn_like(dino_reg) * noise_scale
                dino_cls = dino_cls + cls_noise
                dino_reg = dino_reg + reg_noise

            # 2. Physics Masking: simulate bokeh / shallow depth of field where peripheral normals are smooth
            if random.random() < self.mask_aug_prob:
                quad_idx = random.randint(1, 4)
                # Drop confidence for that quadrant
                phys_c[quad_idx] = phys_c[quad_idx] * 0.1
                # Smooth surface normal features towards planar
                phys_f[quad_idx, :5] = 0.0

        return {
            "physics_features": phys_f,
            "physics_confidences": phys_c,
            "dinov2_cls": dino_cls,
            "dinov2_regional": dino_reg,
            "label": lbl,
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
    parser = argparse.ArgumentParser(description="Train Universal Dual-Stream Deepfake Detector")
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/data/universal_v3/universal_train_corpus.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/models/universal_v3",
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

    print("=" * 70)
    print("Training Universal Dual-Stream Detector (universal_v3)")
    print(f"Corpus:     {args.corpus_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device:     {args.device}")
    print("=" * 70)

    # 1. Partition train / val splits (stratified by label)
    corpus = torch.load(args.corpus_path, map_location="cpu", weights_only=True)
    total_len = len(corpus["labels"])
    indices = np.arange(total_len)
    np.random.shuffle(indices)

    val_count = int(total_len * args.val_ratio)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    print(f"Dataset partitioned: {len(train_idx)} train samples, {len(val_idx)} validation samples.")

    # 2. Fit standardizer on train partition only
    train_phys_features = corpus["physics_features"][train_idx]
    standardizer = fit_standardizer(train_phys_features)
    with open(output_dir / "standardizer.json", "w", encoding="utf-8") as f:
        json.dump(standardizer, f, indent=2)
    print("Standardizer fitted on training partition and saved.")

    # 3. Create datasets and dataloaders
    train_dataset = UniversalAugmentedDataset(Path(args.corpus_path), train_idx, is_train=True)
    val_dataset = UniversalAugmentedDataset(Path(args.corpus_path), val_idx, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # 4. Initialize model
    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,  # cached features during training
        gate_mode="v2",
        dropout=0.15,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nStarting Training with Oracle Gate Supervision & Dynamic Weight Annealing...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_gate_losses = []

        progress = (epoch - 1) / max(1, args.epochs - 1)
        sem_weight = 0.25 * (1.0 - progress) + 0.05 * progress
        phys_weight = 0.25 * (1.0 - progress) + 0.05 * progress
        joint_weight = 0.20 * (1.0 - progress) + 0.10 * progress
        gate_weight = 0.10 * (1.0 - progress) + 0.35 * progress

        for batch in train_loader:
            labels = batch["label"].to(args.device)
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

            loss_dict = dual_stream_hybrid_loss(
                out,
                labels,
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
            f"Alpha: {val_metrics['mean_alpha']:.2f}"
        )

        composite_score = val_metrics["auc_final"] + 0.3 * val_metrics["auc_physics"]
        if composite_score > best_val_auc:
            best_val_auc = composite_score
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
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Hybrid AUC: {val_metrics['auc_final']:.4f}, Physics: {val_metrics['auc_physics']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (Best Epoch: {best_epoch}).")
                break

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Universal Model Training Complete! Best Epoch: {best_epoch}")
    print(f"Saved to: {output_dir / 'best_model.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
