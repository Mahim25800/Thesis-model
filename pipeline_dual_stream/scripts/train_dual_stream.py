"""Training script for Dual-Stream Hybrid Detector (DINOv2 + Regional Physics).
Trains Gated Cross-Attention Fusion head and Regional Physics stream using multi-task supervision.
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import DualStreamCachedDataset, create_group_disjoint_split
from src.models.hybrid_detector import DualStreamHybridDetector
from src.models.gated_fusion import dual_stream_hybrid_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_standardizer(features: torch.Tensor) -> Dict[str, list]:
    """Fit regional feature standardizer [5, 14] on training partition only."""
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
    """Compute AUC and accuracy for all streams (Hybrid, Semantic, Physics)."""
    model.eval()
    all_labels = []
    all_final_probs = []
    all_sem_probs = []
    all_phys_probs = []
    all_alphas = []

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
        "synergy_gain": float(auc_final - auc_sem),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Stream Hybrid Detector")
    parser.add_argument(
        "--physics-cache",
        type=str,
        default="G:/Thesis/pipeline_40k/data/cache_regional_dsine_v2/train_features.pt",
    )
    parser.add_argument(
        "--dinov2-cache",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/data/dinov2_cache_train.pt",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="G:/Thesis/pipeline_40k/data/cache_regional_dsine_v2/train_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/models/dual_stream_v1",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Dual-Stream Hybrid Deepfake Detector Training")
    print(f"Physics cache: {args.physics_cache}")
    print(f"DINOv2 cache:  {args.dinov2_cache}")
    print(f"Device:        {args.device}")
    print("=" * 70)

    # 1. Create group-disjoint splits
    print("Partitioning data with difference-hash group disjointness...")
    train_idx, val_idx = create_group_disjoint_split(args.manifest, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Split completed: {len(train_idx)} train samples, {len(val_idx)} validation samples.")

    # 2. Datasets
    full_dataset = DualStreamCachedDataset(
        physics_cache_path=args.physics_cache,
        dinov2_cache_path=args.dinov2_cache,
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # 3. Fit standardizer on train partition only
    train_phys_features = full_dataset.physics_features[train_idx]
    standardizer = fit_standardizer(train_phys_features)
    with open(output_dir / "standardizer.json", "w", encoding="utf-8") as f:
        json.dump(standardizer, f, indent=2)

    # 4. Initialize model
    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,  # using precomputed tokens during training
        phys_dim=64,
        proj_dim=128,
        num_heads=4,
        dropout=0.15,
    ).to(args.device)

    # Separate parameter groups
    optimizer = torch.optim.AdamW(
        [
            {"params": model.physics_stream.parameters(), "lr": args.lr * 0.5},
            {"params": model.fusion_head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Training Loop
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

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

            loss_dict = dual_stream_hybrid_loss(out, labels)
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss_dict["total_loss"].item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))

        # Evaluate on validation
        val_metrics = evaluate(model, val_loader, standardizer, args.device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = train_loss
        history.append(val_metrics)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_loss:.4f} | "
            f"Hybrid AUC: {val_metrics['auc_final']:.4f} | "
            f"DINOv2 AUC: {val_metrics['auc_semantic']:.4f} | "
            f"Physics AUC: {val_metrics['auc_physics']:.4f} | "
            f"Synergy: +{val_metrics['synergy_gain']:.4f} | "
            f"Alpha: {val_metrics['mean_alpha']:.2f}"
        )

        # Checkpoint if best
        if val_metrics["auc_final"] > best_val_auc:
            best_val_auc = val_metrics["auc_final"]
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
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (Best Epoch: {best_epoch}).")
                break

    # Save final history
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Hybrid Validation AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
