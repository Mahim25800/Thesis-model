"""Train Phase 2 Spatial Alignment and Anomaly Gated Fusion Network.

Key Enhancements in Phase 2:
1. Bidirectional Cross-Attention (Sem <-> Phys)
2. Multi-Instance Spatial Anomaly Pooling (MIL) for localized defect detection
3. Explicit Physical-Semantic Cosine Alignment Metric
4. Multi-Task Alignment & Calibration Loss
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = labels.shape[0]
    device = labels.device
    aug_cls = dinov2_cls.clone()
    aug_reg = dinov2_regional.clone()
    is_counterfactual = torch.zeros(B, dtype=torch.bool, device=device)

    for i in range(B):
        if torch.rand(1).item() < p_disagree:
            is_counterfactual[i] = True
            y = int(labels[i].item())
            if y == 0:
                # Real image: inject semantic false alarm
                noise = torch.randn_like(aug_cls[i]) * 0.45
                aug_cls[i] = aug_cls[i] + noise
                aug_reg[i] = aug_reg[i] + torch.randn_like(aug_reg[i]) * 0.45
            else:
                # Fake image: inject semantic false negative
                aug_cls[i] = aug_cls[i] * 0.4 + torch.randn_like(aug_cls[i]) * 0.15
                aug_reg[i] = aug_reg[i] * 0.4 + torch.randn_like(aug_reg[i]) * 0.15

    return aug_cls, aug_reg, is_counterfactual


def compute_phase2_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    is_counterfactual: torch.Tensor,
    bce: nn.BCEWithLogitsLoss,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    sem_logits = outputs["sem_logits"]
    phys_logits = outputs["phys_logits"]
    joint_logits = outputs["joint_logits"]
    alpha = outputs["alpha"]
    mean_align = outputs["mean_align"]

    # 1. Main classification loss
    loss_main = bce(logits, labels)
    loss_joint = bce(joint_logits, labels)

    # Clean samples stream loss
    clean_mask = ~is_counterfactual
    if clean_mask.sum() > 0:
        loss_sem = bce(sem_logits[clean_mask], labels[clean_mask])
    else:
        loss_sem = torch.tensor(0.0, device=labels.device)
    loss_phys = bce(phys_logits, labels)

    # 2. Gate Supervision (Competence-guided targets)
    with torch.no_grad():
        prob_sem = torch.sigmoid(sem_logits)
        prob_phys = torch.sigmoid(phys_logits)
        err_sem = torch.abs(prob_sem - labels)
        err_phys = torch.abs(prob_phys - labels)
        
        target_alpha = torch.where(
            err_sem < err_phys - 0.15,
            torch.full_like(alpha, 0.88),
            torch.where(
                err_phys < err_sem - 0.15,
                torch.full_like(alpha, 0.12),
                torch.full_like(alpha, 0.65)
            )
        )
    loss_gate = F.binary_cross_entropy(alpha, target_alpha)

    # 3. Alignment Loss:
    # Real photos should have high physical-semantic alignment (mean_align -> 1.0)
    # AI photos typically exhibit misaligned normal-texture correlations
    real_mask = (labels == 0) & clean_mask
    if real_mask.sum() > 0:
        loss_align = F.mse_loss(mean_align[real_mask], torch.ones_like(mean_align[real_mask]))
    else:
        loss_align = torch.tensor(0.0, device=labels.device)

    total_loss = loss_main + 0.6 * loss_joint + 0.3 * loss_sem + 0.2 * loss_phys + 0.4 * loss_gate + 0.15 * loss_align

    return total_loss, {
        "loss_main": float(loss_main.item()),
        "loss_joint": float(loss_joint.item()),
        "loss_gate": float(loss_gate.item()),
        "loss_align": float(loss_align.item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 Spatial Alignment Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--disagree_ratio", type=float, default=0.35)
    parser.add_argument("--val_split", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    print("=" * 80)
    print("PHASE 2: DENSE SPATIAL GEOMETRY & MULTIMODAL ALIGNMENT TRAINING")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("=" * 80)

    # 1. Load Precomputed Training Corpus
    corpus_path = ROOT_DIR / "data/universal_v4/universal_train_corpus.pt"
    print(f"Loading cached training corpus from {corpus_path.name}...")
    corpus = torch.load(corpus_path, map_location="cpu", weights_only=False)

    phys_f = corpus["physics_features"] # [50100, 5, 14]
    phys_c = corpus["physics_confidences"] # [50100, 5, 4]
    dinov2_cls = corpus["dinov2_cls"] # [50100, 768]
    dinov2_reg = corpus["dinov2_regional"] # [50100, 5, 768]
    labels = corpus["labels"] # [50100]

    # Standardize physics features
    mean = phys_f.mean(dim=(0, 1), keepdim=True)
    std = phys_f.std(dim=(0, 1), keepdim=True).clamp(min=1e-5)
    phys_f_norm = (phys_f - mean) / std

    standardizer = {
        "mean": mean.squeeze(0).squeeze(0).tolist(),
        "scale": std.squeeze(0).squeeze(0).tolist(),
    }

    # Split Train/Val
    N = len(labels)
    indices = list(range(N))
    random.shuffle(indices)
    val_size = int(N * args.val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_ds = CachedDataset(
        phys_f_norm[train_idx],
        phys_c[train_idx],
        dinov2_cls[train_idx],
        dinov2_reg[train_idx],
        labels[train_idx],
    )
    val_ds = CachedDataset(
        phys_f_norm[val_idx],
        phys_c[val_idx],
        dinov2_cls[val_idx],
        dinov2_reg[val_idx],
        labels[val_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # 2. Instantiate Phase 2 Model
    model = DualStreamHybridDetector(
        load_pretrained_dinov2=False,
        gate_mode="phase2",
        dropout=0.15,
    ).to(args.device)

    # Initialize weights from baseline v5 physics & projections where compatible
    v5_path = ROOT_DIR / "models/universal_v5_disagreement_gate/best_model.pt"
    if v5_path.exists():
        print(f"Transferring compatible weights from {v5_path.name}...")
        v5_ckpt = torch.load(v5_path, map_location="cpu", weights_only=False)
        m_dict = model.state_dict()
        transferred = 0
        for k, v in v5_ckpt["model_state_dict"].items():
            if k in m_dict and m_dict[k].shape == v.shape:
                m_dict[k] = v
                transferred += 1
        model.load_state_dict(m_dict)
        print(f"Successfully transferred {transferred} compatible parameter tensors.")

    # Only train the fusion head and physics stream (keep DINOv2 frozen)
    trainable_params = [
        {"params": list(model.fusion_head.parameters()), "lr": args.lr},
        {"params": list(model.physics_stream.parameters()), "lr": args.lr * 0.5},
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    bce = nn.BCEWithLogitsLoss()

    out_dir = ROOT_DIR / "models/universal_v6_phase2_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    best_val_acc = 0.0
    history = []

    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            pf = batch["physics_features"].to(args.device)
            pc = batch["physics_confidences"].to(args.device)
            d_cls = batch["dinov2_cls"].to(args.device)
            d_reg = batch["dinov2_regional"].to(args.device)
            y = batch["label"].to(args.device)

            # Apply Disagreement Augmentation
            aug_cls, aug_reg, is_counter = apply_disagreement_augmentation(
                d_cls, d_reg, y, p_disagree=args.disagree_ratio
            )

            optimizer.zero_grad()
            out = model(
                physics_features=pf,
                physics_confidences=pc,
                dinov2_cls=aug_cls,
                dinov2_regional=aug_reg,
            )

            loss, loss_dict = compute_phase2_loss(out, y, is_counter, bce)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        train_loss /= max(1, train_batches)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_alphas = []
        val_temps = []

        with torch.no_grad():
            for batch in val_loader:
                pf = batch["physics_features"].to(args.device)
                pc = batch["physics_confidences"].to(args.device)
                d_cls = batch["dinov2_cls"].to(args.device)
                d_reg = batch["dinov2_regional"].to(args.device)
                y = batch["label"]

                out = model(
                    physics_features=pf,
                    physics_confidences=pc,
                    dinov2_cls=d_cls,
                    dinov2_regional=d_reg,
                )

                prob = out["prob_final"].cpu().numpy()
                val_preds.extend(prob.tolist())
                val_labels.extend(y.numpy().tolist())
                val_alphas.extend(out["alpha"].cpu().numpy().tolist())
                val_temps.extend(out["temperature"].cpu().numpy().tolist())

        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels)
        val_acc = float(accuracy_score(val_labels_arr, (val_preds_arr >= 0.5).astype(int)))
        val_auc = float(roc_auc_score(val_labels_arr, val_preds_arr))
        mean_alpha = float(np.mean(val_alphas))
        mean_temp = float(np.mean(val_temps))

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc*100:6.2f}% | "
            f"Val AUC: {val_auc:7.4f} | "
            f"Mean Alpha: {mean_alpha:.3f} | "
            f"Mean Temp: {mean_temp:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "mean_alpha": mean_alpha,
            "mean_temp": mean_temp,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "standardizer": standardizer,
                    "gate_mode": "phase2",
                    "best_metrics": {"val_acc": val_acc, "val_auc": val_auc},
                },
                out_dir / "best_model.pt",
            )
            print(f"  [*] Saved new best Phase 2 model! (Val AUC: {val_auc:.4f}, Acc: {val_acc*100:.2f}%)")

    # Save final model and training history
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "standardizer": standardizer,
            "gate_mode": "phase2",
        },
        out_dir / "final_model.pt",
    )

    with open(ROOT_DIR / "reports/phase2_alignment_training_log.json", "w") as f:
        json.dump({"history": history, "best_val_auc": best_val_auc, "best_val_acc": best_val_acc}, f, indent=2)

    print("\nPhase 2 Training Completed successfully!")
    print(f"Best Checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
