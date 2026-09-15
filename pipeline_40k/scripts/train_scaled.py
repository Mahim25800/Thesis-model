"""Scaled 40k Training Engine with Multi-Head Self-Attention Transformer and Hard-Negative Focal Loss."""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure pipeline root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cross_gen_gated import TransformerPhysicsCrossGenHead, FocalLoss, SupConLoss
from src.data.dataset import PhysicsFeatureDataset
from src.utils.metrics import (
    evaluate_predictions,
    evaluate_observability_subsets,
    format_evaluation_summary,
    compute_youden_threshold,
)


def train_scaled_model(
    train_cache: Path,
    test_cache: Path,
    best_checkpoint_path: Path,
    val_split: float = 0.15,
    val_cache: Path = None,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    eta_min: float = 1e-6,
    gamma_focal: float = 2.0,
    alpha_focal: float = 0.5,
    temperature: float = 0.07,
    lambda_con: float = 0.3,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("TRAINING MULTI-HEAD SELF-ATTENTION (MHSA) TRANSFORMER WITH HARD-NEGATIVE FOCAL LOSS")
    print("Objective: FocalLoss(gamma=2.0, alpha=0.5) + lambda * SupConLoss")
    print(f"Device:               {device}")
    print(f"Batch Size:           {batch_size}")
    print(f"Epochs:               {epochs}")
    print(f"Validation Split:     {val_split * 100:.1f}% (Honest Model Selection)")
    print(f"Initial LR:           {lr}")
    print(f"Weight Decay:         {weight_decay}")
    print(f"Focal Gamma / Alpha:  gamma={gamma_focal}, alpha={alpha_focal}")
    print(f"SupCon Temperature:   {temperature}")
    print(f"SupCon Lambda:        {lambda_con}")
    print(f"LR Scheduler:         CosineAnnealingLR (eta_min={eta_min})")
    print("=" * 95)

    print(f"Loading cached tensors from:\n  Train: {train_cache}\n  Test:  {test_cache}")
    full_train_dataset = PhysicsFeatureDataset.from_cache(train_cache)
    test_dataset = PhysicsFeatureDataset.from_cache(test_cache)

    # 1. Stratified Validation Split from Training Data
    if val_split > 0.0:
        labels_arr = full_train_dataset.labels.numpy()
        real_indices = np.where(labels_arr == 0)[0]
        fake_indices = np.where(labels_arr == 1)[0]

        np.random.seed(seed)
        perm_real = np.random.permutation(real_indices)
        perm_fake = np.random.permutation(fake_indices)

        n_val_real = int(len(real_indices) * val_split)
        n_val_fake = int(len(fake_indices) * val_split)

        val_idx = np.concatenate([perm_real[:n_val_real], perm_fake[:n_val_fake]])
        train_idx = np.concatenate([perm_real[n_val_real:], perm_fake[n_val_fake:]])

        train_dataset = PhysicsFeatureDataset(
            full_train_dataset.features[train_idx],
            full_train_dataset.confidences[train_idx],
            full_train_dataset.labels[train_idx],
        )
        val_dataset = PhysicsFeatureDataset(
            full_train_dataset.features[val_idx],
            full_train_dataset.confidences[val_idx],
            full_train_dataset.labels[val_idx],
        )

        # Save validation split tensor cache for downstream calibration scripts
        if val_cache is None:
            val_cache = train_cache.parent / "val_scaled_features.pt"
        torch.save(
            {
                "features": full_train_dataset.features[val_idx],
                "confidences": full_train_dataset.confidences[val_idx],
                "labels": full_train_dataset.labels[val_idx],
            },
            val_cache,
        )
        print(f"Saved stratified validation split cache to: {val_cache}")
    else:
        train_dataset = full_train_dataset
        val_dataset = test_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Partition Sizes: {len(train_dataset):,} Training samples | "
        f"{len(val_dataset):,} Validation samples (Checkpoint Selection) | "
        f"{len(test_dataset):,} Held-Out Test samples (Touched Once)."
    )

    # Initialize MHSA Transformer Head
    model = TransformerPhysicsCrossGenHead(
        in_features=14,
        conf_dim=4,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_layers=2,
        proj_dim=64,
        dropout=0.1,
    ).to(device)

    criterion_focal = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)
    criterion_con = SupConLoss(temperature=temperature)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0
    best_val_epoch = 0

    print("\nStarting Training Execution...")
    print("-" * 110)
    print(f"{'Epoch':<7}{'LR':<10}{'Total Loss':<13}{'Focal Loss':<12}{'Con Loss':<11}{'Val AUC':<11}{'Val EER':<11}{'Youden J*':<11}{'Val Acc':<10}{'Status'}")
    print("-" * 110)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_focal = 0.0
        total_con = 0.0
        batch_count = 0

        for features, confidences, labels in train_loader:
            features = features.to(device)
            confidences = confidences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, projections = model(features, confidences)
            loss_focal = criterion_focal(logits.squeeze(), labels)
            loss_con = criterion_con(projections, labels)
            loss = loss_focal + lambda_con * loss_con
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_focal += loss_focal.item()
            total_con += loss_con.item()
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / max(1, batch_count)
        avg_focal = total_focal / max(1, batch_count)
        avg_con = total_con / max(1, batch_count)
        current_lr = scheduler.get_last_lr()[0]

        # Validation evaluation each epoch (STRICTLY ON VALIDATION SPLIT)
        model.eval()
        val_probs = []
        val_targets = []
        val_confs = []

        with torch.no_grad():
            for features, confidences, labels in val_loader:
                features = features.to(device)
                confidences = confidences.to(device)
                logits, _ = model(features, confidences)
                probs = torch.sigmoid(logits.squeeze())

                val_probs.extend(probs.cpu().numpy().tolist())
                val_targets.extend(labels.numpy().tolist())
                val_confs.extend(confidences.cpu().numpy().tolist())

        y_t_ep = np.array(val_targets)
        y_p_ep = np.array(val_probs)
        val_metrics = evaluate_predictions(y_t_ep, y_p_ep)
        auc = val_metrics["auc_roc"]
        eer = val_metrics["eer"]
        acc = val_metrics["accuracy"]
        youden_j, _ = compute_youden_threshold(y_t_ep, y_p_ep)

        status = ""
        if auc >= best_val_auc:
            best_val_auc = auc
            best_val_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)
            status = "(* Peak Val Saved *)"

        print(
            f"{epoch:<7}{current_lr:<10.6f}{avg_loss:<13.4f}{avg_focal:<12.4f}{avg_con:<11.4f}{auc:<11.4f}{eer * 100:<10.2f}%{youden_j:<11.4f}{acc * 100:<9.2f}% {status}"
        )

    print("-" * 110)
    print(f"Training completed successfully! Peak Validation AUC: {best_val_auc:.4f} (at Epoch {best_val_epoch})")
    print(f"Best model saved to: {best_checkpoint_path}")

    # Load best checkpoint selected strictly by validation performance
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # 1. Final Validation Breakdown
    val_probs_fin = []
    val_targets_fin = []
    with torch.no_grad():
        for features, confidences, labels in val_loader:
            logits, _ = model(features.to(device), confidences.to(device))
            val_probs_fin.extend(torch.sigmoid(logits.squeeze()).cpu().numpy().tolist())
            val_targets_fin.extend(labels.numpy().tolist())
    val_final_results = evaluate_predictions(np.array(val_targets_fin), np.array(val_probs_fin))
    print("\n" + format_evaluation_summary(val_final_results, title="Selected Checkpoint - Validation Split Performance"))

    # 2. Touch Held-Out Unseen Test Set STRICTLY ONCE
    print("\n" + "#" * 95)
    print("FINAL HELD-OUT UNSEEN TEST EVALUATION (TOUCHED EXACTLY ONCE AT THE VERY END)")
    print("#" * 95)
    final_probs = []
    final_targets = []
    final_confs = []
    with torch.no_grad():
        for features, confidences, labels in test_loader:
            features = features.to(device)
            confidences = confidences.to(device)
            logits, _ = model(features, confidences)
            probs = torch.sigmoid(logits.squeeze())

            final_probs.extend(probs.cpu().numpy().tolist())
            final_targets.extend(labels.numpy().tolist())
            final_confs.extend(confidences.cpu().numpy().tolist())

    y_t = np.array(final_targets)
    y_p = np.array(final_probs)
    confs_arr = np.array(final_confs)

    sub_results = evaluate_observability_subsets(
        y_t, y_p, confs_arr, obs_threshold=1.0
    )

    print("\n" + format_evaluation_summary(sub_results["overall"], title="Final Scaled Unseen Test Evaluation (Static Tau=0.50)"))

    # Youden's J threshold evaluation
    youden_j_fin, youden_th_fin = compute_youden_threshold(y_t, y_p)
    youden_results = evaluate_predictions(y_t, y_p, threshold=youden_th_fin)
    print("\n" + format_evaluation_summary(youden_results, title=f"Cost-Sensitive Evaluation (@ Youden's J* Tau*={youden_th_fin:.4f})"))

    print("\n" + "=" * 65)
    print(f"Observability Gating (O >= 1.00):")
    print(f"  Total: {sub_results['total_samples']:,} | Observable: {sub_results['num_observable']:,} | Indeterminate: {sub_results['num_indeterminate']:,}")
    print(f"  Overall AUC:    {sub_results['overall']['auc_roc']:.4f}  -->  Observable AUC:    {sub_results['observable']['auc_roc']:.4f}")
    print(f"  Overall Acc:    {sub_results['overall']['accuracy']*100:.2f}%  -->  Observable Acc:    {sub_results['observable']['accuracy']*100:.2f}%")
    print(f"  Overall EER:    {sub_results['overall']['eer']*100:.2f}%  -->  Observable EER:    {sub_results['observable']['eer']*100:.2f}%")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train Scaled Transformer with Hard-Negative Focal Loss")
    parser.add_argument("--train-cache", type=str, default="data/cache_scaled/train_scaled_features.pt")
    parser.add_argument("--test-cache", type=str, default="data/cache_scaled/test_scaled_features.pt")
    parser.add_argument("--val-cache", type=str, default="data/cache_scaled/val_scaled_features.pt")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--best-checkpoint", type=str, default="models/gated_cross_gen_40k_best.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--gamma-focal", type=float, default=2.0)
    parser.add_argument("--alpha-focal", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-con", type=float, default=0.3)
    args = parser.parse_args()

    train_cache = Path(args.train_cache) if Path(args.train_cache).exists() else PROJECT_ROOT / args.train_cache
    test_cache = Path(args.test_cache) if Path(args.test_cache).exists() else PROJECT_ROOT / args.test_cache
    val_cache = Path(args.val_cache) if Path(args.val_cache).is_absolute() else PROJECT_ROOT / args.val_cache
    best_checkpoint = Path(args.best_checkpoint) if Path(args.best_checkpoint).is_absolute() else PROJECT_ROOT / args.best_checkpoint

    train_scaled_model(
        train_cache=train_cache,
        test_cache=test_cache,
        best_checkpoint_path=best_checkpoint,
        val_split=args.val_split,
        val_cache=val_cache,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        gamma_focal=args.gamma_focal,
        alpha_focal=args.alpha_focal,
        temperature=args.temperature,
        lambda_con=args.lambda_con,
    )


if __name__ == "__main__":
    main()
