"""Trains Attention-Gated Cross-Generator Physics Classifier for 50 epochs."""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.models.cross_gen_gated import GatedCrossGenClassifier
from src.data.dataset import PhysicsFeatureDataset
from src.utils.metrics import evaluate_predictions, format_evaluation_summary


def train_model(
    train_cache: Path,
    test_cache: Path,
    checkpoint_dir: Path,
    val_split: float = 0.15,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Validation Split: {val_split * 100:.1f}% for Honest Checkpoint Selection")

    # Load datasets
    full_train_dataset = PhysicsFeatureDataset.from_cache(train_cache)
    test_dataset = PhysicsFeatureDataset.from_cache(test_cache)

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
    else:
        train_dataset = full_train_dataset
        val_dataset = test_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples, "
        f"and {len(test_dataset)} unseen test samples."
    )

    # Initialize model
    model = GatedCrossGenClassifier(module_dims=(5, 4, 3, 2), dropout_rate=0.2).to(device)

    # Criterion & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0
    best_model_path = checkpoint_dir / "best_model.pt"

    print("\nStarting Training (50 Epochs, AdamW lr=1e-3, weight_decay=1e-4)...")
    print("-" * 75)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Val AUC-ROC':<14}{'Val EER':<14}{'Val Acc':<12}{'Status'}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for features, confidences, labels in train_loader:
            features = features.to(device)
            confidences = confidences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features, confidences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_train_loss = total_loss / max(1, batches)

        # Validation evaluation strictly on validation split
        model.eval()
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for features, confidences, labels in val_loader:
                features = features.to(device)
                confidences = confidences.to(device)
                logits = model(features, confidences)
                probs = torch.sigmoid(logits)

                val_probs.extend(probs.cpu().numpy().tolist())
                val_targets.extend(labels.numpy().tolist())

        val_metrics = evaluate_predictions(np.array(val_targets), np.array(val_probs))
        auc = val_metrics["auc_roc"]
        eer = val_metrics["eer"]
        acc = val_metrics["accuracy"]

        status = ""
        if auc >= best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)
            status = "(* Best Checkpoint Saved *)"

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"{epoch:<8}{avg_train_loss:<14.4f}{auc:<14.4f}{eer * 100:<13.2f}%{acc * 100:<11.2f}% {status}"
            )

    # Save latest
    torch.save(model.state_dict(), checkpoint_dir / "latest_model.pt")
    print("-" * 75)
    print(f"Training completed. Best Val AUC: {best_auc:.4f}. Model saved to {best_model_path}")

    # Load best model for final evaluation report on held-out test set (touched once)
    print("\n" + "#" * 75)
    print("FINAL HELD-OUT TEST EVALUATION (TOUCHED ONCE AT COMPLETION)")
    print("#" * 75)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    final_probs = []
    final_targets = []
    with torch.no_grad():
        for features, confidences, labels in test_loader:
            features = features.to(device)
            confidences = confidences.to(device)
            logits = model(features, confidences)
            probs = torch.sigmoid(logits)
            final_probs.extend(probs.cpu().numpy().tolist())
            final_targets.extend(labels.numpy().tolist())

    final_confs = []
    with torch.no_grad():
        for features, confidences, labels in test_loader:
            final_confs.extend(confidences.cpu().numpy().tolist())

    from src.utils.metrics import evaluate_observability_subsets

    sub_results = evaluate_observability_subsets(
        np.array(final_targets), np.array(final_probs), np.array(final_confs), obs_threshold=1.0
    )

    print("\n" + format_evaluation_summary(sub_results["overall"], title="Final Test Evaluation"))

    print("\n" + "=" * 65)
    print(f"Observability Gating (O >= 1.00):")
    print(f"  Total: {sub_results['total_samples']} | Observable: {sub_results['num_observable']} | Indeterminate: {sub_results['num_indeterminate']}")
    print(f"  Overall AUC:    {sub_results['overall']['auc_roc']:.4f}  -->  Observable AUC:    {sub_results['observable']['auc_roc']:.4f}")
    print(f"  Overall Acc:    {sub_results['overall']['accuracy']*100:.2f}%  -->  Observable Acc:    {sub_results['observable']['accuracy']*100:.2f}%")
    print(f"  Overall EER:    {sub_results['overall']['eer']*100:.2f}%  -->  Observable EER:    {sub_results['observable']['eer']*100:.2f}%")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train Gated Cross-Gen Classifier")
    parser.add_argument("--train-cache", type=str, default="data/cache/train_features.pt")
    parser.add_argument("--test-cache", type=str, default="data/cache/test_features.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    train_cache = WORKSPACE_ROOT / args.train_cache
    test_cache = WORKSPACE_ROOT / args.test_cache
    checkpoint_dir = WORKSPACE_ROOT / args.checkpoint_dir

    train_model(
        train_cache=train_cache,
        test_cache=test_cache,
        checkpoint_dir=checkpoint_dir,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
