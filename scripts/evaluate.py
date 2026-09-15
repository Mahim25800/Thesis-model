"""Evaluates trained Gated Cross-Gen Classifier on unseen test data."""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.models.cross_gen_gated import GatedCrossGenClassifier
from src.data.dataset import PhysicsFeatureDataset
from src.utils.metrics import evaluate_predictions, format_evaluation_summary


def evaluate_model(
    checkpoint_path: Path,
    test_cache: Path,
    output_json: Path = None,
    obs_threshold: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Loading unseen test cache from: {test_cache}")

    test_dataset = PhysicsFeatureDataset.from_cache(test_cache)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GatedCrossGenClassifier(module_dims=(5, 4, 3, 2)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    all_probs = []
    all_targets = []
    all_confs = []

    with torch.no_grad():
        for features, confidences, labels in test_loader:
            features = features.to(device)
            confidences = confidences.to(device)
            output = model(features, confidences)
            logits = output[0] if isinstance(output, tuple) else output
            probs = torch.sigmoid(logits.squeeze())

            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_confs.extend(confidences.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    confs = np.array(all_confs)

    from src.utils.metrics import evaluate_observability_subsets, compute_eer

    sub_results = evaluate_observability_subsets(
        y_true, y_probs, confs, obs_threshold=obs_threshold
    )

    overall_metrics = sub_results["overall"]
    obs_metrics = sub_results["observable"]

    eer_val, eer_thresh = compute_eer(y_true, y_probs)
    optimal_metrics = evaluate_predictions(y_true, y_probs, threshold=eer_thresh)

    print("\n" + format_evaluation_summary(overall_metrics, title="Overall Population (Static Threshold 0.50)"))
    print("\n" + format_evaluation_summary(optimal_metrics, title=f"Cost-Sensitive Population (Optimal EER Threshold Tau* = {eer_thresh:.4f})"))

    print("\n" + "=" * 60)
    print(f"Observability Gating Analysis (Threshold O >= {obs_threshold:.2f}):")
    print(f"  Total Test Samples:        {sub_results['total_samples']}")
    print(f"  Observable Samples:        {sub_results['num_observable']} ({sub_results['num_observable'] / sub_results['total_samples'] * 100:.1f}%)")
    print(f"  Low-Evidence/Indeterminate:{sub_results['num_indeterminate']} ({sub_results['num_indeterminate'] / sub_results['total_samples'] * 100:.1f}%)")
    print("-" * 60)
    print(f"  Overall AUC-ROC:           {overall_metrics['auc_roc']:.4f}  -->  Observable AUC-ROC:   {obs_metrics['auc_roc']:.4f}")
    print(f"  Overall Accuracy:          {overall_metrics['accuracy'] * 100:.2f}%  -->  Observable Accuracy:  {obs_metrics['accuracy'] * 100:.2f}%")
    print(f"  Overall EER:               {overall_metrics['eer'] * 100:.2f}%  -->  Observable EER:       {obs_metrics['eer'] * 100:.2f}%")
    print("=" * 60)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(sub_results, f, indent=2)
        print(f"\nSaved metrics to {output_json}")

    return sub_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gated Cross-Gen Classifier")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt")
    parser.add_argument("--test-cache", type=str, default="data/cache/test_features.pt")
    parser.add_argument("--output-json", type=str, default="models/evaluation_results.json")
    parser.add_argument("--obs-threshold", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint_path = WORKSPACE_ROOT / args.checkpoint
    test_cache = WORKSPACE_ROOT / args.test_cache
    output_json = WORKSPACE_ROOT / args.output_json if args.output_json else None

    evaluate_model(checkpoint_path, test_cache, output_json, obs_threshold=args.obs_threshold)


if __name__ == "__main__":
    main()
