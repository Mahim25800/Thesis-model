"""Train an audited multi-entity physics-consistency candidate on legacy caches.

Selection uses only the frozen development partition.  Calibration and the
operating threshold use only the frozen calibration partition.  The historical
test partition is read once after those choices are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_scaled import evaluate_selective_abstention
from src.data.dataset import PhysicsFeatureDataset
from src.models.cross_gen_gated import TransformerPhysicsCrossGenHead
from src.models.improved import legacy_cache_matrix, probability_logits
from src.models.multi_entity import (
    ENTITY_DIMS,
    ENTITY_NAMES,
    MultiEntityConsistencyHead,
    fit_feature_standardizer,
    multi_entity_loss,
    standardize_features,
)
from src.utils.decision_policy import apply_policy_calibration, fit_decision_policy
from src.utils.metrics import evaluate_predictions


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def legacy_probabilities(model, dataset: PhysicsFeatureDataset, indices: np.ndarray) -> np.ndarray:
    batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), 512):
            rows = indices[start:start + 512]
            batches.append(torch.sigmoid(model(dataset.features[rows], dataset.confidences[rows])[0]).cpu().numpy())
    return np.concatenate(batches)


def v2_probabilities(v2_artifact: dict, legacy, dataset: PhysicsFeatureDataset, indices: np.ndarray) -> np.ndarray:
    matrix = legacy_cache_matrix(dataset.features[indices], dataset.confidences[indices])
    tree = v2_artifact["estimator"].predict_proba(matrix)[:, 1]
    transformer = legacy_probabilities(legacy, dataset, indices)
    return float(v2_artifact["blend_weight"]) * tree + (1.0 - float(v2_artifact["blend_weight"])) * transformer


def probabilities(model: MultiEntityConsistencyHead, features: torch.Tensor, confidences: torch.Tensor,
                  standardizer: dict, device: torch.device) -> np.ndarray:
    model.to(device).eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            x = standardize_features(features[start:start + 1024].to(device), standardizer)
            c = confidences[start:start + 1024].to(device)
            outputs.append(torch.sigmoid(model(x, c)).cpu().numpy())
    return np.concatenate(outputs)


def train_one(seed: int, train_features: torch.Tensor, train_confidences: torch.Tensor, train_labels: torch.Tensor,
              development_features: torch.Tensor, development_confidences: torch.Tensor,
              development_labels: np.ndarray, standardizer: dict, device: torch.device,
              epochs: int, patience: int, batch_size: int) -> tuple[MultiEntityConsistencyHead, dict]:
    seed_everything(seed)
    model = MultiEntityConsistencyHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    train_x = standardize_features(train_features, standardizer)
    loader = DataLoader(TensorDataset(train_x, train_confidences, train_labels), batch_size=batch_size,
                        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    best_state, best_metrics, stale = None, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, confidences, labels in loader:
            x, confidences, labels = x.to(device), confidences.to(device), labels.to(device)
            # Drop one available entity in a minority of training examples.  The
            # model must still work when a physical cue is unobservable.
            available = torch.rand_like(confidences) >= 0.08
            augmented_confidences = confidences * available
            output = model(x, augmented_confidences, return_details=True)
            losses = multi_entity_loss(output, labels)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        dev_probability = probabilities(model, development_features, development_confidences, standardizer, device)
        metrics = evaluate_predictions(development_labels, dev_probability, 0.5)
        score = (metrics["auc_roc"], metrics["accuracy"])
        best_score = (-np.inf, -np.inf) if best_metrics is None else (best_metrics["auc_roc"], best_metrics["accuracy"])
        if score > best_score:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics, stale = metrics, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu(), {"seed": seed, "epochs_completed": epoch, "development": best_metrics}


def evaluate_with_policy(labels, raw_probability, confidences, policy):
    calibrated = apply_policy_calibration(probability_logits(raw_probability), policy)
    return {
        "binary": evaluate_predictions(labels, calibrated, policy["decision_threshold"]),
        "raw_threshold_0_5": evaluate_predictions(labels, raw_probability, 0.5),
        "selective": evaluate_selective_abstention(
            labels, calibrated, confidences, policy["tau_low"], policy["tau_high"], policy["obs_threshold"]
        ),
    }, calibrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models/multi_entity_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 47, 53])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0 or not args.seeds:
        raise ValueError("epochs, patience, batch size, and at least one seed must be positive")
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain inside pipeline_40k")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory already contains a run; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)

    integrity_dir = ROOT / "models/improved_v1/integrity"
    with np.load(integrity_dir / "split_indices.npz") as split_file:
        train_indices = split_file["train_indices"].copy()
        development_indices = split_file["development_indices"].copy()
        calibration_indices = split_file["calibration_indices"].copy()
        test_indices = split_file["test_indices"].copy()
    manifest = json.loads((integrity_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train_cache = ROOT / "data/cache_scaled/train_scaled_features.pt"
    test_cache = ROOT / "data/cache_scaled/test_scaled_features.pt"
    if sha256(train_cache) != manifest["source_caches"]["train"]["sha256"] or sha256(test_cache) != manifest["source_caches"]["test"]["sha256"]:
        raise ValueError("Caches changed after the integrity audit")
    training = PhysicsFeatureDataset.from_cache(train_cache)
    testing = PhysicsFeatureDataset.from_cache(test_cache)
    standardizer = fit_feature_standardizer(training.features[train_indices].numpy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    started = time.perf_counter()
    models, training_runs = [], []
    for seed in args.seeds:
        model, run = train_one(
            seed, training.features[train_indices], training.confidences[train_indices], training.labels[train_indices],
            training.features[development_indices], training.confidences[development_indices],
            training.labels[development_indices].numpy().astype(int), standardizer, device,
            args.epochs, args.patience, args.batch_size,
        )
        models.append(model)
        training_runs.append(run)

    def ensemble_probability(features, confidences):
        return np.mean([probabilities(model, features, confidences, standardizer, device) for model in models], axis=0)

    legacy_path = ROOT / "models/gated_cross_gen_40k_best.pt"
    legacy = TransformerPhysicsCrossGenHead().eval()
    legacy.load_state_dict(torch.load(legacy_path, map_location="cpu", weights_only=True))
    v2_path = ROOT / "models/improved_v2/model.joblib"
    v2_artifact = joblib.load(v2_path)
    if v2_artifact.get("feature_schema") != "legacy_cache_14_plus_4_v1":
        raise ValueError("Current v2 artifact is not compatible with the audited cache")

    development_entity = ensemble_probability(training.features[development_indices], training.confidences[development_indices])
    development_v2 = v2_probabilities(v2_artifact, legacy, training, development_indices)
    development_labels = training.labels[development_indices].numpy().astype(int)
    blend_candidates = []
    for entity_weight in np.linspace(0.0, 1.0, 11):
        raw = entity_weight * development_entity + (1.0 - entity_weight) * development_v2
        blend_candidates.append({"entity_weight": float(entity_weight), "metrics": evaluate_predictions(development_labels, raw, 0.5)})
    selected_blend = max(blend_candidates, key=lambda row: (row["metrics"]["accuracy"], row["metrics"]["auc_roc"]))
    entity_weight = selected_blend["entity_weight"]
    selection = {
        "protocol": "Architecture and seeds fixed before test access; entity/v2 blend selected on development only",
        "entity_names": list(ENTITY_NAMES), "entity_dims": list(ENTITY_DIMS),
        "training_runs": training_runs, "blend_candidates": blend_candidates,
        "selected_blend": selected_blend,
        "v2_development": evaluate_predictions(development_labels, development_v2, 0.5),
        "entity_only_development": evaluate_predictions(development_labels, development_entity, 0.5),
    }
    write_json(output / "selection.json", selection)

    calibration_entity = ensemble_probability(training.features[calibration_indices], training.confidences[calibration_indices])
    calibration_v2 = v2_probabilities(v2_artifact, legacy, training, calibration_indices)
    calibration_raw = entity_weight * calibration_entity + (1.0 - entity_weight) * calibration_v2
    calibration_labels = training.labels[calibration_indices].numpy().astype(int)
    policy = fit_decision_policy(probability_logits(calibration_raw), calibration_labels, obs_threshold=1.0, abstention_margin=0.10)
    write_json(output / "policy.json", policy)

    payload = {
        "schema_version": 1,
        "feature_schema": "legacy_cache_14_plus_4_v1",
        "entity_names": list(ENTITY_NAMES), "entity_dims": list(ENTITY_DIMS),
        "architecture": {"d_model": 48, "pair_dim": 64, "dropout": 0.15},
        "standardizer": standardizer, "entity_weight": entity_weight,
        "model_states": [model.state_dict() for model in models],
        "policy": policy,
        "v2_artifact_sha256": sha256(v2_path), "legacy_checkpoint_sha256": sha256(legacy_path),
    }
    torch.save(payload, output / "model.pt")

    test_entity = ensemble_probability(testing.features[test_indices], testing.confidences[test_indices])
    test_v2 = v2_probabilities(v2_artifact, legacy, testing, test_indices)
    test_raw = entity_weight * test_entity + (1.0 - entity_weight) * test_v2
    test_labels = testing.labels[test_indices].numpy().astype(int)
    test_confidences = testing.confidences[test_indices].numpy()
    candidate, candidate_probability = evaluate_with_policy(test_labels, test_raw, test_confidences, policy)
    entity_only = evaluate_predictions(test_labels, test_entity, 0.5)
    v2_metrics, v2_probability = evaluate_with_policy(test_labels, test_v2, test_confidences, v2_artifact["policy"])
    correct_difference = ((candidate_probability >= policy["decision_threshold"]) == test_labels).astype(float) - \
                         ((v2_probability >= v2_artifact["policy"]["decision_threshold"]) == test_labels).astype(float)
    rng = np.random.default_rng(42)
    bootstrap = [correct_difference[rng.integers(0, len(correct_difference), len(correct_difference))].mean() for _ in range(2000)]
    metadata = {
        "schema_version": 1, "device": str(device), "seeds": args.seeds,
        "training_seconds": time.perf_counter() - started,
        "train_cache_sha256": sha256(train_cache), "test_cache_sha256": sha256(test_cache),
        "integrity_split_sha256": sha256(integrity_dir / "split_indices.npz"),
        "limitations": manifest["limitations"] + [
            "The historical normal residual columns came from an unrecoverable random normal head; this candidate cannot validate learned normal physics.",
            "No unseen-generator or external dataset has been used in this run.",
        ],
    }
    write_json(output / "metadata.json", metadata)
    report = {
        "protocol": "Audited historical cache; development-only blend selection; calibration-only policy; test read after choices frozen",
        "test_samples": int(len(test_indices)), "candidate": candidate,
        "entity_only_fixed_0_5": entity_only, "v2_frozen_policy": v2_metrics,
        "comparison_to_v2": {
            "accuracy_difference": float(correct_difference.mean()),
            "paired_bootstrap_95_percent_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "bootstrap_repetitions": 2000,
        },
        "model_sha256": sha256(output / "model.pt"),
        "external_validation_required": True, "limitations": metadata["limitations"],
    }
    write_json(output / "evaluation.json", report)
    np.savez_compressed(output / "test_predictions.npz", row_indices=test_indices, labels=test_labels,
                        candidate_probability=candidate_probability, entity_probability=test_entity,
                        v2_probability=v2_probability)
    print(json.dumps({
        "selected_entity_weight": entity_weight,
        "candidate_test_accuracy": candidate["binary"]["accuracy"],
        "candidate_test_auc": candidate["binary"]["auc_roc"],
        "v2_test_accuracy": v2_metrics["binary"]["accuracy"],
        "comparison": report["comparison_to_v2"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
