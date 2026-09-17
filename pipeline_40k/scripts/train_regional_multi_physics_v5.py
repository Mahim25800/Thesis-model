"""Train the regional DSINE multi-physics detector with group-disjoint splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_scaled import evaluate_selective_abstention
from src.models.improved import probability_logits
from src.models.multi_entity import ENTITY_DIMS, ENTITY_NAMES
from src.models.regional_multi_physics import (
    REGION_COUNT,
    RegionalMultiPhysicsHead,
    fit_regional_standardizer,
    regional_multi_physics_loss,
    standardize_regional,
)
from src.utils.decision_policy import apply_policy_calibration, fit_decision_policy
from src.utils.metrics import evaluate_predictions


SCHEMA = "regional_physics_dsine_v2_5x14_features_5x4_confidences"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or [row["source_position"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"Manifest is not ordered: {path}")
    return rows


def load_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    features = cache["features"].float()
    confidences = cache["confidences"].float()
    labels = cache["labels"].float()
    if features.ndim != 3 or features.shape[1:] != (REGION_COUNT, sum(ENTITY_DIMS)):
        raise ValueError(f"Unexpected regional feature schema: {path}")
    if confidences.shape != (len(features), REGION_COUNT, len(ENTITY_DIMS)) or labels.shape != (len(features),):
        raise ValueError(f"Misaligned regional cache: {path}")
    if not torch.isfinite(features).all() or not torch.isfinite(confidences).all():
        raise ValueError(f"Non-finite regional cache: {path}")
    return features, confidences, labels


def split_train_development_calibration(labels: np.ndarray, groups: np.ndarray):
    indices = np.arange(len(labels))
    remaining, calibration = next(StratifiedGroupKFold(
        n_splits=7, shuffle=True, random_state=20260916
    ).split(indices, labels, groups))
    fit_relative, development_relative = next(StratifiedGroupKFold(
        n_splits=6, shuffle=True, random_state=20260917
    ).split(indices[remaining], labels[remaining], groups[remaining]))
    train, development = remaining[fit_relative], remaining[development_relative]
    for left, right in ((train, development), (train, calibration), (development, calibration)):
        if set(groups[left]) & set(groups[right]):
            raise RuntimeError("Difference-hash groups leaked across partitions")
    return train, development, calibration


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_regions(features: torch.Tensor, confidences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply label-preserving flips, mild noise, and missing-cue simulation."""
    features, confidences = features.clone(), confidences.clone()
    for permutation in ([0, 2, 1, 4, 3], [0, 3, 4, 1, 2]):
        selected = torch.rand(len(features), device=features.device) < 0.5
        if selected.any():
            features[selected] = features[selected][:, permutation]
            confidences[selected] = confidences[selected][:, permutation]
    features = features + 0.025 * torch.randn_like(features)
    entity_available = torch.rand_like(confidences) >= 0.08
    local_available = torch.ones_like(confidences)
    local_available[:, 1:] = (torch.rand(
        len(features), REGION_COUNT - 1, 1, device=features.device
    ) >= 0.08)
    return features, confidences * entity_available * local_available


def probabilities(model: RegionalMultiPhysicsHead, features: torch.Tensor, confidences: torch.Tensor,
                  standardizer: dict, device: torch.device) -> np.ndarray:
    model.to(device).eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            x = standardize_regional(features[start:start + 512].to(device), standardizer)
            c = confidences[start:start + 512].to(device)
            values.append(torch.sigmoid(model(x, c)).cpu().numpy())
    return np.concatenate(values)


def train_one(seed: int, train_features: torch.Tensor, train_confidences: torch.Tensor,
              train_labels: torch.Tensor, development_features: torch.Tensor,
              development_confidences: torch.Tensor, development_labels: np.ndarray,
              standardizer: dict, device: torch.device, epochs: int, patience: int,
              batch_size: int) -> tuple[RegionalMultiPhysicsHead, dict]:
    seed_everything(seed)
    architecture = {"d_model": 48, "pair_dim": 64, "dropout": 0.18}
    model = RegionalMultiPhysicsHead(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=3e-3)
    train_values = standardize_regional(train_features, standardizer)
    loader = DataLoader(
        TensorDataset(train_values, train_confidences, train_labels), batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_metrics, stale = None, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for values, confidences, labels in loader:
            values = values.to(device, non_blocking=True)
            confidences = confidences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            values, confidences = augment_regions(values, confidences)
            output = model(values, confidences, return_details=True)
            loss = regional_multi_physics_loss(output, labels)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        development_probability = probabilities(
            model, development_features, development_confidences, standardizer, device
        )
        metrics = evaluate_predictions(development_labels, development_probability, 0.5)
        score = (metrics["auc_roc"], metrics["accuracy"])
        prior = (-np.inf, -np.inf) if best_metrics is None else (best_metrics["auc_roc"], best_metrics["accuracy"])
        if score > prior:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics, stale = metrics, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite model")
    model.load_state_dict(best_state)
    return model.cpu(), {"seed": seed, "epochs_completed": epoch, "development": best_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache_regional_dsine_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/regional_multi_physics_v5")
    parser.add_argument("--seeds", type=int, nargs="+", default=[59, 61, 67])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0 or not args.seeds:
        raise ValueError("Training settings must be positive")
    cache_dir, output = args.cache_dir.resolve(), args.output_dir.resolve()
    if not cache_dir.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Cache and model paths must remain inside pipeline_40k")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((cache_dir / "provenance.json").read_text(encoding="utf-8"))
    backend = provenance.get("extractor", {}).get("normal_extractor", {}).get("backend")
    if provenance.get("feature_schema") != SCHEMA or backend != "DSINE_v02_kappa":
        raise ValueError("Training requires the recorded regional DSINE cache")
    train_features, train_confidences, train_labels_t = load_cache(cache_dir / "train_features.pt")
    test_features, test_confidences, test_labels_t = load_cache(cache_dir / "test_unseen_features.pt")
    train_manifest = read_manifest(cache_dir / "train_manifest.jsonl")
    test_manifest = read_manifest(cache_dir / "test_unseen_manifest.jsonl")
    if len(train_manifest) != len(train_features) or len(test_manifest) != len(test_features):
        raise ValueError("Cache and manifest counts differ")
    train_labels = train_labels_t.numpy().astype(int)
    test_labels = test_labels_t.numpy().astype(int)
    groups = np.asarray([row["difference_hash_64"] for row in train_manifest])
    test_groups = np.asarray([row["difference_hash_64"] for row in test_manifest])
    train_indices, development_indices, calibration_indices = split_train_development_calibration(train_labels, groups)
    test_indices = np.flatnonzero(~np.isin(test_groups, np.unique(groups)))
    excluded = len(test_groups) - len(test_indices)
    standardizer = fit_regional_standardizer(train_features[train_indices].numpy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    models, runs = [], []
    for seed in args.seeds:
        model, run = train_one(
            seed, train_features[train_indices], train_confidences[train_indices], train_labels_t[train_indices],
            train_features[development_indices], train_confidences[development_indices], train_labels[development_indices],
            standardizer, device, args.epochs, args.patience, args.batch_size,
        )
        models.append(model)
        runs.append(run)

    def ensemble(features: torch.Tensor, confidences: torch.Tensor) -> np.ndarray:
        return np.mean([probabilities(model, features, confidences, standardizer, device) for model in models], axis=0)

    development_probability = ensemble(train_features[development_indices], train_confidences[development_indices])
    calibration_probability = ensemble(train_features[calibration_indices], train_confidences[calibration_indices])
    policy = fit_decision_policy(
        probability_logits(calibration_probability), train_labels[calibration_indices],
        obs_threshold=1.0, abstention_margin=0.10,
    )
    architecture = {"d_model": 48, "pair_dim": 64, "dropout": 0.18}
    payload = {
        "schema_version": 2, "feature_schema": SCHEMA, "regions": provenance["regions"],
        "entity_names": list(ENTITY_NAMES), "entity_dims": list(ENTITY_DIMS),
        "architecture": architecture, "standardizer": standardizer,
        "model_states": [model.state_dict() for model in models], "policy": policy,
        "cache_provenance_sha256": sha256(cache_dir / "provenance.json"),
    }
    torch.save(payload, output / "model.pt")
    write_json(output / "selection.json", {
        "protocol": "difference-hash group-disjoint fit/development/calibration; no unseen-test selection",
        "split_sizes": {"train": len(train_indices), "development": len(development_indices), "calibration": len(calibration_indices)},
        "training_runs": runs,
        "development": evaluate_predictions(train_labels[development_indices], development_probability, 0.5),
    })
    write_json(output / "policy.json", policy)
    test_probability = ensemble(test_features[test_indices], test_confidences[test_indices])
    calibrated = apply_policy_calibration(probability_logits(test_probability), policy)
    confidence_summary = test_confidences[test_indices].mean(dim=1).numpy()
    report = {
        "protocol": "final internal unseen test; train-overlapping difference hashes excluded",
        "test_samples": len(test_indices), "excluded_train_overlap_test_rows": excluded,
        "binary": evaluate_predictions(test_labels[test_indices], calibrated, policy["decision_threshold"]),
        "raw_threshold_0_5": evaluate_predictions(test_labels[test_indices], test_probability, 0.5),
        "selective": evaluate_selective_abstention(
            test_labels[test_indices], calibrated, confidence_summary,
            policy["tau_low"], policy["tau_high"], policy["obs_threshold"],
        ),
    }
    write_json(output / "evaluation.json", report)
    write_json(output / "metadata.json", {
        "device": str(device), "seeds": args.seeds, "training_seconds": time.perf_counter() - started,
        "train_cache_sha256": sha256(cache_dir / "train_features.pt"),
        "test_cache_sha256": sha256(cache_dir / "test_unseen_features.pt"),
        "cache_provenance_sha256": sha256(cache_dir / "provenance.json"),
        "extractor": provenance["extractor"],
        "limitations": [
            "Internal test performance does not establish unseen-generator generalization.",
            "Synthbuster remains a frozen external benchmark and is not used for selection.",
        ],
    })
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
