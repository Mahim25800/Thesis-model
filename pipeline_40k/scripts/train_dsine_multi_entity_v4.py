"""Train and evaluate the DSINE-regenerated multi-entity detector.

The raw-image DSINE cache is the only feature source.  Development and
calibration partitions are group-disjoint by image difference hash; the
regenerated unseen-test cache is never used for fitting or model selection.
"""

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
from src.models.multi_entity import (
    ENTITY_DIMS,
    ENTITY_NAMES,
    MultiEntityConsistencyHead,
    fit_feature_standardizer,
    multi_entity_loss,
    standardize_features,
)
from src.models.improved import probability_logits
from src.utils.decision_policy import apply_policy_calibration, fit_decision_policy
from src.utils.metrics import evaluate_predictions


SCHEMA = "physics_cache_dsine_v1_14_features_4_confidences"


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
        raise ValueError(f"Manifest is missing ordered source positions: {path}")
    return rows


def load_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    features, confidences, labels = cache["features"].float(), cache["confidences"].float(), cache["labels"].float()
    if features.ndim != 2 or features.shape[1] != sum(ENTITY_DIMS) or confidences.shape != (len(features), len(ENTITY_DIMS)):
        raise ValueError(f"Unexpected DSINE cache schema: {path}")
    if labels.shape != (len(features),) or not torch.isfinite(features).all() or not torch.isfinite(confidences).all():
        raise ValueError(f"Non-finite or misaligned DSINE cache: {path}")
    return features, confidences, labels


def split_train_development_calibration(labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create about 71/14/14 percent group-disjoint fit/dev/calibration sets."""
    all_indices = np.arange(len(labels))
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=20260916)
    remaining, calibration = next(outer.split(all_indices, labels, groups))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=20260917)
    train_relative, development_relative = next(inner.split(
        all_indices[remaining], labels[remaining], groups[remaining]
    ))
    train, development = remaining[train_relative], remaining[development_relative]
    partitions = (train, development, calibration)
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            if set(groups[partitions[left]]) & set(groups[partitions[right]]):
                raise RuntimeError("Difference-hash groups leaked across fit partitions")
    return train, development, calibration


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probabilities(model: MultiEntityConsistencyHead, features: torch.Tensor, confidences: torch.Tensor,
                  standardizer: dict, device: torch.device) -> np.ndarray:
    model.to(device).eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            values = standardize_features(features[start:start + 1024].to(device), standardizer)
            result.append(torch.sigmoid(model(values, confidences[start:start + 1024].to(device))).cpu().numpy())
    return np.concatenate(result)


def train_one(seed: int, features: torch.Tensor, confidences: torch.Tensor, labels: torch.Tensor,
              development_features: torch.Tensor, development_confidences: torch.Tensor,
              development_labels: np.ndarray, standardizer: dict, device: torch.device,
              epochs: int, patience: int, batch_size: int) -> tuple[MultiEntityConsistencyHead, dict]:
    seed_everything(seed)
    model = MultiEntityConsistencyHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    values = standardize_features(features, standardizer)
    loader = DataLoader(
        TensorDataset(values, confidences, labels), batch_size=batch_size, shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_metrics, stale = None, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_values, batch_confidences, batch_labels in loader:
            batch_values = batch_values.to(device)
            batch_confidences = batch_confidences.to(device)
            batch_labels = batch_labels.to(device)
            available = torch.rand_like(batch_confidences) >= 0.08
            output = model(batch_values, batch_confidences * available, return_details=True)
            loss = multi_entity_loss(output, batch_labels)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        development_probability = probabilities(model, development_features, development_confidences, standardizer, device)
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
    model.load_state_dict(best_state)
    return model.cpu(), {"seed": seed, "epochs_completed": epoch, "development": best_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache_regenerated_dsine_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/dsine_multi_entity_v4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 47, 53])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0 or not args.seeds:
        raise ValueError("epochs, patience, batch size, and at least one seed must be positive")
    cache_dir, output = args.cache_dir.resolve(), args.output_dir.resolve()
    if not cache_dir.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Cache and output paths must remain inside pipeline_40k")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory already contains a run; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((cache_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("feature_schema") != SCHEMA or provenance.get("normal_extractor", {}).get("backend") != "DSINE_v02_kappa":
        raise ValueError("Training requires the recorded DSINE regenerated cache")
    train_features, train_confidences, train_labels_t = load_cache(cache_dir / "train_features.pt")
    test_features, test_confidences, test_labels_t = load_cache(cache_dir / "test_unseen_features.pt")
    train_manifest, test_manifest = read_manifest(cache_dir / "train_manifest.jsonl"), read_manifest(cache_dir / "test_unseen_manifest.jsonl")
    if len(train_manifest) != len(train_features) or len(test_manifest) != len(test_features):
        raise ValueError("Cache and manifest sample counts differ")
    train_labels = train_labels_t.numpy().astype(int)
    test_labels = test_labels_t.numpy().astype(int)
    if not np.all((train_labels == 0) | (train_labels == 1)) or not np.all((test_labels == 0) | (test_labels == 1)):
        raise ValueError("Expected binary cache labels")
    groups = np.asarray([row["difference_hash_64"] for row in train_manifest])
    test_groups = np.asarray([row["difference_hash_64"] for row in test_manifest])
    train_indices, development_indices, calibration_indices = split_train_development_calibration(train_labels, groups)
    overlapping_test_rows = np.flatnonzero(np.isin(test_groups, np.unique(groups)))
    test_indices = np.flatnonzero(~np.isin(test_groups, np.unique(groups)))
    if not len(test_indices):
        raise RuntimeError("All unseen-test samples overlap train difference-hash groups")

    standardizer = fit_feature_standardizer(train_features[train_indices].numpy())
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
        probability_logits(calibration_probability), train_labels[calibration_indices], obs_threshold=1.0, abstention_margin=0.10
    )
    write_json(output / "selection.json", {
        "protocol": "group-disjoint train/development/calibration split; untouched regenerated unseen test",
        "entity_names": list(ENTITY_NAMES), "entity_dims": list(ENTITY_DIMS),
        "split_sizes": {"train": int(len(train_indices)), "development": int(len(development_indices)), "calibration": int(len(calibration_indices))},
        "training_runs": runs,
        "development_entity_only": evaluate_predictions(train_labels[development_indices], development_probability, 0.5),
    })
    write_json(output / "policy.json", policy)
    payload = {
        "schema_version": 1, "feature_schema": SCHEMA,
        "entity_names": list(ENTITY_NAMES), "entity_dims": list(ENTITY_DIMS),
        "architecture": {"d_model": 48, "pair_dim": 64, "dropout": 0.15},
        "standardizer": standardizer, "model_states": [model.state_dict() for model in models], "policy": policy,
        "cache_provenance_sha256": sha256(cache_dir / "provenance.json"),
    }
    torch.save(payload, output / "model.pt")

    test_probability = ensemble(test_features[test_indices], test_confidences[test_indices])
    calibrated = apply_policy_calibration(probability_logits(test_probability), policy)
    test_report = {
        "protocol": "final regenerated unseen test; all train-overlapping difference-hash groups excluded",
        "test_samples": int(len(test_indices)), "excluded_train_overlap_test_rows": int(len(overlapping_test_rows)),
        "binary": evaluate_predictions(test_labels[test_indices], calibrated, policy["decision_threshold"]),
        "raw_threshold_0_5": evaluate_predictions(test_labels[test_indices], test_probability, 0.5),
        "selective": evaluate_selective_abstention(
            test_labels[test_indices], calibrated, test_confidences[test_indices].numpy(),
            policy["tau_low"], policy["tau_high"], policy["obs_threshold"],
        ),
    }
    write_json(output / "evaluation.json", test_report)
    write_json(output / "metadata.json", {
        "device": str(device), "seeds": args.seeds, "training_seconds": time.perf_counter() - started,
        "train_cache_sha256": sha256(cache_dir / "train_features.pt"),
        "test_cache_sha256": sha256(cache_dir / "test_unseen_features.pt"),
        "cache_provenance_sha256": sha256(cache_dir / "provenance.json"),
        "normal_extractor": provenance["normal_extractor"],
        "limitations": [
            "This is an internal regenerated unseen split, not an external unseen-generator benchmark.",
            "Difference hash is a near-duplicate safeguard, not a proof of semantic independence.",
        ],
    })
    print(json.dumps(test_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
