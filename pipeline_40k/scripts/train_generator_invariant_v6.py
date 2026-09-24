"""Train regional physics with source-domain adversarial invariance.

Source and held-out generator domains are configurable. The held-out domain is
never optimized against and is used only for development/model selection. Final
external benchmarks are not read by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_scaled import evaluate_selective_abstention
from scripts.train_regional_multi_physics_v5 import (
    SCHEMA,
    augment_regions,
    probabilities,
    read_manifest,
    sha256,
    split_train_development_calibration,
    write_json,
)
from src.models.improved import probability_logits
from src.models.multi_entity import ENTITY_DIMS, ENTITY_NAMES
from src.models.regional_multi_physics import (
    RegionalMultiPhysicsHead,
    fit_regional_standardizer,
    regional_multi_physics_loss,
    standardize_regional,
)
from src.utils.decision_policy import apply_policy_calibration, fit_decision_policy
from src.utils.metrics import evaluate_predictions


SOURCE_DOMAINS = ("adm", "biggan", "vqdm")
HELD_OUT_DOMAIN = "wukong"


def load_configured_cache(path: Path, entity_dims: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a regional cache whose entity layout is recorded by the caller."""
    cache = torch.load(path, map_location="cpu", weights_only=True)
    features = cache["features"].float()
    confidences = cache["confidences"].float()
    labels = cache["labels"].float()
    expected_features = (5, sum(entity_dims))
    expected_confidences = (len(features), 5, len(entity_dims))
    if features.ndim != 3 or features.shape[1:] != expected_features:
        raise ValueError(f"Unexpected regional feature schema: {path}")
    if confidences.shape != expected_confidences or labels.shape != (len(features),):
        raise ValueError(f"Misaligned regional cache: {path}")
    if not torch.isfinite(features).all() or not torch.isfinite(confidences).all():
        raise ValueError(f"Non-finite regional cache: {path}")
    if ((confidences < 0) | (confidences > 1)).any():
        raise ValueError(f"Confidence values outside [0, 1]: {path}")
    return features, confidences, labels


def load_training_augmentation(
    cache: Path, name: str, source_labels: torch.Tensor, entity_dims: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a complete raw-image augmentation cache aligned to its source rows.

    Augmented rows are appended only to fitting subsets. The caller must retain
    original rows for calibration and every development/test evaluation.
    """
    provenance = json.loads((cache / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("feature_schema") != SCHEMA:
        raise ValueError("Augmentation cache has an incompatible feature schema")
    record = provenance.get("train") if name == "train" else provenance.get("domains", {}).get(name)
    if not record or not record.get("complete"):
        raise ValueError(f"Augmentation cache is incomplete for {name}")
    features, confidences, labels = load_configured_cache(cache / f"{name}_features.pt", entity_dims)
    if labels.shape != source_labels.shape or not torch.equal(labels, source_labels.float()):
        raise ValueError(f"Augmentation labels do not align with source cache for {name}")
    if int(record.get("samples", -1)) != len(source_labels) or int(record.get("source_samples", -1)) != len(source_labels):
        raise ValueError(f"Augmentation provenance is not row-complete for {name}")
    return features, confidences


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float):
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, domains: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, domains)
        )

    def forward(self, embedding: torch.Tensor, strength: float) -> torch.Tensor:
        return self.layers(_GradientReverse.apply(embedding, strength))


def split_source(labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train, calibration = next(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=20260918
    ).split(indices, labels, groups))
    if set(groups[train]) & set(groups[calibration]):
        raise RuntimeError("Source-domain groups leaked into calibration")
    return train, calibration


def balanced_unique_development(labels: np.ndarray, groups: np.ndarray, forbidden: set[str]):
    eligible = np.flatnonzero(~np.isin(groups, np.asarray(sorted(forbidden))))
    real = eligible[labels[eligible] == 0]
    fake = eligible[labels[eligible] == 1]
    count = min(len(real), len(fake))
    if count < 100:
        raise RuntimeError("Too few group-unique held-domain real/fake samples for development")
    return np.concatenate([real[:count], fake[:count]])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(
    seed: int,
    train_features: torch.Tensor,
    train_confidences: torch.Tensor,
    train_labels: torch.Tensor,
    train_domains: torch.Tensor,
    internal_dev: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    external_dev: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    standardizer: dict,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    domain_loss_weight: float,
    use_entity_attention: bool,
    entity_loss_weight: float,
    entity_dims: tuple[int, ...],
    source_domains: tuple[str, ...],
) -> tuple[RegionalMultiPhysicsHead, dict]:
    seed_everything(seed)
    architecture = {
        "d_model": 48, "pair_dim": 64, "dropout": 0.18,
        "use_entity_attention": use_entity_attention,
        "entity_dims": entity_dims,
    }
    model = RegionalMultiPhysicsHead(**architecture).to(device)
    adversary = DomainClassifier(128, len(source_domains) + 1).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(adversary.parameters()), lr=5e-4, weight_decay=3e-3
    )
    counts = torch.bincount(train_domains, minlength=len(source_domains) + 1).float()
    sample_weights = (1.0 / counts.clamp_min(1))[train_domains]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_domains), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_values = standardize_regional(train_features, standardizer)
    loader = DataLoader(
        TensorDataset(train_values, train_confidences, train_labels, train_domains),
        batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=device.type == "cuda",
    )
    best_state, best_metrics, stale = None, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        adversary.train()
        progress = (epoch - 1) / max(epochs - 1, 1)
        grl_strength = 0.10 * (2.0 / (1.0 + math.exp(-8.0 * progress)) - 1.0)
        for values, confidences, labels, domains in loader:
            values = values.to(device, non_blocking=True)
            confidences = confidences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domains = domains.to(device, non_blocking=True)
            values, confidences = augment_regions(values, confidences)
            output = model(values, confidences, return_details=True)
            classification = regional_multi_physics_loss(
                output, labels, entity_weight=entity_loss_weight
            )["total"]
            domain_loss = F.cross_entropy(adversary(output["embedding"], grl_strength), domains)
            loss = classification + domain_loss_weight * domain_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adversary.parameters()), 2.0)
            optimizer.step()
        internal_probability = probabilities(model, internal_dev[0], internal_dev[1], standardizer, device)
        external_probability = probabilities(model, external_dev[0], external_dev[1], standardizer, device)
        internal_metrics = evaluate_predictions(internal_dev[2], internal_probability, 0.5)
        external_metrics = evaluate_predictions(external_dev[2], external_probability, 0.5)
        worst_auc = min(internal_metrics["auc_roc"], external_metrics["auc_roc"])
        score = (worst_auc, 0.5 * (internal_metrics["auc_roc"] + external_metrics["auc_roc"]))
        prior = (-np.inf, -np.inf) if best_metrics is None else tuple(best_metrics["selection_score"])
        if score > prior:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "selection_score": list(score), "internal_development": internal_metrics,
                "held_out_generator_development": external_metrics,
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training failed to produce a selected model")
    model.load_state_dict(best_state)
    return model.cpu(), {"seed": seed, "epochs_completed": epoch, **best_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, default=ROOT / "data/cache_regional_dsine_v2")
    parser.add_argument("--genimage-cache", type=Path, default=ROOT / "data/cache_genimage_regional_v2")
    parser.add_argument("--base-augmentation-cache", type=Path, default=None)
    parser.add_argument("--genimage-augmentation-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/generator_invariant_v6")
    parser.add_argument("--seeds", type=int, nargs="+", default=[71, 73, 79])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--domain-loss-weight", type=float, default=0.10)
    parser.add_argument("--learned-entity-attention", action="store_true")
    parser.add_argument("--entity-loss-weight", type=float, default=0.03)
    parser.add_argument("--disabled-entities", nargs="*", default=[])
    parser.add_argument("--source-domains", nargs="+", default=list(SOURCE_DOMAINS))
    parser.add_argument("--held-out-domain", default=HELD_OUT_DOMAIN)
    parser.add_argument("--feature-schema", default=SCHEMA)
    parser.add_argument("--entity-dims", type=int, nargs="+", default=list(ENTITY_DIMS))
    parser.add_argument("--entity-names", nargs="+", default=list(ENTITY_NAMES))
    args = parser.parse_args()
    base_cache, genimage_cache, output = args.base_cache.resolve(), args.genimage_cache.resolve(), args.output_dir.resolve()
    augmentation_paths = (args.base_augmentation_cache, args.genimage_augmentation_cache)
    if any(path is None for path in augmentation_paths) and any(path is not None for path in augmentation_paths):
        raise ValueError("Provide both augmentation cache paths or neither")
    base_augmentation_cache = None if args.base_augmentation_cache is None else args.base_augmentation_cache.resolve()
    genimage_augmentation_cache = None if args.genimage_augmentation_cache is None else args.genimage_augmentation_cache.resolve()
    all_paths = (base_cache, genimage_cache, output) + (() if base_augmentation_cache is None else (base_augmentation_cache, genimage_augmentation_cache))
    if not all(path.is_relative_to(ROOT) for path in all_paths):
        raise ValueError("All paths must remain inside pipeline_40k")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty model directory")
    if not 0.0 <= args.domain_loss_weight <= 1.0:
        raise ValueError("domain-loss-weight must lie in [0, 1]")
    if not 0.0 <= args.entity_loss_weight <= 1.0:
        raise ValueError("entity-loss-weight must lie in [0, 1]")
    entity_dims = tuple(args.entity_dims)
    entity_names = tuple(args.entity_names)
    source_domains = tuple(args.source_domains)
    held_out_domain = str(args.held_out_domain)
    if not entity_dims or any(value <= 0 for value in entity_dims) or len(entity_dims) != len(entity_names):
        raise ValueError("entity-dims and entity-names must describe positive, aligned entities")
    if len(set(entity_names)) != len(entity_names) or any(name not in entity_names for name in args.disabled_entities):
        raise ValueError("disabled-entities must be unique names from entity-names")
    available_domains = {"adm", "biggan", "vqdm", "wukong"}
    if len(source_domains) < 2 or len(set(source_domains)) != len(source_domains) or held_out_domain in source_domains:
        raise ValueError("source-domains must be unique and exclude the held-out domain")
    if (set(source_domains) | {held_out_domain}) - available_domains:
        raise ValueError("source-domains and held-out-domain must be GenImage domains")
    output.mkdir(parents=True, exist_ok=True)

    base_provenance = json.loads((base_cache / "provenance.json").read_text(encoding="utf-8"))
    genimage_provenance = json.loads((genimage_cache / "provenance.json").read_text(encoding="utf-8"))
    if base_provenance.get("feature_schema") != args.feature_schema or genimage_provenance.get("feature_schema") != args.feature_schema:
        raise ValueError("Both caches must use the requested feature schema")
    if tuple(base_provenance.get("entity_dims", ENTITY_DIMS)) != entity_dims \
            or tuple(genimage_provenance.get("entity_dims", ENTITY_DIMS)) != entity_dims:
        raise ValueError("Cache provenance entity dimensions do not match the requested layout")
    base_features, base_confidences, base_labels_t = load_configured_cache(base_cache / "train_features.pt", entity_dims)
    test_features, test_confidences, test_labels_t = load_configured_cache(base_cache / "test_unseen_features.pt", entity_dims)
    base_manifest = read_manifest(base_cache / "train_manifest.jsonl")
    test_manifest = read_manifest(base_cache / "test_unseen_manifest.jsonl")
    base_labels = base_labels_t.numpy().astype(int)
    base_groups = np.asarray([row["difference_hash_64"] for row in base_manifest])
    base_train, base_development, base_calibration = split_train_development_calibration(base_labels, base_groups)

    if base_augmentation_cache is None:
        base_augmented_features, base_augmented_confidences = None, None
    else:
        base_augmented_features, base_augmented_confidences = load_training_augmentation(
            base_augmentation_cache, "train", base_labels_t, entity_dims
        )
    train_feature_parts = [base_features[base_train]]
    train_confidence_parts = [base_confidences[base_train]]
    train_label_parts = [base_labels_t[base_train]]
    train_domain_parts = [torch.zeros(len(base_train), dtype=torch.long)]
    if base_augmented_features is not None:
        train_feature_parts.append(base_augmented_features[base_train])
        train_confidence_parts.append(base_augmented_confidences[base_train])
        train_label_parts.append(base_labels_t[base_train])
        train_domain_parts.append(torch.zeros(len(base_train), dtype=torch.long))
    calibration_feature_parts = [base_features[base_calibration]]
    calibration_confidence_parts = [base_confidences[base_calibration]]
    calibration_label_parts = [base_labels_t[base_calibration]]
    forbidden_groups = set(base_groups[base_train].tolist())
    source_split_sizes = {}
    for domain_index, domain in enumerate(source_domains, start=1):
        features, confidences, labels_t = load_configured_cache(genimage_cache / f"{domain}_features.pt", entity_dims)
        if genimage_augmentation_cache is None:
            augmented_features, augmented_confidences = None, None
        else:
            augmented_features, augmented_confidences = load_training_augmentation(
                genimage_augmentation_cache, domain, labels_t, entity_dims
            )
        manifest = read_manifest(genimage_cache / f"{domain}_manifest.jsonl")
        labels = labels_t.numpy().astype(int)
        groups = np.asarray([row["difference_hash_64"] for row in manifest])
        source_train, source_calibration = split_source(labels, groups)
        train_feature_parts.append(features[source_train])
        train_confidence_parts.append(confidences[source_train])
        train_label_parts.append(labels_t[source_train])
        train_domain_parts.append(torch.full((len(source_train),), domain_index, dtype=torch.long))
        if augmented_features is not None:
            train_feature_parts.append(augmented_features[source_train])
            train_confidence_parts.append(augmented_confidences[source_train])
            train_label_parts.append(labels_t[source_train])
            train_domain_parts.append(torch.full((len(source_train),), domain_index, dtype=torch.long))
        calibration_feature_parts.append(features[source_calibration])
        calibration_confidence_parts.append(confidences[source_calibration])
        calibration_label_parts.append(labels_t[source_calibration])
        forbidden_groups.update(groups[source_train].tolist())
        source_split_sizes[domain] = {
            "train_original": len(source_train), "train_augmented": 0 if augmented_features is None else len(source_train),
            "calibration_original": len(source_calibration),
        }

    held_features, held_confidences, held_labels_t = load_configured_cache(genimage_cache / f"{held_out_domain}_features.pt", entity_dims)
    held_manifest = read_manifest(genimage_cache / f"{held_out_domain}_manifest.jsonl")
    held_labels = held_labels_t.numpy().astype(int)
    held_groups = np.asarray([row["difference_hash_64"] for row in held_manifest])
    held_indices = balanced_unique_development(held_labels, held_groups, forbidden_groups)

    train_features = torch.cat(train_feature_parts)
    train_confidences = torch.cat(train_confidence_parts)
    train_labels_t = torch.cat(train_label_parts)
    train_domains = torch.cat(train_domain_parts)
    calibration_features = torch.cat(calibration_feature_parts)
    calibration_confidences = torch.cat(calibration_confidence_parts)
    calibration_labels_t = torch.cat(calibration_label_parts)
    disabled_indices = [entity_names.index(name) for name in args.disabled_entities]

    def mask_disabled(confidences: torch.Tensor) -> torch.Tensor:
        if not disabled_indices:
            return confidences
        result = confidences.clone()
        result[:, :, disabled_indices] = 0.0
        return result

    train_confidences = mask_disabled(train_confidences)
    calibration_confidences = mask_disabled(calibration_confidences)
    standardizer = fit_regional_standardizer(train_features.numpy(), entity_dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    internal_dev = (
        base_features[base_development], mask_disabled(base_confidences[base_development]),
        base_labels[base_development]
    )
    external_dev = (
        held_features[held_indices], mask_disabled(held_confidences[held_indices]), held_labels[held_indices]
    )
    models, runs = [], []
    for seed in args.seeds:
        model, run = train_one(
            seed, train_features, train_confidences, train_labels_t, train_domains,
            internal_dev, external_dev, standardizer, device, args.epochs, args.patience, args.batch_size,
            args.domain_loss_weight,
            args.learned_entity_attention,
            args.entity_loss_weight,
            entity_dims, source_domains,
        )
        models.append(model)
        runs.append(run)

    def ensemble(features: torch.Tensor, confidences: torch.Tensor) -> np.ndarray:
        return np.mean([probabilities(model, features, confidences, standardizer, device) for model in models], axis=0)

    calibration_probability = ensemble(calibration_features, calibration_confidences)
    policy = fit_decision_policy(
        probability_logits(calibration_probability), calibration_labels_t.numpy().astype(int),
        obs_threshold=1.0, abstention_margin=0.10,
    )
    architecture = {
        "d_model": 48, "pair_dim": 64, "dropout": 0.18,
        "use_entity_attention": args.learned_entity_attention,
        "entity_dims": entity_dims,
    }
    torch.save({
        "schema_version": 4, "feature_schema": args.feature_schema, "regions": base_provenance["regions"],
        "entity_names": list(entity_names), "entity_dims": list(entity_dims),
        "architecture": architecture, "standardizer": standardizer,
        "model_states": [model.state_dict() for model in models], "policy": policy,
        "training_domains": ["diffusiondb_coco", *source_domains],
        "held_out_development_domain": held_out_domain,
        "domain_loss_weight": args.domain_loss_weight,
        "entity_loss_weight": args.entity_loss_weight,
        "disabled_entities": list(args.disabled_entities),
        "raw_image_augmentation": base_augmentation_cache is not None,
    }, output / "model.pt")
    write_json(output / "selection.json", {
        "protocol": "source-domain adversarial training; configured GenImage generator held out for model selection",
        "training_samples": len(train_features),
        "domain_loss_weight": args.domain_loss_weight,
        "entity_loss_weight": args.entity_loss_weight,
        "disabled_entities": list(args.disabled_entities),
        "raw_image_augmentation": base_augmentation_cache is not None,
        "domain_counts": {str(index): int((train_domains == index).sum()) for index in range(len(source_domains) + 1)},
        "source_domains": list(source_domains),
        "held_out_development_domain": held_out_domain,
        "source_split_sizes": source_split_sizes,
        "internal_development_samples": len(base_development),
        "held_domain_unique_balanced_development_samples": len(held_indices),
        "training_runs": runs,
        "ensemble_internal_development": evaluate_predictions(
            internal_dev[2], ensemble(internal_dev[0], internal_dev[1]), 0.5
        ),
        "ensemble_held_out_generator_development": evaluate_predictions(
            external_dev[2], ensemble(external_dev[0], external_dev[1]), 0.5
        ),
    })
    write_json(output / "policy.json", policy)

    test_groups = np.asarray([row["difference_hash_64"] for row in test_manifest])
    test_indices = np.flatnonzero(~np.isin(test_groups, np.unique(base_groups)))
    test_labels = test_labels_t.numpy().astype(int)
    masked_test_confidences = mask_disabled(test_confidences[test_indices])
    test_probability = ensemble(test_features[test_indices], masked_test_confidences)
    calibrated = apply_policy_calibration(probability_logits(test_probability), policy)
    report = {
        "protocol": "final internal unseen test after held-generator-based model selection",
        "test_samples": len(test_indices),
        "binary": evaluate_predictions(test_labels[test_indices], calibrated, policy["decision_threshold"]),
        "raw_threshold_0_5": evaluate_predictions(test_labels[test_indices], test_probability, 0.5),
        "selective": evaluate_selective_abstention(
            test_labels[test_indices], calibrated, masked_test_confidences.mean(dim=1).numpy(),
            policy["tau_low"], policy["tau_high"], policy["obs_threshold"],
        ),
    }
    write_json(output / "evaluation_internal.json", report)
    write_json(output / "metadata.json", {
        "device": str(device), "seeds": args.seeds, "training_seconds": time.perf_counter() - started,
        "domain_loss_weight": args.domain_loss_weight,
        "entity_loss_weight": args.entity_loss_weight,
        "disabled_entities": list(args.disabled_entities),
        "base_cache_provenance_sha256": sha256(base_cache / "provenance.json"),
        "genimage_cache_provenance_sha256": sha256(genimage_cache / "provenance.json"),
        "base_augmentation_cache_provenance_sha256": None if base_augmentation_cache is None else sha256(base_augmentation_cache / "provenance.json"),
        "genimage_augmentation_cache_provenance_sha256": None if genimage_augmentation_cache is None else sha256(genimage_augmentation_cache / "provenance.json"),
        "final_external_test": "Synthbuster remains unread and frozen until this artifact is complete",
    })
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
