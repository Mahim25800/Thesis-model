"""Audit the frozen v14 ensemble for entity, regional, and branch reliance.

This script is descriptive only. It never fits a model, threshold, calibration,
or ensemble weight. Synthbuster metrics are reported only because v14 was frozen
before that benchmark was opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_scaled import evaluate_selective_abstention
from scripts.select_physics_ensemble_v7 import branch_probability, load_branch
from scripts.train_regional_multi_physics_v5 import load_cache, read_manifest
from src.models.improved import probability_logits
from src.utils.decision_policy import apply_policy_calibration
from src.utils.metrics import evaluate_predictions


SCHEMA = "regional_physics_dsine_v2_5x14_features_5x4_confidences"
ENTITY_NAMES = ("illumination", "specular_optics", "surface_normals", "chromatic_shadows")
REGION_NAMES = ("global", "top_left", "top_right", "bottom_left", "bottom_right")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ensemble(path: Path, device: torch.device):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("model_type") != "physics_probability_ensemble":
        raise ValueError("Expected a physics probability ensemble artifact")
    if artifact.get("feature_schema") != SCHEMA:
        raise ValueError("Unexpected v14 feature schema")
    branches = []
    for specification in artifact.get("branches", []):
        branch_path = (ROOT / specification["path"]).resolve()
        if not branch_path.is_relative_to(ROOT) or sha256(branch_path) != specification["sha256"]:
            raise ValueError(f"Frozen branch hash mismatch: {branch_path}")
        branch_artifact, models = load_branch(branch_path, device)
        branches.append((specification, branch_artifact, models))
    if not branches or not any(float(item[0]["weight"]) > 0 for item in branches):
        raise ValueError("Ensemble has no active branches")
    return artifact, branches


def predict_raw(branches, features: torch.Tensor, confidences: torch.Tensor, device: torch.device):
    result = np.zeros(len(features), dtype=np.float64)
    branch_rows = {}
    for specification, artifact, models in branches:
        probability = branch_probability(artifact, models, features, confidences, device)
        branch_rows[specification["name"]] = probability
        result += float(specification["weight"]) * probability
    return result, branch_rows


def prediction_set(ensemble, branches, features, confidences, labels, device):
    raw, branch_rows = predict_raw(branches, features, confidences, device)
    calibrated = apply_policy_calibration(probability_logits(raw), ensemble["policy"])
    return {
        "raw": raw,
        "calibrated": calibrated,
        "branch_rows": branch_rows,
        "binary": evaluate_predictions(labels, calibrated, ensemble["policy"]["decision_threshold"]),
        "selective": evaluate_selective_abstention(
            labels, calibrated, confidences.mean(dim=1).numpy(), ensemble["policy"]["tau_low"],
            ensemble["policy"]["tau_high"], ensemble["policy"]["obs_threshold"],
        ),
    }


def ablate_confidence(confidences: torch.Tensor, *, entity: int | None = None, region: int | None = None):
    result = confidences.clone()
    if entity is not None:
        result[:, :, entity] = 0.0
    if region is not None:
        result[:, region, :] = 0.0
    return result


def bootstrap_auc_delta(labels: np.ndarray, full: np.ndarray, ablated: np.ndarray, seed: int = 20260924, repeats: int = 1000):
    labels = np.asarray(labels, dtype=int)
    real, fake = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    if not len(real) or not len(fake):
        return {"mean_auc_delta": None, "ci95": [None, None], "repeats": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = np.concatenate([rng.choice(real, len(real), replace=True), rng.choice(fake, len(fake), replace=True)])
        delta = evaluate_predictions(labels[indices], full[indices], 0.5)["auc_roc"] - evaluate_predictions(labels[indices], ablated[indices], 0.5)["auc_roc"]
        values.append(float(delta))
    return {
        "mean_auc_delta": float(np.mean(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "repeats": repeats,
        "resampling": "class-stratified paired bootstrap",
    }


def audit_dataset(name, ensemble, branches, features, confidences, labels, device, repeats):
    labels = np.asarray(labels, dtype=int)
    full = prediction_set(ensemble, branches, features, confidences, labels, device)
    ablations = {}
    for index, entity in enumerate(ENTITY_NAMES):
        masked = ablate_confidence(confidences, entity=index)
        prediction = prediction_set(ensemble, branches, features, masked, labels, device)
        ablations[f"without_{entity}"] = {
            "binary": prediction["binary"],
            "auc_delta_from_full": float(prediction["binary"]["auc_roc"] - full["binary"]["auc_roc"]),
            "bootstrap_full_minus_ablated": bootstrap_auc_delta(labels, full["calibrated"], prediction["calibrated"], repeats=repeats),
        }
    global_only = confidences.clone()
    global_only[:, 1:, :] = 0.0
    prediction = prediction_set(ensemble, branches, features, global_only, labels, device)
    ablations["global_only"] = {
        "binary": prediction["binary"],
        "auc_delta_from_full": float(prediction["binary"]["auc_roc"] - full["binary"]["auc_roc"]),
        "bootstrap_full_minus_ablated": bootstrap_auc_delta(labels, full["calibrated"], prediction["calibrated"], repeats=repeats),
    }
    for index, region in enumerate(REGION_NAMES[1:], start=1):
        masked = ablate_confidence(confidences, region=index)
        prediction = prediction_set(ensemble, branches, features, masked, labels, device)
        ablations[f"without_{region}"] = {
            "binary": prediction["binary"],
            "auc_delta_from_full": float(prediction["binary"]["auc_roc"] - full["binary"]["auc_roc"]),
            "bootstrap_full_minus_ablated": bootstrap_auc_delta(labels, full["calibrated"], prediction["calibrated"], repeats=repeats),
        }
    active = {name: rows for name, rows in full["branch_rows"].items() if next(float(item[0]["weight"]) for item in branches if item[0]["name"] == name) > 0.0}
    names = sorted(active)
    correlations = {
        f"{left}__{right}": float(np.corrcoef(active[left], active[right])[0, 1])
        for position, left in enumerate(names) for right in names[position + 1:]
    }
    return {
        "dataset": name,
        "samples": int(len(labels)),
        "full": {"binary": full["binary"], "selective": full["selective"]},
        "ablations": ablations,
        "branch_probability_correlations": correlations,
    }


def internal_unseen(cache: Path):
    train_features, _, train_labels = load_cache(cache / "train_features.pt")
    test_features, test_confidences, test_labels = load_cache(cache / "test_unseen_features.pt")
    train_manifest = read_manifest(cache / "train_manifest.jsonl")
    test_manifest = read_manifest(cache / "test_unseen_manifest.jsonl")
    groups = {row["difference_hash_64"] for row in train_manifest}
    indices = np.asarray([index for index, row in enumerate(test_manifest) if row["difference_hash_64"] not in groups], dtype=int)
    if not len(indices):
        raise ValueError("No group-disjoint internal unseen samples")
    return test_features[indices], test_confidences[indices], test_labels.numpy().astype(int)[indices]


def synthbuster_sets(cache: Path):
    real = torch.load(cache / "raise_real.pt", map_location="cpu", weights_only=True)
    rows = {}
    for path in sorted(cache.glob("*.pt")):
        if path.stem == "raise_real":
            continue
        fake = torch.load(path, map_location="cpu", weights_only=True)
        count = min(len(real["features"]), len(fake["features"]))
        rows[path.stem] = (
            torch.cat([real["features"][:count], fake["features"][:count]]).float(),
            torch.cat([real["confidences"][:count], fake["confidences"][:count]]).float(),
            np.concatenate([np.zeros(count, dtype=int), np.ones(count, dtype=int)]),
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/physics_ensemble_v14/model.pt")
    parser.add_argument("--regional-cache", type=Path, default=ROOT / "data/cache_regional_dsine_v2")
    parser.add_argument("--genimage-cache", type=Path, default=ROOT / "data/cache_genimage_regional_v2")
    parser.add_argument("--synthbuster-cache", type=Path, default=ROOT / "data/cache_synthbuster_regional_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "models/physics_ensemble_v14/robustness_audit.json")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--skip-synthbuster", action="store_true")
    args = parser.parse_args()
    paths = [args.model.resolve(), args.regional_cache.resolve(), args.genimage_cache.resolve(), args.output.resolve()]
    if not args.skip_synthbuster:
        paths.append(args.synthbuster_cache.resolve())
    if not all(path.is_relative_to(ROOT) for path in paths):
        raise ValueError("All paths must remain inside pipeline_40k")
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap-repeats must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble, branches = load_ensemble(args.model.resolve(), device)
    internal_features, internal_confidences, internal_labels = internal_unseen(args.regional_cache.resolve())
    report = {
        "protocol": "descriptive frozen-artifact audit; no model, policy, threshold, or weight fitting",
        "model": str(args.model.resolve().relative_to(ROOT)).replace("\\", "/"),
        "device": str(device),
        "branches": [
            {"name": item[0]["name"], "weight": item[0]["weight"], "path": item[0]["path"], "sha256": item[0]["sha256"]}
            for item in branches
        ],
        "datasets": {
            "internal_unseen": audit_dataset("internal_unseen", ensemble, branches, internal_features, internal_confidences, internal_labels, device, args.bootstrap_repeats),
        },
    }
    wukong = torch.load(args.genimage_cache.resolve() / "wukong_features.pt", map_location="cpu", weights_only=True)
    report["datasets"]["genimage_wukong_development"] = audit_dataset(
        "genimage_wukong_development", ensemble, branches, wukong["features"].float(), wukong["confidences"].float(),
        wukong["labels"].numpy().astype(int), device, args.bootstrap_repeats,
    )
    if not args.skip_synthbuster:
        report["datasets"]["synthbuster"] = {
            name: audit_dataset(name, ensemble, branches, features, confidences, labels, device, args.bootstrap_repeats)
            for name, (features, confidences, labels) in synthbuster_sets(args.synthbuster_cache.resolve()).items()
        }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
