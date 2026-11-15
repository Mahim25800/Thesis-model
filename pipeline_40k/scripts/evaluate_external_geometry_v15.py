"""Post-freeze Synthbuster evaluation for the five-entity physics-geometry model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_projective_geometry_v15 import GEOMETRY_DIM, geometry_for_image
from scripts.evaluate_scaled import evaluate_selective_abstention
from scripts.train_regional_multi_physics_v5 import probabilities
from src.extractors.perspective_vp import PerspectiveVPExtractor
from src.models.improved import probability_logits
from src.models.regional_multi_physics import RegionalMultiPhysicsHead
from src.utils.decision_policy import apply_policy_calibration
from src.utils.metrics import evaluate_predictions


SCHEMA = "regional_physics_geometry_v15_5x22_features_5x5_confidences"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def atomic_torch_save(value: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def source_paths(directory: Path, limit: int | None) -> list[Path]:
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images in {directory}")
    return paths if limit is None else paths[:limit]


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def select_physics_rows(cache_dir: Path, name: str, filenames: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.load(cache_dir / f"{name}.pt", map_location="cpu", weights_only=True)
    manifest = read_manifest(cache_dir / f"{name}.jsonl")
    positions = {row["filename"]: int(row["index"]) for row in manifest}
    missing = set(filenames) - set(positions)
    if missing:
        raise ValueError(f"Physics cache {name} misses selected filenames")
    index = torch.tensor([positions[value] for value in filenames], dtype=torch.long)
    features, confidences = values["features"][index].float(), values["confidences"][index].float()
    if features.shape != (len(filenames), 5, 14) or confidences.shape != (len(filenames), 5, 4):
        raise ValueError(f"External physics cache has an invalid schema: {name}")
    return features, confidences


def combined_cache(name: str, paths: list[Path], source_features: torch.Tensor,
                   source_confidences: torch.Tensor, cache_dir: Path) -> dict:
    output = cache_dir / f"{name}.pt"
    names = np.asarray([path.name for path in paths])
    if output.is_file():
        cached = torch.load(output, map_location="cpu", weights_only=True)
        if cached["features"].shape == (len(paths), 5, 22) and cached["confidences"].shape == (len(paths), 5, 5) \
                and np.array_equal(cached["filenames"], names):
            return cached
        raise ValueError(f"Existing geometry cache is incompatible: {output}")
    extractor = PerspectiveVPExtractor()
    geometry, confidence = [], []
    for path in tqdm(paths, desc=f"Synthbuster geometry {name}"):
        value, certainty = geometry_for_image(path, extractor)
        geometry.append(value)
        confidence.append(certainty)
    combined = {
        "features": torch.cat([source_features, torch.stack(geometry)], dim=2),
        "confidences": torch.cat([source_confidences, torch.stack(confidence)], dim=2),
        "filenames": names,
    }
    atomic_torch_save(combined, output)
    return combined


def load_model(path: Path, device: torch.device) -> tuple[dict, list[RegionalMultiPhysicsHead]]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("feature_schema") != SCHEMA or tuple(artifact.get("entity_dims", ())) != (5, 4, 3, 2, GEOMETRY_DIM):
        raise ValueError("Model does not use the five-entity physics-geometry schema")
    models = []
    for state in artifact["model_states"]:
        model = RegionalMultiPhysicsHead(**artifact["architecture"]).to(device).eval()
        model.load_state_dict(state)
        models.append(model)
    return artifact, models


def probability(models: list[RegionalMultiPhysicsHead], features: torch.Tensor, confidences: torch.Tensor,
                standardizer: dict, device: torch.device) -> np.ndarray:
    return np.mean([probabilities(model, features, confidences, standardizer, device) for model in models], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "data/external_synthbuster")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--physics-cache", type=Path, default=ROOT / "data/cache_synthbuster_regional_v2")
    parser.add_argument("--geometry-cache", type=Path, default=ROOT / "data/cache_synthbuster_physics_geometry_v15")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generators", nargs="+", default=None)
    parser.add_argument("--limit-per-source", type=int, default=None)
    args = parser.parse_args()
    benchmark, model_path = args.benchmark_root.resolve(), args.model.resolve()
    physics_cache, geometry_cache, output = args.physics_cache.resolve(), args.geometry_cache.resolve(), args.output_dir.resolve()
    if not all(path.is_relative_to(ROOT) for path in (benchmark, model_path, physics_cache, geometry_cache, output)):
        raise ValueError("All paths must remain inside pipeline_40k")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty output directory")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact, models = load_model(model_path, device)
    real_dir = benchmark / "raise" / "real_RAISE_1k"
    fake_root = benchmark / "synthbuster" / "synthbuster"
    generators = sorted(path.name for path in fake_root.iterdir() if path.is_dir()) if args.generators is None else args.generators
    real_by_name = {path.name: path for path in source_paths(real_dir, None)}
    selected_fake = {name: source_paths(fake_root / name, args.limit_per_source) for name in generators}
    common = {name: sorted(set(real_by_name) & {path.name for path in paths}) for name, paths in selected_fake.items()}
    if any(not values for values in common.values()):
        raise ValueError("A selected generator has no paired RAISE filenames")
    real_names = sorted(set().union(*(set(values) for values in common.values())))
    real_features, real_confidences = select_physics_rows(physics_cache, "raise_real", real_names)
    real = combined_cache("raise_real", [real_by_name[name] for name in real_names], real_features, real_confidences, geometry_cache)
    real_index = {name: index for index, name in enumerate(real_names)}
    report = {
        "protocol": "post-selection frozen evaluation; geometry is masked only for a post-freeze ablation",
        "model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "device": str(device), "generators": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir()
    geometry_index = artifact["entity_names"].index("projective_geometry")
    for name, fake_paths in selected_fake.items():
        fake_by_name = {path.name: path for path in fake_paths}
        names = common[name]
        fake_features, fake_confidences = select_physics_rows(physics_cache, name, names)
        fake = combined_cache(name, [fake_by_name[item] for item in names], fake_features, fake_confidences, geometry_cache)
        real_rows = torch.tensor([real_index[item] for item in names], dtype=torch.long)
        features = torch.cat([real["features"][real_rows], fake["features"]])
        confidences = torch.cat([real["confidences"][real_rows], fake["confidences"]])
        labels = np.concatenate([np.zeros(len(names), dtype=int), np.ones(len(names), dtype=int)])
        raw = probability(models, features, confidences, artifact["standardizer"], device)
        calibrated = apply_policy_calibration(probability_logits(raw), artifact["policy"])
        masked_confidences = confidences.clone()
        masked_confidences[:, :, geometry_index] = 0.0
        masked_raw = probability(models, features, masked_confidences, artifact["standardizer"], device)
        masked_calibrated = apply_policy_calibration(probability_logits(masked_raw), artifact["policy"])
        np.savez_compressed(
            prediction_dir / f"{name}.npz", labels=labels.astype(np.int8), filenames=np.asarray(names),
            raw_probability=raw.astype(np.float32), calibrated_probability=calibrated.astype(np.float32),
            geometry_masked_probability=masked_calibrated.astype(np.float32),
        )
        report["generators"][name] = {
            "pairs": len(names),
            "binary": evaluate_predictions(labels, calibrated, artifact["policy"]["decision_threshold"]),
            "geometry_masked_ablation": evaluate_predictions(labels, masked_calibrated, artifact["policy"]["decision_threshold"]),
            "raw_threshold_0_5": evaluate_predictions(labels, raw, 0.5),
            "selective": evaluate_selective_abstention(
                labels, calibrated, confidences.mean(dim=1).numpy(), artifact["policy"]["tau_low"],
                artifact["policy"]["tau_high"], artifact["policy"]["obs_threshold"],
            ),
        }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
