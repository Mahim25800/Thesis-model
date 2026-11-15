"""Evaluate a frozen regional multi-physics model on external Synthbuster."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_scaled import evaluate_selective_abstention
from src.data.preprocessor import validate_and_load_image
from src.extractors.regional_physics import REGION_NAMES, RegionalPhysicsExtractor
from src.extractors.surface_normals import SurfaceNormalsExtractor
from src.models.improved import probability_logits
from src.models.multi_entity import ENTITY_DIMS, ENTITY_NAMES
from src.models.regional_multi_physics import RegionalMultiPhysicsHead, standardize_regional
from src.utils.decision_policy import apply_policy_calibration
from src.utils.metrics import evaluate_predictions


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SCHEMA = "regional_physics_dsine_v2_5x14_features_5x4_confidences"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def cache_source(name: str, paths: list[Path], output: Path, extractor: RegionalPhysicsExtractor) -> dict:
    cache_path = output / f"{name}.pt"
    manifest_path = output / f"{name}.jsonl"
    if cache_path.is_file() and manifest_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache["features"].shape == (len(paths), len(REGION_NAMES), sum(ENTITY_DIMS)) \
                and cache["confidences"].shape == (len(paths), len(REGION_NAMES), len(ENTITY_DIMS)):
            return cache
        raise ValueError(f"Existing regional cache is incompatible: {cache_path}")
    features, confidences, records = [], [], []
    for index, path in enumerate(paths, start=1):
        image = validate_and_load_image(path)
        feature, confidence = extractor.extract(image)
        features.append(feature.cpu())
        confidences.append(confidence.cpu())
        records.append({
            "index": index - 1, "filename": path.name, "sha256": sha256_file(path),
            "width": int(image.shape[1]), "height": int(image.shape[0]),
        })
        if index % 50 == 0 or index == len(paths):
            print(f"Regional external cache {name}: {index}/{len(paths)}", flush=True)
    cache = {"features": torch.stack(features).float(), "confidences": torch.stack(confidences).float()}
    atomic_torch_save(cache, cache_path)
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    return cache


def probability(models: list[RegionalMultiPhysicsHead], features: torch.Tensor, confidences: torch.Tensor,
                standardizer: dict, device: torch.device) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            x = standardize_regional(features[start:start + 512].to(device), standardizer)
            c = confidences[start:start + 512].to(device)
            values.append(np.mean([torch.sigmoid(model(x, c)).cpu().numpy() for model in models], axis=0))
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "data/external_synthbuster")
    parser.add_argument("--model", type=Path, default=ROOT / "models/generator_invariant_v6/model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/generator_invariant_v6/external_synthbuster")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache_synthbuster_regional_v2")
    parser.add_argument("--generators", nargs="+", default=None)
    parser.add_argument("--limit-per-source", type=int, default=None)
    args = parser.parse_args()
    root, model_path = args.benchmark_root.resolve(), args.model.resolve()
    output, cache_dir = args.output_dir.resolve(), args.cache_dir.resolve()
    if not all(path.is_relative_to(ROOT) for path in (root, model_path, output, cache_dir)):
        raise ValueError("All paths must remain inside pipeline_40k")
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    if artifact.get("feature_schema") != SCHEMA or tuple(artifact.get("entity_dims", ())) != ENTITY_DIMS:
        raise ValueError("Model is not compatible with regional physics features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for state in artifact["model_states"]:
        model = RegionalMultiPhysicsHead(**artifact["architecture"]).to(device).eval()
        model.load_state_dict(state)
        models.append(model)
    normals = SurfaceNormalsExtractor(normal_backend="dsine", dsine_device=str(device))
    extractor = RegionalPhysicsExtractor(normals)
    real_dir = root / "raise" / "real_RAISE_1k"
    fake_root = root / "synthbuster" / "synthbuster"
    generators = sorted(path.name for path in fake_root.iterdir() if path.is_dir()) if args.generators is None else args.generators
    real_by_name = {path.name: path for path in source_paths(real_dir, None)}
    selected_fake = {name: source_paths(fake_root / name, args.limit_per_source) for name in generators}
    common = {name: sorted(set(real_by_name) & {path.name for path in paths}) for name, paths in selected_fake.items()}
    if any(not names for names in common.values()):
        raise ValueError("A selected generator has no matching RAISE filenames")
    real_names = sorted(set().union(*[set(names) for names in common.values()]))
    real_paths = [real_by_name[name] for name in real_names]
    real_cache = cache_source("raise_real", real_paths, cache_dir, extractor)
    real_index = {path.name: index for index, path in enumerate(real_paths)}
    policy = artifact["policy"]
    report = {
        "protocol": "frozen regional model, standardizer, calibration, and threshold; no Synthbuster fitting",
        "benchmark": "Synthbuster generated images paired with RAISE real images",
        "model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "device": str(device), "extractor": extractor.provenance(), "generators": {},
    }
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for name, fake_paths in selected_fake.items():
        fake_by_name = {path.name: path for path in fake_paths}
        names = common[name]
        fake_cache = cache_source(name, [fake_by_name[item] for item in names], cache_dir, extractor)
        real_rows = torch.tensor([real_index[item] for item in names], dtype=torch.long)
        features = torch.cat([real_cache["features"][real_rows], fake_cache["features"]])
        confidences = torch.cat([real_cache["confidences"][real_rows], fake_cache["confidences"]])
        if artifact.get("disabled_entities"):
            confidences = confidences.clone()
            disabled_indices = [ENTITY_NAMES.index(item) for item in artifact["disabled_entities"]]
            confidences[:, :, disabled_indices] = 0.0
        labels = np.concatenate([np.zeros(len(names), dtype=int), np.ones(len(names), dtype=int)])
        raw = probability(models, features, confidences, artifact["standardizer"], device)
        calibrated = apply_policy_calibration(probability_logits(raw), policy)
        np.savez_compressed(
            prediction_dir / f"{name}.npz",
            labels=labels.astype(np.int8),
            raw_probability=raw.astype(np.float32),
            calibrated_probability=calibrated.astype(np.float32),
            filenames=np.asarray(names),
        )
        report["generators"][name] = {
            "pairs": len(names),
            "binary": evaluate_predictions(labels, calibrated, policy["decision_threshold"]),
            "raw_threshold_0_5": evaluate_predictions(labels, raw, 0.5),
            "selective": evaluate_selective_abstention(
                labels, calibrated, confidences.mean(dim=1).numpy(),
                policy["tau_low"], policy["tau_high"], policy["obs_threshold"],
            ),
        }
        print(f"Evaluated frozen regional model on {name}: {len(names)} pairs", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
