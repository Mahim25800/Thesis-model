"""Build provenance-preserving, training-only DSINE feature augmentation caches.

Each output row is a deterministic JPEG recompression, Gaussian blur, or
downscale-upscale version of one existing cache row.  The original feature
cache remains untouched and is kept alongside this augmented copy during
fitting.  Development, calibration, internal test, and external benchmark rows
must continue to use their original features only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_regional_dsine_v2 import SCHEMA, atomic_text, sha256_file
from scripts.train_regional_multi_physics_v5 import read_manifest
from src.data.preprocessor import validate_and_load_image
from src.extractors.dsine_normals import DEFAULT_CHECKPOINT
from src.extractors.regional_physics import REGION_NAMES, RegionalPhysicsExtractor
from src.extractors.surface_normals import SurfaceNormalsExtractor


ENTITY_DIMS = (5, 4, 3, 2)


def atomic_torch_save(value: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_rng(parent_sha256: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{parent_sha256}".encode("ascii")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="big", signed=False))


def augment_image(image: np.ndarray, parent_sha256: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply exactly one seeded, label-preserving image transformation."""
    rng = deterministic_rng(parent_sha256, seed)
    kind = ("jpeg", "gaussian_blur", "resize_jitter")[int(rng.integers(0, 3))]
    if kind == "jpeg":
        quality = int(rng.integers(70, 101))
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("OpenCV could not encode JPEG augmentation")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("OpenCV could not decode JPEG augmentation")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), {"type": kind, "quality": quality}
    if kind == "gaussian_blur":
        # The protocol permits sigma in [0, 1]. Keep a small positive lower
        # bound so an augmented copy is never accidentally an identity copy.
        sigma = float(rng.uniform(0.05, 1.0))
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma), {"type": kind, "sigma": sigma}
    factor = float(rng.uniform(0.65, 1.0))
    height, width = image.shape[:2]
    resized_width, resized_height = max(1, round(width * factor)), max(1, round(height * factor))
    downsampled = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return restored, {
        "type": kind, "downscale_factor": factor,
        "downsample_interpolation": "area", "restore_interpolation": "lanczos4",
    }


def shard_paths(output: Path, name: str, start: int, stop: int) -> tuple[Path, Path]:
    base = output / "shards" / name / f"{start:06d}_{stop:06d}"
    return base.with_suffix(".pt"), base.with_suffix(".jsonl")


def completed_shard(tensor_path: Path, manifest_path: Path, start: int, stop: int) -> bool:
    if not tensor_path.is_file() or not manifest_path.is_file():
        return False
    values = torch.load(tensor_path, map_location="cpu", weights_only=True)
    expected = torch.arange(start, stop, dtype=torch.int64)
    valid = (
        values.get("features", torch.empty(0)).shape == (stop - start, len(REGION_NAMES), sum(ENTITY_DIMS))
        and values.get("confidences", torch.empty(0)).shape == (stop - start, len(REGION_NAMES), len(ENTITY_DIMS))
        and torch.equal(values.get("source_positions", torch.empty(0, dtype=torch.int64)), expected)
    )
    if not valid:
        raise ValueError(f"Corrupt augmentation shard: {tensor_path}")
    return len(manifest_path.read_text(encoding="utf-8").splitlines()) == stop - start


def build_partition(
    *, name: str, source_cache: Path, output: Path, extractor: RegionalPhysicsExtractor,
    shard_size: int, seed: int, limit: int | None,
) -> dict[str, Any]:
    source_path = source_cache / f"{name}_features.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    manifest = read_manifest(source_cache / f"{name}_manifest.jsonl")
    total = len(manifest) if limit is None else min(len(manifest), limit)
    if source["features"].shape != (len(manifest), len(REGION_NAMES), sum(ENTITY_DIMS)):
        raise ValueError(f"Source feature rows and manifest do not align: {source_path}")
    if source["confidences"].shape != (len(manifest), len(REGION_NAMES), len(ENTITY_DIMS)) \
            or source["labels"].shape != (len(manifest),):
        raise ValueError(f"Source cache tensors do not align: {source_path}")

    tensor_paths, manifest_paths = [], []
    for start in range(0, total, shard_size):
        stop = min(start + shard_size, total)
        tensor_path, manifest_path = shard_paths(output, name, start, stop)
        if not completed_shard(tensor_path, manifest_path, start, stop):
            features, confidences, records = [], [], []
            for position in tqdm(range(start, stop), desc=f"augmented DSINE: {name} {start}:{stop}"):
                parent = manifest[position]
                relative_path = Path(parent["relative_path"])
                path = (ROOT / relative_path).resolve()
                if not path.is_relative_to(ROOT) or not path.is_file():
                    raise FileNotFoundError(f"Missing in-project source image: {relative_path}")
                image = validate_and_load_image(path)
                augmented, transform = augment_image(image, str(parent["sha256"]), seed)
                feature, confidence = extractor.extract(augmented)
                features.append(feature.cpu())
                confidences.append(confidence.cpu())
                records.append({
                    "source_position": position,
                    "parent_relative_path": parent["relative_path"],
                    "parent_sha256": parent["sha256"],
                    "parent_difference_hash_64": parent["difference_hash_64"],
                    "label": int(parent["label"]),
                    "augmentation_seed": seed,
                    "transform": transform,
                })
            atomic_torch_save({
                "features": torch.stack(features).float(),
                "confidences": torch.stack(confidences).float(),
                "labels": source["labels"][start:stop].clone().float(),
                "source_positions": torch.arange(start, stop, dtype=torch.int64),
            }, tensor_path)
            atomic_text(manifest_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
            del features, confidences, records
            gc.collect()
        tensor_paths.append(tensor_path)
        manifest_paths.append(manifest_path)
    pieces = [torch.load(path, map_location="cpu", weights_only=True) for path in tensor_paths]
    merged = {key: torch.cat([piece[key] for piece in pieces]) for key in ("features", "confidences", "labels")}
    del pieces
    gc.collect()
    target = output / f"{name}_features.pt"
    atomic_torch_save(merged, target)
    atomic_text(output / f"{name}_manifest.jsonl", "".join(path.read_text(encoding="utf-8") for path in manifest_paths))
    labels = merged["labels"].numpy().astype(int)
    return {
        "samples": int(len(labels)), "source_samples": int(len(manifest)),
        "complete": bool(total == len(manifest)), "real": int(np.sum(labels == 0)), "fake": int(np.sum(labels == 1)),
        "cache_sha256": sha256_file(target), "shards": len(tensor_paths),
        "source_cache_features_sha256": sha256_file(source_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, default=ROOT / "data/cache_regional_dsine_v2")
    parser.add_argument("--genimage-cache", type=Path, default=ROOT / "data/cache_genimage_regional_v2")
    parser.add_argument("--base-output", type=Path, default=ROOT / "data/cache_regional_dsine_v16_augmented")
    parser.add_argument("--genimage-output", type=Path, default=ROOT / "data/cache_genimage_regional_v16_augmented")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only; cannot be used for selection.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = [args.base_cache.resolve(), args.genimage_cache.resolve(), args.base_output.resolve(), args.genimage_output.resolve(), args.checkpoint.resolve()]
    if not all(path.is_relative_to(ROOT) for path in paths):
        raise ValueError("All inputs and outputs must remain inside pipeline_40k")
    if args.shard_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Shard size and optional limit must be positive")
    for output in (args.base_output.resolve(), args.genimage_output.resolve()):
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError("Choose empty output directories or pass --resume")
        output.mkdir(parents=True, exist_ok=True)
    base_provenance = json.loads((args.base_cache.resolve() / "provenance.json").read_text(encoding="utf-8"))
    genimage_provenance = json.loads((args.genimage_cache.resolve() / "provenance.json").read_text(encoding="utf-8"))
    if base_provenance.get("feature_schema") != SCHEMA or genimage_provenance.get("feature_schema") != SCHEMA:
        raise ValueError("Both source caches must use the regional DSINE feature schema")
    normals = SurfaceNormalsExtractor(normal_backend="dsine", dsine_checkpoint=args.checkpoint.resolve(), dsine_device=args.device)
    extractor = RegionalPhysicsExtractor(normals)
    protocol = {
        "purpose": "one deterministic non-identity raw-image augmentation per fitting image; append only during model fitting",
        "seed": args.seed,
        "transforms": {
            "jpeg_quality": [70, 100], "gaussian_blur_sigma": [0.05, 1.0],
            "resize_downscale_factor": [0.65, 1.0], "composition": "exactly one transform per augmented copy",
        },
        "exclusions": ["calibration", "internal_development", "internal_test", "held_generator_development", "Synthbuster", "confirmation"],
    }
    base_record = {"schema_version": 1, "feature_schema": SCHEMA, "regions": list(REGION_NAMES), "extractor": extractor.provenance(), "protocol": protocol}
    genimage_record = base_record | {"source_provenance": genimage_provenance.get("source_provenance", {})}
    for output, record in ((args.base_output.resolve(), base_record), (args.genimage_output.resolve(), genimage_record)):
        provenance = output / "provenance.json"
        if provenance.exists():
            existing = json.loads(provenance.read_text(encoding="utf-8"))
            if existing.get("extractor", {}).get("normal_extractor", {}).get("checkpoint_sha256") != record["extractor"]["normal_extractor"]["checkpoint_sha256"] \
                    or existing.get("protocol") != protocol:
                raise ValueError("Cannot resume an augmentation cache with different provenance")
        else:
            atomic_text(provenance, json.dumps(record, indent=2, sort_keys=True) + "\n")
    base_record["train"] = build_partition(
        name="train", source_cache=args.base_cache.resolve(), output=args.base_output.resolve(), extractor=extractor,
        shard_size=args.shard_size, seed=args.seed, limit=args.limit,
    )
    atomic_text(args.base_output.resolve() / "provenance.json", json.dumps(base_record, indent=2, sort_keys=True) + "\n")
    genimage_record["domains"] = {}
    for domain in ("adm", "biggan", "vqdm", "wukong"):
        genimage_record["domains"][domain] = build_partition(
            name=domain, source_cache=args.genimage_cache.resolve(), output=args.genimage_output.resolve(), extractor=extractor,
            shard_size=args.shard_size, seed=args.seed, limit=args.limit,
        )
        atomic_text(args.genimage_output.resolve() / "provenance.json", json.dumps(genimage_record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"base": base_record, "genimage": genimage_record}, indent=2), flush=True)


if __name__ == "__main__":
    main()
