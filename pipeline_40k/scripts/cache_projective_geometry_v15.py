"""Add confidence-aware projective geometry to a frozen regional physics cache.

The input cache supplies DSINE-based four-entity features. This script only
adds an eighth-dimensional geometry descriptor for the same global and four
quadrant crops; it never recomputes or alters the original physics values.
Shard files make interrupted CUDA/CPU sessions resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import validate_and_load_image
from src.extractors.perspective_vp import GEOMETRY_DIM, PerspectiveVPExtractor
from src.extractors.regional_physics import REGION_NAMES, quadrant_boxes


INPUT_SCHEMA = "regional_physics_dsine_v2_5x14_features_5x4_confidences"
OUTPUT_SCHEMA = "regional_physics_geometry_v15_5x22_features_5x5_confidences"
SHARD_SIZE = 200


def sha256(path: Path) -> str:
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


def atomic_text(content: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or [row["source_position"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"Manifest is missing or unordered: {path}")
    return rows


def cache_files(cache_dir: Path, selected: list[str] | None) -> list[tuple[str, Path, Path]]:
    files = []
    for feature_path in sorted(cache_dir.glob("*_features.pt")):
        name = feature_path.stem.removesuffix("_features")
        if selected is not None and name not in selected:
            continue
        manifest_path = cache_dir / f"{name}_manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing source manifest for {feature_path}")
        files.append((name, feature_path, manifest_path))
    if not files:
        raise ValueError("No selected cache partitions found")
    return files


def shard_path(output: Path, name: str, start: int, stop: int) -> Path:
    return output / "shards" / name / f"{start:06d}_{stop:06d}.pt"


def valid_shard(path: Path, count: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        return value["features"].shape == (count, len(REGION_NAMES), GEOMETRY_DIM) \
            and value["confidences"].shape == (count, len(REGION_NAMES), 1) \
            and torch.isfinite(value["features"]).all() and torch.isfinite(value["confidences"]).all()
    except (KeyError, RuntimeError, ValueError):
        return False


def geometry_for_image(image: torch.Tensor | object, extractor: PerspectiveVPExtractor) -> tuple[torch.Tensor, torch.Tensor]:
    image_np = validate_and_load_image(image)
    rows, confidences = [], []
    for y1, y2, x1, x2 in quadrant_boxes(*image_np.shape[:2]):
        feature, confidence = extractor(image_np[y1:y2, x1:x2])
        rows.append(feature)
        confidences.append([confidence])
    result = torch.stack(rows).float()
    confidence_result = torch.tensor(confidences, dtype=torch.float32)
    if result.shape != (len(REGION_NAMES), GEOMETRY_DIM) or confidence_result.shape != (len(REGION_NAMES), 1):
        raise RuntimeError("Geometry extractor emitted an incompatible regional shape")
    return result, confidence_result


def build_partition(name: str, feature_path: Path, manifest_path: Path, output: Path,
                    extractor: PerspectiveVPExtractor, shard_size: int, limit: int | None) -> dict:
    source = torch.load(feature_path, map_location="cpu", weights_only=True)
    physics, physics_confidence, labels = source["features"].float(), source["confidences"].float(), source["labels"].float()
    if physics.shape[1:] != (len(REGION_NAMES), 14) or physics_confidence.shape != (len(physics), len(REGION_NAMES), 4):
        raise ValueError(f"Input cache is not four-entity regional physics: {feature_path}")
    manifest = read_manifest(manifest_path)
    if len(manifest) != len(physics) or labels.shape != (len(physics),):
        raise ValueError(f"Cache and manifest rows differ: {name}")
    if limit is not None:
        physics, physics_confidence, labels, manifest = (
            physics[:limit], physics_confidence[:limit], labels[:limit], manifest[:limit]
        )

    shards = []
    for start in range(0, len(manifest), shard_size):
        stop = min(start + shard_size, len(manifest))
        path = shard_path(output, name, start, stop)
        if not valid_shard(path, stop - start):
            features, confidences = [], []
            for row in tqdm(manifest[start:stop], desc=f"geometry {name} {start}:{stop}"):
                source_path = (ROOT / row["relative_path"]).resolve()
                if not source_path.is_relative_to(ROOT) or not source_path.is_file():
                    raise FileNotFoundError(f"Manifest image is unavailable inside pipeline_40k: {source_path}")
                feature, confidence = geometry_for_image(source_path, extractor)
                features.append(feature)
                confidences.append(confidence)
            atomic_torch_save({"features": torch.stack(features), "confidences": torch.stack(confidences)}, path)
        shards.append(torch.load(path, map_location="cpu", weights_only=True))

    geometry_features = torch.cat([item["features"] for item in shards])
    geometry_confidences = torch.cat([item["confidences"] for item in shards])
    combined_features = torch.cat([physics, geometry_features], dim=2)
    combined_confidences = torch.cat([physics_confidence, geometry_confidences], dim=2)
    if combined_features.shape != (len(physics), len(REGION_NAMES), 14 + GEOMETRY_DIM):
        raise RuntimeError("Combined feature cache has an invalid shape")
    if combined_confidences.shape != (len(physics), len(REGION_NAMES), 5):
        raise RuntimeError("Combined confidence cache has an invalid shape")
    atomic_torch_save(
        {"features": combined_features, "confidences": combined_confidences, "labels": labels},
        output / feature_path.name,
    )
    atomic_text(manifest_path.read_text(encoding="utf-8"), output / manifest_path.name)
    return {
        "samples": len(physics), "feature_source_sha256": sha256(feature_path),
        "manifest_source_sha256": sha256(manifest_path), "geometry_shards": len(shards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--partitions", nargs="+", default=None)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--limit-per-partition", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source, output = args.input_cache.resolve(), args.output_cache.resolve()
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Cache paths must remain inside pipeline_40k")
    if args.shard_size <= 0 or (args.limit_per_partition is not None and args.limit_per_partition <= 0):
        raise ValueError("shard-size and limit-per-partition must be positive")
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("feature_schema") != INPUT_SCHEMA:
        raise ValueError("Input cache does not use the official regional DSINE schema")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Choose an empty output cache or pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    geometry = PerspectiveVPExtractor()
    output_provenance = {
        "schema_version": 1,
        "feature_schema": OUTPUT_SCHEMA,
        "regions": list(REGION_NAMES),
        "entity_names": ["illumination", "specular_optics", "surface_normals", "chromatic_shadows", "projective_geometry"],
        "entity_dims": [5, 4, 3, 2, GEOMETRY_DIM],
        "source_cache": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_provenance_sha256": sha256(source / "provenance.json"),
        "geometry": {
            "extractor": "PerspectiveVPExtractor",
            "feature_names": [
                "line_density", "mean_line_length", "line_coverage", "orientation_entropy",
                "dominant_orientation_support", "vanishing_point_hypothesis_support",
                "vanishing_point_hypothesis_compactness", "cross_scale_orientation_agreement",
            ],
            "confidence": "line-observability and projective-hypothesis support",
            "claim_boundary": "descriptors are image-space projective evidence, not a proof that an image violates a physical law",
        },
        "partitions": {},
    }
    for name, feature_path, manifest_path in cache_files(source, args.partitions):
        output_provenance["partitions"][name] = build_partition(
            name, feature_path, manifest_path, output, geometry, args.shard_size, args.limit_per_partition
        )
        atomic_text(json.dumps(output_provenance, indent=2, sort_keys=True) + "\n", output / "provenance.json")
    print(json.dumps(output_provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
