"""Build a resumable, versioned physics cache with official DSINE normals.

This always writes a new cache directory. Each completed shard contains the
source positions it covers, so interrupted CPU runs can resume safely without
mixing historical and DSINE-derived descriptors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import validate_and_load_image
from src.extractors.dsine_normals import DEFAULT_CHECKPOINT
from src.models.pipeline import PhysicsDeepfakePipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SCHEMA = "physics_cache_dsine_v1_14_features_4_confidences"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] >= small[:, :-1]).reshape(-1)
    return f"{int(''.join('1' if value else '0' for value in bits), 2):016x}"


def atomic_torch_save(value, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def samples(partition: Path, limit_per_class: int | None = None):
    rows = []
    for directory_name, label in (("real", 0), ("fake", 1)):
        directory = partition / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {directory}")
        class_rows = [(path, label) for path in sorted(directory.iterdir())
                      if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()]
        rows.extend(class_rows[:limit_per_class])
    if not rows:
        raise ValueError(f"No supported images found in {partition}")
    return rows


def shard_paths(output: Path, name: str, start: int, stop: int):
    base = output / "shards" / name / f"{start:06d}_{stop:06d}"
    return base.with_suffix(".pt"), base.with_suffix(".jsonl")


def completed_shard(tensor_path: Path, manifest_path: Path, start: int, stop: int) -> bool:
    if not tensor_path.is_file() or not manifest_path.is_file():
        return False
    data = torch.load(tensor_path, map_location="cpu", weights_only=True)
    expected = torch.arange(start, stop, dtype=torch.int64)
    if data["features"].shape != (stop - start, 14) or data["confidences"].shape != (stop - start, 4):
        raise ValueError(f"Corrupt DSINE shard shape: {tensor_path}")
    if not torch.equal(data["source_positions"], expected):
        raise ValueError(f"DSINE shard has unexpected source positions: {tensor_path}")
    if len(manifest_path.read_text(encoding="utf-8").splitlines()) != stop - start:
        raise ValueError(f"DSINE shard manifest length mismatch: {manifest_path}")
    return True


def build_partition(
    name: str,
    partition: Path,
    output: Path,
    pipeline: PhysicsDeepfakePipeline,
    shard_size: int,
    limit_per_class: int | None,
) -> dict:
    source_rows = samples(partition, limit_per_class)
    shard_files, manifest_files = [], []
    for start in range(0, len(source_rows), shard_size):
        stop = min(start + shard_size, len(source_rows))
        tensor_path, manifest_path = shard_paths(output, name, start, stop)
        if completed_shard(tensor_path, manifest_path, start, stop):
            shard_files.append(tensor_path)
            manifest_files.append(manifest_path)
            continue
        features, confidences, labels, records = [], [], [], []
        for source_position in tqdm(range(start, stop), desc=f"DSINE cache: {name} {start}:{stop}"):
            path, label = source_rows[source_position]
            image = validate_and_load_image(path)
            delta, confidence, _ = pipeline.extract_features(image)
            features.append(delta.cpu())
            confidences.append(confidence.cpu())
            labels.append(label)
            records.append({
                "source_position": source_position,
                "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "label": label,
                "sha256": sha256_file(path),
                "difference_hash_64": difference_hash(image),
                "width": int(image.shape[1]), "height": int(image.shape[0]),
            })
        atomic_torch_save({
            "features": torch.stack(features).float(),
            "confidences": torch.stack(confidences).float(),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "source_positions": torch.arange(start, stop, dtype=torch.int64),
        }, tensor_path)
        atomic_text(manifest_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        shard_files.append(tensor_path)
        manifest_files.append(manifest_path)

    pieces = [torch.load(path, map_location="cpu", weights_only=True) for path in shard_files]
    merged = {
        "features": torch.cat([piece["features"] for piece in pieces]),
        "confidences": torch.cat([piece["confidences"] for piece in pieces]),
        "labels": torch.cat([piece["labels"] for piece in pieces]),
    }
    if merged["features"].shape != (len(source_rows), 14) or merged["confidences"].shape != (len(source_rows), 4):
        raise RuntimeError("DSINE cache emitted an unexpected merged schema")
    cache_path = output / f"{name}_features.pt"
    atomic_torch_save(merged, cache_path)
    atomic_text(output / f"{name}_manifest.jsonl", "".join(path.read_text(encoding="utf-8") for path in manifest_files))
    labels = merged["labels"].numpy().astype(int)
    return {"samples": len(labels), "real": int(np.sum(labels == 0)), "fake": int(np.sum(labels == 1)),
            "cache_sha256": sha256_file(cache_path), "shards": len(shard_files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/scaled/train")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data/scaled/test_unseen")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cache_regenerated_dsine_v1")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fov-degrees", type=float, default=60.0)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Cache only the first N sorted images in each class; intended for a smoke test.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if args.limit_per_class is not None and args.limit_per_class <= 0:
        raise ValueError("limit-per-class must be positive")
    output, train_dir, test_dir = args.output_dir.resolve(), args.train_dir.resolve(), args.test_dir.resolve()
    if not output.is_relative_to(ROOT) or not train_dir.is_relative_to(ROOT) or not test_dir.is_relative_to(ROOT):
        raise ValueError("Raw inputs and outputs must remain inside pipeline_40k")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Choose a new empty DSINE cache directory or pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    pipeline = PhysicsDeepfakePipeline(device=args.device, normal_backend="dsine", normal_checkpoint=args.checkpoint)
    preamble = {"schema_version": 1, "feature_schema": SCHEMA, "normal_extractor": pipeline.extractor_normals.provenance(),
                "shard_size": args.shard_size, "limit_per_class": args.limit_per_class,
                "difference_hash_note": "64-bit dHash supports near-duplicate triage; group-aware split enforcement is a separate audit step."}
    preamble_path = output / "provenance.json"
    if preamble_path.exists():
        previous = json.loads(preamble_path.read_text(encoding="utf-8"))
        if previous.get("normal_extractor", {}).get("checkpoint_sha256") != preamble["normal_extractor"]["checkpoint_sha256"]:
            raise ValueError("Cannot resume a cache built with a different DSINE checkpoint")
    else:
        atomic_text(preamble_path, json.dumps(preamble, indent=2, sort_keys=True) + "\n")
    train_summary = build_partition("train", train_dir, output, pipeline, args.shard_size, args.limit_per_class)
    test_summary = build_partition("test_unseen", test_dir, output, pipeline, args.shard_size, args.limit_per_class)
    provenance = preamble | {"train": train_summary, "test_unseen": test_summary,
                              "raw_image_inference": "new features were produced from raw RGB with the recorded DSINE recipe"}
    atomic_text(preamble_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
