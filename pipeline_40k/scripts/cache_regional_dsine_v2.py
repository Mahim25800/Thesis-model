"""Build a resumable five-region, four-entity physics cache with DSINE."""

from __future__ import annotations

import argparse
import gc
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
from src.extractors.regional_physics import REGION_NAMES, RegionalPhysicsExtractor
from src.extractors.surface_normals import SurfaceNormalsExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SCHEMA = "regional_physics_dsine_v2_5x14_features_5x4_confidences"


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


def atomic_torch_save(value: dict, output: Path) -> None:
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


def samples(partition: Path, limit_per_class: int | None) -> list[tuple[Path, int]]:
    rows = []
    for directory_name, label in (("real", 0), ("fake", 1)):
        directory = partition / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {directory}")
        found = [(path, label) for path in sorted(directory.iterdir())
                 if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        rows.extend(found if limit_per_class is None else found[:limit_per_class])
    if not rows:
        raise ValueError(f"No supported images found in {partition}")
    return rows


def shard_paths(output: Path, name: str, start: int, stop: int) -> tuple[Path, Path]:
    base = output / "shards" / name / f"{start:06d}_{stop:06d}"
    return base.with_suffix(".pt"), base.with_suffix(".jsonl")


def completed_shard(tensor_path: Path, manifest_path: Path, start: int, stop: int) -> bool:
    if not tensor_path.is_file() or not manifest_path.is_file():
        return False
    data = torch.load(tensor_path, map_location="cpu", weights_only=True)
    expected = torch.arange(start, stop, dtype=torch.int64)
    if data["features"].shape != (stop - start, len(REGION_NAMES), 14):
        raise ValueError(f"Corrupt regional feature shard: {tensor_path}")
    if data["confidences"].shape != (stop - start, len(REGION_NAMES), 4):
        raise ValueError(f"Corrupt regional confidence shard: {tensor_path}")
    if not torch.equal(data["source_positions"], expected):
        raise ValueError(f"Unexpected source positions: {tensor_path}")
    return len(manifest_path.read_text(encoding="utf-8").splitlines()) == stop - start


def build_partition(name: str, partition: Path, output: Path, extractor: RegionalPhysicsExtractor,
                    shard_size: int, limit_per_class: int | None) -> dict:
    source_rows = samples(partition, limit_per_class)
    tensor_files, manifest_files = [], []
    for start in range(0, len(source_rows), shard_size):
        stop = min(start + shard_size, len(source_rows))
        tensor_path, manifest_path = shard_paths(output, name, start, stop)
        if not completed_shard(tensor_path, manifest_path, start, stop):
            features, confidences, labels, records = [], [], [], []
            for source_position in tqdm(range(start, stop), desc=f"regional DSINE: {name} {start}:{stop}"):
                path, label = source_rows[source_position]
                # Hash before DSINE allocates its working tensors. This keeps
                # the file buffer outside the peak inference-memory window.
                file_hash = sha256_file(path)
                image = validate_and_load_image(path)
                feature, confidence = extractor.extract(image)
                features.append(feature.cpu())
                confidences.append(confidence.cpu())
                labels.append(label)
                records.append({
                    "source_position": source_position,
                    "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "label": label,
                    "sha256": file_hash,
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
            del features, confidences, labels, records
            gc.collect()
        tensor_files.append(tensor_path)
        manifest_files.append(manifest_path)
    pieces = [torch.load(path, map_location="cpu", weights_only=True) for path in tensor_files]
    merged = {key: torch.cat([piece[key] for piece in pieces]) for key in ("features", "confidences", "labels")}
    del pieces
    gc.collect()
    cache_path = output / f"{name}_features.pt"
    atomic_torch_save(merged, cache_path)
    atomic_text(output / f"{name}_manifest.jsonl", "".join(
        path.read_text(encoding="utf-8") for path in manifest_files
    ))
    labels_np = merged["labels"].numpy().astype(int)
    return {
        "samples": len(labels_np), "real": int(np.sum(labels_np == 0)), "fake": int(np.sum(labels_np == 1)),
        "cache_sha256": sha256_file(cache_path), "shards": len(tensor_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/scaled/train")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data/scaled/test_unseen")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cache_regional_dsine_v2")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output, train_dir, test_dir = args.output_dir.resolve(), args.train_dir.resolve(), args.test_dir.resolve()
    if not all(path.is_relative_to(ROOT) for path in (output, train_dir, test_dir, args.checkpoint.resolve())):
        raise ValueError("All inputs and outputs must remain inside pipeline_40k")
    if args.shard_size <= 0 or (args.limit_per_class is not None and args.limit_per_class <= 0):
        raise ValueError("Shard size and optional limit must be positive")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Choose an empty output directory or pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    normal_extractor = SurfaceNormalsExtractor(
        normal_backend="dsine", dsine_checkpoint=args.checkpoint, dsine_device=args.device
    )
    extractor = RegionalPhysicsExtractor(normal_extractor)
    preamble = {
        "schema_version": 2, "feature_schema": SCHEMA, "regions": list(REGION_NAMES),
        "extractor": extractor.provenance(), "shard_size": args.shard_size,
        "limit_per_class": args.limit_per_class,
    }
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        old_hash = previous.get("extractor", {}).get("normal_extractor", {}).get("checkpoint_sha256")
        new_hash = preamble["extractor"]["normal_extractor"].get("checkpoint_sha256")
        if old_hash != new_hash or previous.get("regions") != preamble["regions"]:
            raise ValueError("Cannot resume a cache with different extraction provenance")
    else:
        atomic_text(provenance_path, json.dumps(preamble, indent=2, sort_keys=True) + "\n")
    train = build_partition("train", train_dir, output, extractor, args.shard_size, args.limit_per_class)
    test = build_partition("test_unseen", test_dir, output, extractor, args.shard_size, args.limit_per_class)
    provenance = preamble | {"train": train, "test_unseen": test}
    atomic_text(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
