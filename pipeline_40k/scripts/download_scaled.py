"""High-Throughput Resumable Streaming Downloader for Scaled 40k+ Dataset.

Partitions:
- train/real:       16,000 images (COCO Train 2017)
- train/fake:       16,000 images (DiffusionDB Stable Diffusion)
- test_unseen/real:  4,000 images (COCO Val 2017)
- test_unseen/fake:  4,000 images (different DiffusionDB subset, not an unseen generator)

Features:
- Multi-threaded disk writer pool.
- Automatic resume capability: skips already-downloaded images by inspecting directory state.
- On-the-fly RGB JPEG verification with automatic corruption rejection.
"""

import sys
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Optional, Set
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Ensure pipeline root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import save_verified_jpeg


def get_existing_file_indices(directory: Path, prefix: str) -> Set[int]:
    """Scans directory for existing valid JPEG files and extracts index numbers."""
    if not directory.exists():
        return set()
    indices = set()
    for file_path in directory.glob(f"{prefix}_*.jpg"):
        name_stem = file_path.stem
        try:
            idx = int(name_stem.split("_")[-1])
            with Image.open(file_path) as check:
                check.verify()
            with Image.open(file_path) as check:
                check.load()
                if check.mode != "RGB":
                    continue
            indices.add(idx)
        except (ValueError, OSError, SyntaxError):
            continue
    return indices


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save(image, path):
    """Only publish a destination after decoding the complete temporary JPEG."""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_verified_jpeg(image, temporary)
        with Image.open(temporary) as check:
            check.load()
            if check.mode != "RGB":
                raise ValueError("Encoded image is not RGB")
        digest = _sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_manifest(path, manifest):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def stream_partition(
    dataset_name: str,
    split_name: str,
    dest_dir: Path,
    prefix: str,
    target_count: int,
    config_name: Optional[str] = None,
    max_workers: int = 8,
    stream_offset: int = 0,
    revision: Optional[str] = None,
    max_failures: int = 1000,
):
    """Download one explicit source with source-row filenames and checked resume.

    A missing/corrupt source is an error. Existing legacy files without provenance
    cannot safely be resumed; use a new destination directory to regenerate them.
    Source-row indexes (including failed/skipped rows) preserve stream alignment.
    """
    if target_count <= 0 or max_workers <= 0 or stream_offset < 0 or max_failures < 0:
        raise ValueError("Positive target/workers and nonnegative offset/failure limit required")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    source = {"dataset": dataset_name, "config": config_name, "split": split_name,
              "stream_offset": stream_offset, "revision": revision, "prefix": prefix}
    manifest_path = dest_dir / f"{prefix}_download_manifest.json"
    disk_files = set(dest_dir.glob(f"{prefix}_*.jpg"))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("source") != source:
            raise ValueError(f"Source/provenance mismatch in {manifest_path}; use a new destination")
    elif disk_files:
        raise ValueError(f"Cannot safely resume legacy files without source provenance in {dest_dir}; use a new destination")
    else:
        manifest = {"schema_version": 1, "source": source, "images": {},
                    "status": "incomplete", "failure_count": 0, "target_count": target_count}
        _save_manifest(manifest_path, manifest)
    existing_indices = get_existing_file_indices(dest_dir, prefix)
    recorded = {int(index) for index in manifest["images"]}
    expected_files = {dest_dir / f"{prefix}_{index:06d}.jpg" for index in recorded}
    if existing_indices != recorded or disk_files != expected_files:
        raise ValueError(f"Missing, corrupt or untracked files in {dest_dir}; provenance must be repaired before resuming")
    for index in recorded:
        path = dest_dir / f"{prefix}_{index:06d}.jpg"
        record = manifest["images"][str(index)]
        if record["source_row"] != stream_offset + index or _sha256(path) != record["sha256"]:
            raise ValueError(f"Provenance/hash mismatch for {path}")
    if len(recorded) > target_count:
        raise ValueError(f"Destination already contains {len(recorded)} images, exceeding requested {target_count}")
    current_count = len(recorded)
    if current_count == target_count:
        print(f"[{prefix.upper()}] Verified complete partition: {current_count}/{target_count}")
        return manifest
    manifest.update(target_count=target_count, status="incomplete")
    _save_manifest(manifest_path, manifest)
    print(f"[{prefix.upper()}] Verified {current_count}/{target_count}; source {dataset_name}/{split_name}")
    try:
        kwargs = {"split": split_name, "streaming": True}
        if revision is not None:
            kwargs["revision"] = revision
        if config_name:
            ds = load_dataset(dataset_name, config_name, **kwargs)
        else:
            ds = load_dataset(dataset_name, **kwargs)
        stream_iter = iter(ds)
    except Exception as e:
        raise RuntimeError(f"Requested source {dataset_name}/{split_name} failed; no source substitution permitted") from e
    for row in range(stream_offset):
        try:
            next(stream_iter)
        except StopIteration as error:
            raise RuntimeError(f"Source exhausted at row {row} before offset {stream_offset}") from error

    failures = 0
    initial_failure_count = int(manifest.get("failure_count", 0))
    index = 0
    exhausted = False
    pbar = tqdm(total=target_count, initial=current_count, desc=f"Ingesting {dest_dir.name}/{prefix}")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while current_count < target_count and not exhausted:
                pending = []
                stream_error = None
                while len(pending) < min(max_workers * 4, target_count - current_count):
                    try:
                        sample = next(stream_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    except Exception as error:
                        stream_error = error
                        break
                    source_index = index
                    index += 1
                    if source_index in existing_indices:
                        continue
                    image = sample.get("image")
                    if image is None:
                        failures += 1
                        if failures > max_failures:
                            break
                        continue
                    path = dest_dir / f"{prefix}_{source_index:06d}.jpg"
                    pending.append((source_index, path, executor.submit(_atomic_save, image, path)))
                for source_index, path, future in pending:
                    try:
                        digest = future.result()
                    except Exception as error:
                        failures += 1
                        print(f"Rejected source row {stream_offset + source_index}: {error}")
                        continue
                    manifest["images"][str(source_index)] = {
                        "source_row": stream_offset + source_index, "file": path.name, "sha256": digest}
                    existing_indices.add(source_index)
                    current_count += 1
                    pbar.update(1)
                manifest["failure_count"] = initial_failure_count + failures
                _save_manifest(manifest_path, manifest)
                if stream_error is not None:
                    raise RuntimeError("Requested source failed during iteration; partial progress recorded") from stream_error
                if failures > max_failures:
                    raise RuntimeError(f"Exceeded failure limit ({max_failures}); verified {current_count}/{target_count}")
    finally:
        pbar.close()
    if current_count != target_count or len(get_existing_file_indices(dest_dir, prefix)) != target_count:
        raise RuntimeError(f"Source exhausted or verification failed: {current_count}/{target_count} images; source was not repeated")
    manifest["status"] = "complete"
    _save_manifest(manifest_path, manifest)
    print(f"[{prefix.upper()}] Verified {current_count}/{target_count} images with source-row provenance")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Scaled 40k+ Deepfake Dataset Streaming Downloader")
    parser.add_argument("--base-dir", type=str, default="data/scaled")
    parser.add_argument("--train-real", type=int, default=16000)
    parser.add_argument("--train-fake", type=int, default=16000)
    parser.add_argument("--test-real", type=int, default=4000)
    parser.add_argument("--test-fake", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    base_dir = PROJECT_ROOT / args.base_dir

    train_real_dir = base_dir / "train" / "real"
    train_fake_dir = base_dir / "train" / "fake"
    test_real_dir = base_dir / "test_unseen" / "real"
    test_fake_dir = base_dir / "test_unseen" / "fake"

    print("=" * 70)
    print("SCALED 40,000+ IMAGE DATASET STREAMING INGESTION")
    print(f"Base Directory: {base_dir}")
    print(f"Train Real Target:       {args.train_real} (COCO Train 2017)")
    print(f"Train Fake Target:       {args.train_fake} (DiffusionDB 50k)")
    print(f"Test Unseen Real Target: {args.test_real} (COCO Val 2017)")
    print(f"Test Unseen Fake Target: {args.test_fake} (Held-out DiffusionDB 50k)")
    print("=" * 70)

    # 1. Train Real: COCO Train 2017
    stream_partition(
        dataset_name="detection-datasets/coco",
        split_name="train",
        dest_dir=train_real_dir,
        prefix="real",
        target_count=args.train_real,
        max_workers=args.workers,
    )

    # 2. Train Fake: DiffusionDB 50k (Stable Diffusion)
    stream_partition(
        dataset_name="svjack/diffusiondb_2m_random_50k",
        split_name="train",
        dest_dir=train_fake_dir,
        prefix="fake",
        target_count=args.train_fake,
        max_workers=args.workers,
    )

    # 3. Test Unseen Real: COCO Val 2017
    stream_partition(
        dataset_name="detection-datasets/coco",
        split_name="val",
        dest_dir=test_real_dir,
        prefix="real",
        target_count=args.test_real,
        max_workers=args.workers,
    )

    # 4. Test Unseen Fake: Distinct Generative Partition (svjack/diffusiondb_random_10k)
    stream_partition(
        dataset_name="svjack/diffusiondb_random_10k",
        split_name="train",
        dest_dir=test_fake_dir,
        prefix="fake",
        target_count=args.test_fake,
        max_workers=args.workers,
        stream_offset=0,
    )

    print("\n" + "=" * 70)
    print("DATASET INGESTION & QUALITY ASSURANCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
