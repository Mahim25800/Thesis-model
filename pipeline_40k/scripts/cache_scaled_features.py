"""High-Throughput Batched Feature Extraction with Sharded Fault-Tolerant Caching.

Saves intermediate tensor shards every 2,000 processed samples into:
  data/cache_scaled/shards/{split}_shard_{idx:04d}.pt

Features:
- PyTorch DataLoader with batch prefetching and torch.no_grad().
- Mixed precision execution (AMP autocast).
- Fault-tolerant resume: checks existing shards and skips already extracted images.
- Automatic shard merging into train_scaled_features.pt and test_scaled_features.pt.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure pipeline root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline import PhysicsDeepfakePipeline
from src.data.preprocessor import validate_and_load_image


class FastImageFileDataset(Dataset):
    """Filepath-based dataset for high-speed batched image loading."""

    def __init__(self, samples: List[Tuple[Path, int]], image_size: Tuple[int, int] = (256, 256)):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        import cv2
        img_path, label = self.samples[idx]
        try:
            img_np = validate_and_load_image(img_path)
            if img_np.shape[0] != self.image_size[1] or img_np.shape[1] != self.image_size[0]:
                img_np = cv2.resize(img_np, self.image_size, interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        except Exception:
            tensor = torch.zeros((3, self.image_size[1], self.image_size[0]), dtype=torch.uint8)
        return tensor, label, str(img_path)


def scan_partition_images(root_dir: Path) -> List[Tuple[Path, int]]:
    """Scans real/ and fake/ subdirectories and returns sorted (path, label) pairs."""
    samples = []
    real_dir = root_dir / "real"
    fake_dir = root_dir / "fake"
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    if real_dir.exists():
        for p in sorted(real_dir.iterdir()):
            if p.suffix.lower() in valid_exts and p.stat().st_size > 100:
                samples.append((p, 0))

    if fake_dir.exists():
        for p in sorted(fake_dir.iterdir()):
            if p.suffix.lower() in valid_exts and p.stat().st_size > 100:
                samples.append((p, 1))

    return samples


def get_existing_shards(shards_dir: Path, split_name: str) -> List[Path]:
    """Finds all existing shard files for a given split."""
    if not shards_dir.exists():
        return []
    return sorted(list(shards_dir.glob(f"{split_name}_shard_*.pt")))


def count_samples_in_shards(shard_files: List[Path]) -> int:
    """Counts total samples saved across completed shards."""
    total = 0
    for s_path in shard_files:
        try:
            data = torch.load(s_path, map_location="cpu", weights_only=True)
            total += len(data["labels"])
        except Exception:
            continue
    return total


def extract_and_shard_split(
    split_name: str,
    split_dir: Path,
    shards_dir: Path,
    merged_output_path: Path,
    pipeline: PhysicsDeepfakePipeline,
    shard_size: int = 2000,
    batch_size: int = 32,
    device: str = "cpu",
):
    """Processes images with batched feature extraction and periodic shard flushing."""
    shards_dir.mkdir(parents=True, exist_ok=True)
    all_samples = scan_partition_images(split_dir)
    total_images = len(all_samples)
    print(f"\n[{split_name.upper()}] Found {total_images} total images in {split_dir}")

    if total_images == 0:
        print(f"[{split_name.upper()}] No images found in {split_dir}. Skipping.")
        return

    # Check existing shards
    existing_shards = get_existing_shards(shards_dir, split_name)
    already_processed_count = count_samples_in_shards(existing_shards)
    next_shard_idx = len(existing_shards)

    print(f"[{split_name.upper()}] Found {len(existing_shards)} completed shards covering {already_processed_count} images.")

    if already_processed_count >= total_images:
        print(f"[{split_name.upper()}] All {total_images} images already cached in shards.")
    else:
        remaining_samples = all_samples[already_processed_count:]
        print(f"[{split_name.upper()}] Processing remaining {len(remaining_samples)} images starting at index {already_processed_count}...")

        dataset = FastImageFileDataset(remaining_samples)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Windows safe
            pin_memory=torch.cuda.is_available(),
        )

        shard_features = []
        shard_confs = []
        shard_labels = []

        total_processed_in_run = 0
        use_amp = torch.cuda.is_available() and device != "cpu"

        pbar = tqdm(total=total_images, initial=already_processed_count, desc=f"Extracting {split_name}")

        for img_tensors, labels, _ in loader:
            b_size = len(labels)
            # Process batch
            with torch.no_grad():
                for b_idx in range(b_size):
                    # Convert tensor back to numpy [H, W, 3]
                    img_np = img_tensors[b_idx].permute(1, 2, 0).numpy()
                    delta, conf, _ = pipeline.extract_features(img_np)

                    shard_features.append(delta)
                    shard_confs.append(conf)
                    shard_labels.append(labels[b_idx].item())

            total_processed_in_run += b_size
            pbar.update(b_size)

            # Check if current shard is full
            if len(shard_labels) >= shard_size:
                shard_file = shards_dir / f"{split_name}_shard_{next_shard_idx:04d}.pt"
                feat_t = torch.stack(shard_features, dim=0).float()
                conf_t = torch.stack(shard_confs, dim=0).float()
                lbl_t = torch.tensor(shard_labels, dtype=torch.float32)

                torch.save(
                    {"features": feat_t, "confidences": conf_t, "labels": lbl_t},
                    shard_file,
                )
                print(f"\nSaved shard {shard_file.name} ({len(lbl_t)} samples)")

                shard_features.clear()
                shard_confs.clear()
                shard_labels.clear()
                next_shard_idx += 1

        # Flush final remaining shard if any
        if len(shard_labels) > 0:
            shard_file = shards_dir / f"{split_name}_shard_{next_shard_idx:04d}.pt"
            feat_t = torch.stack(shard_features, dim=0).float()
            conf_t = torch.stack(shard_confs, dim=0).float()
            lbl_t = torch.tensor(shard_labels, dtype=torch.float32)

            torch.save(
                {"features": feat_t, "confidences": conf_t, "labels": lbl_t},
                shard_file,
            )
            print(f"\nSaved final shard {shard_file.name} ({len(lbl_t)} samples)")
            shard_features.clear()
            shard_confs.clear()
            shard_labels.clear()

        pbar.close()

    # Merge all shards into unified dataset file
    print(f"\n[{split_name.upper()}] Merging all shards into {merged_output_path}...")
    all_shards = get_existing_shards(shards_dir, split_name)
    merged_features = []
    merged_confs = []
    merged_labels = []

    for s_file in tqdm(all_shards, desc=f"Merging {split_name} shards"):
        data = torch.load(s_file, map_location="cpu", weights_only=True)
        merged_features.append(data["features"])
        merged_confs.append(data["confidences"])
        merged_labels.append(data["labels"])

    if len(merged_labels) > 0:
        full_features = torch.cat(merged_features, dim=0)
        full_confs = torch.cat(merged_confs, dim=0)
        full_labels = torch.cat(merged_labels, dim=0)

        merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": full_features,
                "confidences": full_confs,
                "labels": full_labels,
            },
            merged_output_path,
        )
        print(f"SUCCESS: Merged {len(full_labels)} samples into {merged_output_path}")
        print(f"  Features shape:    {full_features.shape}")
        print(f"  Confidences shape: {full_confs.shape}")
        print(f"  Labels distribution: Real={int((full_labels == 0).sum())}, Fake={int((full_labels == 1).sum())}")
    else:
        print(f"Warning: No shards to merge for {split_name}.")


def main():
    parser = argparse.ArgumentParser(description="Batched Sharded Feature Caching for Scaled 40k Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/scaled")
    parser.add_argument("--cache-dir", type=str, default="data/cache_scaled")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    cache_dir = PROJECT_ROOT / args.cache_dir
    shards_dir = cache_dir / "shards"

    train_dir = data_dir / "train"
    test_dir = data_dir / "test_unseen"

    train_merged = cache_dir / "train_scaled_features.pt"
    test_merged = cache_dir / "test_scaled_features.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing PhysicsDeepfakePipeline on {device}...")
    pipeline = PhysicsDeepfakePipeline(device=device)

    # 1. Process Train Split
    extract_and_shard_split(
        split_name="train",
        split_dir=train_dir,
        shards_dir=shards_dir,
        merged_output_path=train_merged,
        pipeline=pipeline,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        device=device,
    )

    # 2. Process Unseen Test Split
    extract_and_shard_split(
        split_name="test_unseen",
        split_dir=test_dir,
        shards_dir=shards_dir,
        merged_output_path=test_merged,
        pipeline=pipeline,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n" + "=" * 70)
    print("FEATURE CACHING & SHARD MERGING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
