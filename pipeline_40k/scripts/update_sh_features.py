"""Updates cached 40k feature tensors with deterministic Spherical Harmonics (SH) features.

Backs up original tensors to:
  data/cache_scaled/train_scaled_features.pt.bak
  data/cache_scaled/test_scaled_features.pt.bak

Replaces:
  features[:, 0:5]    <- new deterministic 5D delta_light
  confidences[:, 0]   <- new deterministic 1D c_light
Preserving:
  features[:, 5:14]   <- corneal optics, surface normals, chromatic shadow
  confidences[:, 1:4] <- specular, normal, shadow confidences
  labels              <- ground truth labels
"""

import sys
import shutil
import time
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.illumination_sh import IlluminationSHExtractor
from src.data.preprocessor import validate_and_load_image


def get_image_samples(partition_dir: Path) -> List[Path]:
    """Returns sorted list of real image paths followed by fake image paths."""
    real_dir = partition_dir / "real"
    fake_dir = partition_dir / "fake"
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    real_files = sorted([p for p in real_dir.iterdir() if p.suffix.lower() in valid_exts and p.stat().st_size > 100])
    fake_files = sorted([p for p in fake_dir.iterdir() if p.suffix.lower() in valid_exts and p.stat().st_size > 100])

    return real_files + fake_files


def extract_single_image(args: Tuple[int, Path, IlluminationSHExtractor]) -> Tuple[int, torch.Tensor, float]:
    idx, img_path, extractor = args
    try:
        img_np = validate_and_load_image(img_path)
        delta, conf = extractor.extract(img_np)
        return idx, delta, conf
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return idx, torch.zeros(5), 0.05


def update_feature_cache(
    cache_path: Path,
    images_dir: Path,
    extractor: IlluminationSHExtractor,
    num_workers: int = 8,
):
    print(f"\nProcessing cache: {cache_path}")
    print(f"Images directory: {images_dir}")

    # 1. Backup if not already backed up
    bak_path = cache_path.with_suffix(".pt.bak")
    if not bak_path.exists():
        print(f"Creating backup: {bak_path}")
        shutil.copy2(cache_path, bak_path)
    else:
        print(f"Backup already exists at: {bak_path}")

    # 2. Load existing cache
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
    features = cache_data["features"]
    confidences = cache_data["confidences"]
    labels = cache_data["labels"]
    n_samples = len(labels)

    # 3. Scan images
    image_paths = get_image_samples(images_dir)
    if len(image_paths) != n_samples:
        raise ValueError(
            f"Image count mismatch: found {len(image_paths)} images, but cache has {n_samples} samples."
        )

    print(f"Extracting deterministic SH for {n_samples:,} images using {num_workers} threads...")
    work_items = [(i, path, extractor) for i, path in enumerate(image_paths)]

    new_deltas = [None] * n_samples
    new_confs = [None] * n_samples

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, delta, conf in tqdm(
            executor.map(extract_single_image, work_items),
            total=n_samples,
            desc=f"Updating {cache_path.name}",
        ):
            new_deltas[idx] = delta
            new_confs[idx] = conf

    elapsed = time.time() - t0
    print(f"Extraction completed in {elapsed:.2f}s ({elapsed/n_samples*1000:.2f} ms/image).")

    new_deltas_tensor = torch.stack(new_deltas, dim=0).float()
    new_confs_tensor = torch.tensor(new_confs, dtype=torch.float32)

    # 4. Check for NaNs or Infs
    assert not torch.isnan(new_deltas_tensor).any(), "NaN detected in new deltas!"
    assert not torch.isinf(new_deltas_tensor).any(), "Inf detected in new deltas!"
    assert not torch.isnan(new_confs_tensor).any(), "NaN detected in new confidences!"

    # 5. Update columns
    features[:, 0:5] = new_deltas_tensor
    confidences[:, 0] = new_confs_tensor

    # 6. Save updated cache
    torch.save(
        {
            "features": features,
            "confidences": confidences,
            "labels": labels,
        },
        cache_path,
    )
    print(f"Successfully saved updated cache to {cache_path}")
    print(f"Delta Light stats - Mean: {features[:, 0:5].mean(dim=0).tolist()}, Std: {features[:, 0:5].std(dim=0).tolist()}")
    print(f"Conf Light stats  - Mean: {confidences[:, 0].mean().item():.4f}, Std: {confidences[:, 0].std().item():.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Update cache with deterministic SH features")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    train_cache = PROJECT_ROOT / "data/cache_scaled/train_scaled_features.pt"
    test_cache = PROJECT_ROOT / "data/cache_scaled/test_scaled_features.pt"
    train_dir = PROJECT_ROOT / "data/scaled/train"
    test_dir = PROJECT_ROOT / "data/scaled/test_unseen"

    extractor = IlluminationSHExtractor()

    print("=" * 70)
    print("UPDATING 40k DATASET WITH DETERMINISTIC SPHERICAL HARMONICS")
    print("Ramamoorthi & Hanrahan / Basri & Jacobs Closed-Form Analytical SH Fitting")
    print("=" * 70)

    # Update train cache
    update_feature_cache(train_cache, train_dir, extractor, num_workers=args.workers)

    # Update test cache
    update_feature_cache(test_cache, test_dir, extractor, num_workers=args.workers)

    print("\n" + "=" * 70)
    print("ALL CACHED FEATURES SUCCESSFULLY UPDATED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
