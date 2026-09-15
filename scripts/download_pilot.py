"""Downloads and builds genuine pilot dataset from authentic photographic and AI-generated sources.

- Authentic Real Camera Photos: MS-COCO (detection-datasets/coco, val split)
- Authentic AI-Generated Images: DiffusionDB (Stable Diffusion generations via poloclub/diffusiondb or svjack/diffusiondb_random_10k)

Partitions:
- data/pilot/train/real: 400 images
- data/pilot/train/fake: 400 images
- data/pilot/test_unseen/real: 100 images
- data/pilot/test_unseen/fake: 100 images
"""

import os
import sys
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.preprocessor import save_verified_jpeg


def setup_dirs(base: str = "data/pilot"):
    base_path = Path(base)
    for split in ["train/real", "train/fake", "test_unseen/real", "test_unseen/fake"]:
        (base_path / split).mkdir(parents=True, exist_ok=True)


def build_genuine_pilot(
    base: str = "data/pilot",
    train_real: int = 400,
    train_fake: int = 400,
    test_real: int = 100,
    test_fake: int = 100,
):
    setup_dirs(base)
    total_real = train_real + test_real
    total_fake = train_fake + test_fake

    # 1. Authentic Real Camera Images (COCO Val 2017)
    print(f"\n--- [1/2] Fetching authentic real camera photos (COCO, target: {total_real}) ---")
    coco = load_dataset("detection-datasets/coco", split="val", streaming=True)
    real_count = 0

    with tqdm(total=total_real, desc="Downloading Real Images") as pbar:
        for sample in coco:
            img = sample["image"].convert("RGB")
            if real_count < train_real:
                dest = Path(base) / f"train/real/real_{real_count:05d}.jpg"
            elif real_count < total_real:
                dest = Path(base) / f"test_unseen/real/real_{(real_count - train_real):05d}.jpg"
            else:
                break

            save_verified_jpeg(img, dest)
            real_count += 1
            pbar.update(1)
            if real_count >= total_real:
                break

    # 2. Authentic AI-Generated Images (Stable Diffusion / DiffusionDB)
    print(f"\n--- [2/2] Fetching genuine AI-generated images (DiffusionDB, target: {total_fake}) ---")
    try:
        ai_dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k", split="train", streaming=True)
        iter(ai_dataset)
    except Exception as e:
        print(f"Direct poloclub/diffusiondb script not supported in datasets library ({e}).")
        print("Using standard Parquet stream from svjack/diffusiondb_random_10k...")
        ai_dataset = load_dataset("svjack/diffusiondb_random_10k", split="train", streaming=True)

    fake_count = 0
    with tqdm(total=total_fake, desc="Downloading Fake Images") as pbar:
        for sample in ai_dataset:
            img = sample["image"].convert("RGB")
            if fake_count < train_fake:
                dest = Path(base) / f"train/fake/fake_{fake_count:05d}.jpg"
            elif fake_count < total_fake:
                dest = Path(base) / f"test_unseen/fake/fake_{(fake_count - train_fake):05d}.jpg"
            else:
                break

            save_verified_jpeg(img, dest)
            fake_count += 1
            pbar.update(1)
            if fake_count >= total_fake:
                break

    print("\nDataset successfully built with genuine camera photos and AI generations.")

    # Validation
    split_targets = {
        "train/real": train_real,
        "train/fake": train_fake,
        "test_unseen/real": test_real,
        "test_unseen/fake": test_fake,
    }
    for split, expected in split_targets.items():
        split_dir = Path(base) / split
        imgs = list(split_dir.glob("*.jpg"))
        print(f"Verified {split}: {len(imgs)}/{expected} intact images.")
        assert len(imgs) >= expected, f"Expected {expected} images in {split}, got {len(imgs)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and build authentic pilot dataset")
    parser.add_argument("--base-dir", type=str, default="data/pilot")
    parser.add_argument("--train-real", type=int, default=400)
    parser.add_argument("--train-fake", type=int, default=400)
    parser.add_argument("--test-real", type=int, default=100)
    parser.add_argument("--test-fake", type=int, default=100)
    parser.add_argument("--scale", type=int, default=None, help="Convenience flag: scales train real and fake to N")
    args = parser.parse_args()

    train_r = args.scale if args.scale else args.train_real
    train_f = args.scale if args.scale else args.train_fake

    pilot_base = str(WORKSPACE_ROOT / args.base_dir)
    build_genuine_pilot(
        pilot_base,
        train_real=train_r,
        train_fake=train_f,
        test_real=args.test_real,
        test_fake=args.test_fake,
    )
