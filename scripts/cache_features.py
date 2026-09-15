"""Runs the 4 physics extractors across pilot images and caches features to disk."""

import sys
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.models.pipeline import PhysicsDeepfakePipeline
from src.data.dataset import ImagePilotDataset


def process_and_cache_split(
    split_dir: Path,
    cache_path: Path,
    pipeline: PhysicsDeepfakePipeline,
):
    """Iterates through an image split, runs extractors, and saves cached tensors."""
    dataset = ImagePilotDataset(split_dir)
    print(f"Extracting physics features for {split_dir} ({len(dataset)} images)...")

    features_list = []
    confidences_list = []
    labels_list = []

    for idx in tqdm(range(len(dataset)), desc=f"Caching {split_dir.name}"):
        img_np, label, path_str = dataset[idx]
        delta, confidences, _ = pipeline.extract_features(img_np)

        features_list.append(delta)
        confidences_list.append(confidences)
        labels_list.append(label)

    features_tensor = torch.stack(features_list, dim=0).float()  # [N, 14]
    confidences_tensor = torch.stack(confidences_list, dim=0).float()  # [N, 4]
    labels_tensor = torch.tensor(labels_list, dtype=torch.float32)  # [N]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features_tensor,
            "confidences": confidences_tensor,
            "labels": labels_tensor,
        },
        cache_path,
    )

    print(f"Cached {len(labels_tensor)} items to {cache_path}")
    print(f"  Features shape:    {features_tensor.shape}")
    print(f"  Confidences shape: {confidences_tensor.shape}")
    print(f"  Labels distribution: Real={int((labels_tensor == 0).sum())}, Fake={int((labels_tensor == 1).sum())}")


def main():
    parser = argparse.ArgumentParser(description="Extract and cache physics features")
    parser.add_argument("--pilot-dir", type=str, default="data/pilot")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    args = parser.parse_args()

    pilot_dir = WORKSPACE_ROOT / args.pilot_dir
    cache_dir = WORKSPACE_ROOT / args.cache_dir

    train_dir = pilot_dir / "train"
    test_dir = pilot_dir / "test_unseen"

    train_cache = cache_dir / "train_features.pt"
    test_cache = cache_dir / "test_features.pt"

    pipeline = PhysicsDeepfakePipeline(device="cpu")

    process_and_cache_split(train_dir, train_cache, pipeline)
    process_and_cache_split(test_dir, test_cache, pipeline)

    print("\nFeature caching complete!")


if __name__ == "__main__":
    main()
