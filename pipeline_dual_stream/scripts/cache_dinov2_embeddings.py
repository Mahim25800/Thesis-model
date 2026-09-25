"""Pre-extract and cache DINOv2 foundation embeddings for fast training & evaluation.
Extracts global CLS token and 4-quadrant pooled regional tokens aligned with Regional Physics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.models.dinov2_stream import DINOv2Stream


class ManifestImageDataset(Dataset):
    """Loads raw images according to an existing manifest JSONL."""

    def __init__(
        self,
        manifest_path: Path,
        base_dir: Path,
        transform: transforms.Compose,
        max_samples: int = 0,
    ):
        self.samples: List[Tuple[Path, int]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if 0 < max_samples <= idx:
                    break
                record = json.loads(line)
                rel_path = record["relative_path"]
                full_path = base_dir / rel_path
                self.samples.append((full_path, record["label"]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                tensor = self.transform(img_rgb)
        except Exception as e:
            # Fallback for corrupted image (should not happen in curated sets)
            print(f"Error loading {path}: {e}")
            tensor = torch.zeros(3, 224, 224)
        return tensor, label, str(path)


def main():
    parser = argparse.ArgumentParser(description="Cache DINOv2 features for dataset")
    parser.add_argument(
        "--manifest",
        type=str,
        default="G:/Thesis/pipeline_40k/data/cache_regional_dsine_v2/train_manifest.jsonl",
        help="Path to manifest JSONL",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="G:/Thesis/pipeline_40k",
        help="Base root directory for relative paths",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/data/dinov2_cache_train.pt",
        help="Output .pt cache path",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for extraction")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = Path(args.base_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Initializing DINOv2 model on {args.device}...")
    stream = DINOv2Stream(pretrained=True, freeze_backbone=True, img_size=224).to(args.device)
    stream.eval()

    # Preprocessing transform
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading manifest from {manifest_path}...")
    dataset = ManifestImageDataset(manifest_path, base_dir, preprocess, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cls_tokens_list = []
    regional_tokens_list = []
    labels_list = []

    print(f"Extracting DINOv2 features for {len(dataset)} images...")
    start_time = time.time()

    with torch.no_grad():
        for batch_imgs, batch_labels, _ in tqdm(dataloader, desc="DINOv2 Caching"):
            batch_imgs = batch_imgs.to(args.device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda" if "cuda" in args.device else "cpu"):
                cls_token, patch_tokens = stream.extract_patch_tokens(batch_imgs)
                quad_tokens = stream.pool_quadrants(patch_tokens)
                regional_tokens = torch.cat([cls_token.unsqueeze(1), quad_tokens], dim=1)

            cls_tokens_list.append(cls_token.cpu().float())
            regional_tokens_list.append(regional_tokens.cpu().float())
            labels_list.append(batch_labels)

    all_cls = torch.cat(cls_tokens_list, dim=0)
    all_reg = torch.cat(regional_tokens_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.1f}s ({len(dataset)/elapsed:.1f} img/s)")
    print(f"Saving cache to {output_path}...")
    torch.save(
        {
            "dinov2_cls": all_cls,
            "dinov2_regional": all_reg,
            "labels": all_labels,
            "provenance": {
                "manifest": str(manifest_path),
                "model": "vit_base_patch14_dinov2",
                "sample_count": len(all_labels),
                "timestamp": time.time(),
            },
        },
        output_path,
    )
    print(f"Successfully saved {len(all_labels)} embeddings to {output_path}")


if __name__ == "__main__":
    main()
