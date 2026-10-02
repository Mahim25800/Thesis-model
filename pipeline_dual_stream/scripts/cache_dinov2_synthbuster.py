"""Extract and cache DINOv2 foundation embeddings for RAISE and all Synthbuster generators.
Produces [N, 768] CLS tokens and [N, 5, 768] quadrant-pooled regional tokens matching Regional Physics.
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


class SynthbusterDataset(Dataset):
    """Loads images matching a Synthbuster JSONL manifest."""

    def __init__(self, img_dir: Path, manifest_path: Path, label: int, transform: transforms.Compose):
        self.samples: List[Tuple[Path, int]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                fn = rec.get("filename")
                p = img_dir / fn
                self.samples.append((p, label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        p, label = self.samples[idx]
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                tensor = self.transform(img_rgb)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            tensor = torch.zeros(3, 224, 224)
        return tensor, label, str(p)


def extract_for_dataset(
    stream: DINOv2Stream,
    dataset: Dataset,
    output_path: Path,
    device: str,
    batch_size: int = 64,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    cls_list, reg_list, lbl_list = [], [], []

    with torch.no_grad():
        for batch_imgs, batch_labels, _ in tqdm(loader, desc=output_path.stem):
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                cls_tok, patch_tok = stream.extract_patch_tokens(batch_imgs)
                quad_tok = stream.pool_quadrants(patch_tok)
                reg_tok = torch.cat([cls_tok.unsqueeze(1), quad_tok], dim=1)

            cls_list.append(cls_tok.cpu().float())
            reg_list.append(reg_tok.cpu().float())
            lbl_list.append(batch_labels)

    all_cls = torch.cat(cls_list, dim=0)
    all_reg = torch.cat(reg_list, dim=0)
    all_lbl = torch.cat(lbl_list, dim=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dinov2_cls": all_cls,
            "dinov2_regional": all_reg,
            "labels": all_lbl,
            "sample_count": len(all_lbl),
            "timestamp": time.time(),
        },
        output_path,
    )
    print(f"Saved {len(all_lbl)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading DINOv2 backbone on {args.device}...")
    stream = DINOv2Stream(pretrained=True, freeze_backbone=True, img_size=224).to(args.device)
    stream.eval()

    synth_base = Path("G:/Thesis/pipeline_40k/data/external_synthbuster")
    cache_base = Path("G:/Thesis/pipeline_40k/data/cache_synthbuster_regional_v2")
    out_dir = Path("G:/Thesis/pipeline_dual_stream/data/synthbuster_dino")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "name": "raise_real",
            "img_dir": synth_base / "raise/real_RAISE_1k",
            "manifest": cache_base / "raise_real.jsonl",
            "label": 0,
            "out": out_dir / "dinov2_cache_raise_real.pt",
        },
        {
            "name": "dalle2",
            "img_dir": synth_base / "synthbuster/synthbuster/dalle2",
            "manifest": cache_base / "dalle2.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_dalle2.pt",
        },
        {
            "name": "dalle3",
            "img_dir": synth_base / "synthbuster/synthbuster/dalle3",
            "manifest": cache_base / "dalle3.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_dalle3.pt",
        },
        {
            "name": "firefly",
            "img_dir": synth_base / "synthbuster/synthbuster/firefly",
            "manifest": cache_base / "firefly.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_firefly.pt",
        },
        {
            "name": "glide",
            "img_dir": synth_base / "synthbuster/synthbuster/glide",
            "manifest": cache_base / "glide.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_glide.pt",
        },
        {
            "name": "midjourney-v5",
            "img_dir": synth_base / "synthbuster/synthbuster/midjourney-v5",
            "manifest": cache_base / "midjourney-v5.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_midjourney_v5.pt",
        },
        {
            "name": "stable-diffusion-1-3",
            "img_dir": synth_base / "synthbuster/synthbuster/stable-diffusion-1-3",
            "manifest": cache_base / "stable-diffusion-1-3.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_sd13.pt",
        },
        {
            "name": "stable-diffusion-1-4",
            "img_dir": synth_base / "synthbuster/synthbuster/stable-diffusion-1-4",
            "manifest": cache_base / "stable-diffusion-1-4.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_sd14.pt",
        },
        {
            "name": "stable-diffusion-2",
            "img_dir": synth_base / "synthbuster/synthbuster/stable-diffusion-2",
            "manifest": cache_base / "stable-diffusion-2.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_sd2.pt",
        },
        {
            "name": "stable-diffusion-xl",
            "img_dir": synth_base / "synthbuster/synthbuster/stable-diffusion-xl",
            "manifest": cache_base / "stable-diffusion-xl.jsonl",
            "label": 1,
            "out": out_dir / "dinov2_cache_sdxl.pt",
        },
    ]

    for t in tasks:
        if t["out"].exists():
            print(f"Skipping {t['name']}, already cached at {t['out']}.")
            continue
        print(f"\nProcessing {t['name']}...")
        ds = SynthbusterDataset(t["img_dir"], t["manifest"], t["label"], preprocess)
        extract_for_dataset(stream, ds, t["out"], args.device, args.batch_size)

    print("\nAll Synthbuster and RAISE DINOv2 caches complete!")


if __name__ == "__main__":
    main()
