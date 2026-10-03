"""Extract and cache DINOv2 and Regional Physics features for the Chameleon dataset.
Supports shard-based incremental caching with automatic resume.
Extracts:
  - Regional Physics features: [N, 5, 14]
  - Regional Physics confidences: [N, 5, 4]
  - DINOv2 CLS tokens: [N, 768]
  - DINOv2 Regional tokens: [N, 5, 768]
  - Ground truth labels: [N] (0.0 = Real Camera, 1.0 = AI-Generated)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if "G:/Thesis" not in sys.path:
    sys.path.append("G:/Thesis")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from pipeline_40k.src.extractors.surface_normals import SurfaceNormalsExtractor
from pipeline_40k.src.extractors.regional_physics import RegionalPhysicsExtractor
from src.models.dinov2_stream import DINOv2Stream


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_chameleon_samples(
    data_dir: Path,
    limit_per_class: int | None = None,
    seed: int = 42,
) -> List[Tuple[Path, float]]:
    """Collect (path, label) pairs for real and fake classes in Chameleon."""
    real_dir = data_dir / "0_real"
    fake_dir = data_dir / "1_fake"

    if not real_dir.is_dir() or not fake_dir.is_dir():
        # Check if nested inside Chameleon/Chameleon/test
        alt_real = data_dir / "Chameleon" / "test" / "0_real"
        alt_fake = data_dir / "Chameleon" / "test" / "1_fake"
        if alt_real.is_dir() and alt_fake.is_dir():
            real_dir = alt_real
            fake_dir = alt_fake
        else:
            raise FileNotFoundError(f"Cannot find 0_real and 1_fake in {data_dir}")

    real_files = sorted([p for p in real_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    fake_files = sorted([p for p in fake_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images in {data_dir}.")

    if limit_per_class is not None:
        rng = np.random.RandomState(seed)
        if len(real_files) > limit_per_class:
            idx_r = rng.choice(len(real_files), size=limit_per_class, replace=False)
            real_files = [real_files[i] for i in sorted(idx_r)]
        if len(fake_files) > limit_per_class:
            idx_f = rng.choice(len(fake_files), size=limit_per_class, replace=False)
            fake_files = [fake_files[i] for i in sorted(idx_f)]
        print(f"Subsampled to {len(real_files)} real and {len(fake_files)} fake images (limit_per_class={limit_per_class}).")

    samples: List[Tuple[Path, float]] = []
    # Interleave real and fake evenly so shards contain both classes
    max_len = max(len(real_files), len(fake_files))
    for i in range(max_len):
        if i < len(real_files):
            samples.append((real_files[i], 0.0))
        if i < len(fake_files):
            samples.append((fake_files[i], 1.0))

    return samples


def process_shard(
    shard_samples: List[Tuple[Path, float]],
    phys_extractor: RegionalPhysicsExtractor,
    dino_stream: DINOv2Stream,
    dino_transform: transforms.Compose,
    device: str,
    dino_batch_size: int = 32,
) -> dict:
    """Extract physics and DINOv2 features for a list of samples."""
    phys_feats, phys_confs, labels, valid_paths = [], [], [], []
    dino_tensors = []

    for img_path, label in shard_samples:
        try:
            with Image.open(img_path) as pil:
                if max(pil.size) > 1600:
                    pil.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                pil_rgb = pil.convert("RGB")
                arr = np.array(pil_rgb)
                p_feat, p_conf = phys_extractor.extract(arr)
                t_img = dino_transform(pil_rgb)

            phys_feats.append(p_feat)
            phys_confs.append(p_conf)
            labels.append(label)
            valid_paths.append(str(img_path.name))
            dino_tensors.append(t_img)
        except (Exception, MemoryError) as e:
            print(f"Warning: Failed processing {img_path}: {e}")
            continue

    if not dino_tensors:
        raise RuntimeError("No images were successfully processed in shard")

    # Batched DINOv2 extraction
    dino_cls_list, dino_reg_list = [], []
    n = len(dino_tensors)
    for i in range(0, n, dino_batch_size):
        batch = torch.stack(dino_tensors[i : i + dino_batch_size]).to(device)
        with torch.no_grad():
            cls_tok, patch_tok = dino_stream.extract_patch_tokens(batch)
            quad_tok = dino_stream.pool_quadrants(patch_tok)
            reg_tok = torch.cat([cls_tok.unsqueeze(1), quad_tok], dim=1)

        dino_cls_list.append(cls_tok.cpu().float())
        dino_reg_list.append(reg_tok.cpu().float())

    return {
        "physics_features": torch.stack(phys_feats).float(),
        "physics_confidences": torch.stack(phys_confs).float(),
        "dinov2_cls": torch.cat(dino_cls_list, dim=0),
        "dinov2_regional": torch.cat(dino_reg_list, dim=0),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "filenames": valid_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract feature cache for Chameleon dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/Chameleon/Chameleon/test",
        help="Path to Chameleon test folder containing 0_real and 1_fake",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/universal_v4/chameleon",
        help="Output folder for shards and final cache",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Limit samples per class for quick testing (None = all)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="Number of samples per shard checkpoint",
    )
    parser.add_argument(
        "--dino-batch-size",
        type=int,
        default=32,
        help="Batch size for DINOv2 ViT backbone",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CHAMELEON FEATURE CACHE BUILDER")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Shard size: {args.shard_size}")
    print(f"Limit per class: {args.limit_per_class}")
    print("=" * 80)

    samples = find_chameleon_samples(data_dir, limit_per_class=args.limit_per_class)
    total_samples = len(samples)
    num_shards = (total_samples + args.shard_size - 1) // args.shard_size
    print(f"Total samples to process: {total_samples} across {num_shards} shards.")

    # Check which shards are already complete
    completed_shards = []
    shards_to_process = []
    for s_idx in range(num_shards):
        s_path = shards_dir / f"shard_{s_idx:04d}.pt"
        if s_path.is_file():
            completed_shards.append(s_idx)
        else:
            shards_to_process.append(s_idx)

    print(f"Shards already cached: {len(completed_shards)} / {num_shards}")
    print(f"Shards remaining to process: {len(shards_to_process)}")

    if shards_to_process:
        # Initialize extractors
        print("\nLoading DINOv2 Stream...")
        dino_stream = DINOv2Stream(
            model_name="vit_base_patch14_dinov2",
            freeze_backbone=True,
            img_size=224,
        ).to(args.device)
        dino_stream.eval()

        dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("Loading Regional Physics Extractor (DSINE + SH + Optics)...")
        normals = SurfaceNormalsExtractor(
            normal_backend="dsine",
            dsine_checkpoint="G:/Thesis/pipeline_40k/models/normal_estimators/dsine/exp002_kappa/dsine.pt",
            dsine_device=args.device,
        )
        phys_extractor = RegionalPhysicsExtractor(normals=normals)

        t_start = time.time()
        for idx_in_todo, s_idx in enumerate(shards_to_process):
            s_start = s_idx * args.shard_size
            s_end = min(s_start + args.shard_size, total_samples)
            shard_slice = samples[s_start:s_end]

            print(f"\n--- Shard {s_idx + 1}/{num_shards} [{s_start}:{s_end}] ({len(shard_slice)} images) ---")
            t_shard_0 = time.time()

            shard_data = process_shard(
                shard_slice,
                phys_extractor=phys_extractor,
                dino_stream=dino_stream,
                dino_transform=dino_transform,
                device=args.device,
                dino_batch_size=args.dino_batch_size,
            )

            s_path = shards_dir / f"shard_{s_idx:04d}.pt"
            torch.save(shard_data, s_path)

            del shard_data
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            dt_shard = time.time() - t_shard_0
            elapsed = time.time() - t_start
            remaining_shards = len(shards_to_process) - (idx_in_todo + 1)
            eta_s = (elapsed / (idx_in_todo + 1)) * remaining_shards if (idx_in_todo + 1) > 0 else 0
            print(f"Shard {s_idx + 1} completed in {dt_shard:.1f}s ({len(shard_slice)/dt_shard:.2f} img/s). ETA: {eta_s/60:.1f} min")

    # Combine all shards into final cache
    print("\nMerging all shards into unified chameleon_cache.pt...")
    all_pf, all_pc, all_dc, all_dr, all_lbl, all_fn = [], [], [], [], [], []

    for s_idx in range(num_shards):
        s_path = shards_dir / f"shard_{s_idx:04d}.pt"
        if not s_path.is_file():
            raise FileNotFoundError(f"Missing shard {s_path}")
        s_data = torch.load(s_path, map_location="cpu", weights_only=True)
        all_pf.append(s_data["physics_features"])
        all_pc.append(s_data["physics_confidences"])
        all_dc.append(s_data["dinov2_cls"])
        all_dr.append(s_data["dinov2_regional"])
        all_lbl.append(s_data["labels"])
        all_fn.extend(s_data["filenames"])

    combined = {
        "physics_features": torch.cat(all_pf, dim=0),
        "physics_confidences": torch.cat(all_pc, dim=0),
        "dinov2_cls": torch.cat(all_dc, dim=0),
        "dinov2_regional": torch.cat(all_dr, dim=0),
        "labels": torch.cat(all_lbl, dim=0),
        "filenames": all_fn,
        "num_samples": sum(len(x) for x in all_lbl),
        "num_real": int((torch.cat(all_lbl, dim=0) == 0.0).sum()),
        "num_fake": int((torch.cat(all_lbl, dim=0) == 1.0).sum()),
    }

    final_cache_path = output_dir / "chameleon_cache.pt"
    torch.save(combined, final_cache_path)
    print(f"\nFinal Chameleon cache saved to {final_cache_path.resolve()}")
    print(f"Total Cached Samples: {combined['num_samples']} (Real: {combined['num_real']}, Fake: {combined['num_fake']})")
    print(f"Feature Shapes:")
    print(f"  physics_features:    {combined['physics_features'].shape}")
    print(f"  physics_confidences: {combined['physics_confidences'].shape}")
    print(f"  dinov2_cls:          {combined['dinov2_cls'].shape}")
    print(f"  dinov2_regional:     {combined['dinov2_regional'].shape}")
    print(f"  labels:              {combined['labels'].shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()
