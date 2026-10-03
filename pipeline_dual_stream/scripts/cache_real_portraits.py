"""Extract and cache DINOv2 and Regional Physics features for CelebA real portraits.
Splits:
- Train: 1,600 real photographic portraits
- Held-out Test: 400 real photographic portraits
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if "G:/Thesis" not in sys.path:
    sys.path.append("G:/Thesis")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_40k.src.extractors.surface_normals import SurfaceNormalsExtractor
from pipeline_40k.src.extractors.regional_physics import RegionalPhysicsExtractor
from src.models.dinov2_stream import DINOv2Stream


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = ROOT_DIR / "data" / "universal_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cache_path = output_dir / "celeba_portraits_train.pt"
    test_cache_path = output_dir / "celeba_portraits_test.pt"

    # Read parquet table
    parquet_path = ROOT_DIR / "data" / "celeba_train.parquet"
    print(f"Reading parquet from {parquet_path}...")
    table = pq.read_table(str(parquet_path))
    num_total = 2000
    n_train = 1600
    n_test = 400

    print(f"Selecting {num_total} samples: {n_train} train, {n_test} held-out test.")

    # 1. Initialize extractors
    print("Initializing DINOv2 Stream...")
    dino_stream = DINOv2Stream(
        model_name="vit_base_patch14_dinov2",
        freeze_backbone=True,
        img_size=224,
    ).to(device)
    dino_stream.eval()

    dino_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Initializing Regional Physics Extractor (DSINE + SH + Optics)...")
    normals = SurfaceNormalsExtractor(
        normal_backend="dsine",
        dsine_checkpoint="G:/Thesis/pipeline_40k/models/normal_estimators/dsine/exp002_kappa/dsine.pt",
        dsine_device=device,
    )
    phys_extractor = RegionalPhysicsExtractor(normals=normals)

    # 2. Extract features
    all_phys_feats = []
    all_phys_confs = []
    all_dino_cls = []
    all_dino_reg = []
    all_labels = []

    t0 = time.time()
    batch_imgs_dino = []
    batch_indices = []

    for idx in range(num_total):
        img_dict = table["image"][idx].as_py()
        pil_img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
        arr = np.array(pil_img)

        # Physics extraction
        p_feat, p_conf = phys_extractor.extract(arr)
        all_phys_feats.append(p_feat)
        all_phys_confs.append(p_conf)
        all_labels.append(0.0)  # Real camera photos

        # Prep for DINOv2
        t_img = dino_transform(pil_img)
        batch_imgs_dino.append(t_img)
        batch_indices.append(idx)

        # Batch DINOv2 forward pass every 32 samples or at end
        if len(batch_imgs_dino) == 32 or idx == num_total - 1:
            batch_tensor = torch.stack(batch_imgs_dino).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                    cls_tok, patch_tok = dino_stream.extract_patch_tokens(batch_tensor)
                    quad_tok = dino_stream.pool_quadrants(patch_tok)
                    reg_tok = torch.cat([cls_tok.unsqueeze(1), quad_tok], dim=1)

            all_dino_cls.append(cls_tok.cpu().float())
            all_dino_reg.append(reg_tok.cpu().float())
            batch_imgs_dino = []
            batch_indices = []

        if (idx + 1) % 100 == 0 or (idx + 1) == num_total:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            rem = (num_total - (idx + 1)) / max(rate, 1e-4)
            print(f"[{idx+1}/{num_total}] Extracted features at {rate:.2f} img/s. ETA: {rem:.1f}s")

    # Stack all tensors
    phys_feats = torch.stack(all_phys_feats, dim=0).float()
    phys_confs = torch.stack(all_phys_confs, dim=0).float()
    dino_cls = torch.cat(all_dino_cls, dim=0).float()
    dino_reg = torch.cat(all_dino_reg, dim=0).float()
    labels = torch.tensor(all_labels, dtype=torch.float32)

    print("Extraction complete!")
    print(f"Shapes -> Phys: {phys_feats.shape}, DINO CLS: {dino_cls.shape}, DINO Reg: {dino_reg.shape}")

    # Split into train and held-out test
    train_dict = {
        "physics_features": phys_feats[:n_train],
        "physics_confidences": phys_confs[:n_train],
        "dinov2_cls": dino_cls[:n_train],
        "dinov2_regional": dino_reg[:n_train],
        "labels": labels[:n_train],
        "source": "CelebA_real_portraits_train",
    }
    test_dict = {
        "physics_features": phys_feats[n_train:],
        "physics_confidences": phys_confs[n_train:],
        "dinov2_cls": dino_cls[n_train:],
        "dinov2_regional": dino_reg[n_train:],
        "labels": labels[n_train:],
        "source": "CelebA_real_portraits_held_out_test",
    }

    torch.save(train_dict, train_cache_path)
    torch.save(test_dict, test_cache_path)
    print(f"Saved {n_train} train portraits to {train_cache_path}")
    print(f"Saved {n_test} held-out test portraits to {test_cache_path}")


if __name__ == "__main__":
    main()
