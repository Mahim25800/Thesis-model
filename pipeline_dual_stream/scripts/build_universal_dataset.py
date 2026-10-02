"""Build Unified Multi-Generator & Diverse-Real Dataset for Universal Deepfake Detection (dual_stream_v3).
Combines:
  - Scaled COCO + DiffusionDB (32,000)
  - GenImage BigGAN (4,000)
  - GenImage VQDM (4,000)
  - GenImage Wukong (4,000)
  - GenImage ADM (4,000)
  - GenImage Glide (2,000)
  - GenImage SDv4 (2,000)
  - GenImage SDv5 (2,000)
  - RAISE DSLR Pristine/Blurred Real (500 train, 500 held-out test)
Total Train Samples: 54,500 (27,500 Real / 27,000 Fake).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent


def build_unified_corpus(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Assembling Universal Multi-Generator & Diverse-Real Corpus...")
    print("=" * 70)

    # 1. Sources to combine
    sources = [
        {
            "name": "scaled_coco_diffusiondb",
            "phys": "G:/Thesis/pipeline_40k/data/cache_regional_dsine_v2/train_features.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/dinov2_cache_train.pt",
            "slice": slice(None),  # all 32,000
        },
        {
            "name": "genimage_biggan",
            "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_regional_v2/biggan_features.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/dinov2_cache_biggan.pt",
            "slice": slice(None),  # 4,000
        },
        {
            "name": "genimage_vqdm",
            "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_regional_v2/vqdm_features.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/dinov2_cache_vqdm.pt",
            "slice": slice(None),  # 4,000
        },
        {
            "name": "genimage_wukong",
            "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_regional_v2/wukong_features.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/dinov2_cache_wukong.pt",
            "slice": slice(None),  # 4,000
        },
        {
            "name": "genimage_adm",
            "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_regional_v2/adm_features.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/dinov2_cache_adm.pt",
            "slice": slice(None),  # 4,000
        },
        {
            "name": "genimage_glide",
            "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_confirmatory_disjoint_regional_v2/glide_features.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/dinov2_cache_glide.pt",
            "slice": slice(None),  # 2,000
        },
        {
            "name": "genimage_sdv4",
            "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_confirmatory_disjoint_regional_v2/sdv4_features.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/dinov2_cache_sdv4.pt",
            "slice": slice(None),  # 2,000
        },
        {
            "name": "genimage_sdv5",
            "phys": "G:/Thesis/pipeline_40k/data/cache_genimage_confirmatory_disjoint_regional_v2/sdv5_features.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/dinov2_cache_sdv5.pt",
            "slice": slice(None),  # 2,000
        },
        {
            "name": "raise_dslr_train",
            "phys": "G:/Thesis/pipeline_40k/data/cache_synthbuster_regional_v2/raise_real.pt",
            "dino": "G:/Thesis/pipeline_dual_stream/data/synthbuster_dino/dinov2_cache_raise_real.pt",
            "slice": slice(0, 500),  # First 500 for training, remaining 500 for testing
        },
    ]

    all_phys_feats = []
    all_phys_confs = []
    all_dino_cls = []
    all_dino_reg = []
    all_labels = []
    domain_tags = []

    for src in sources:
        name = src["name"]
        sl = src["slice"]
        print(f"Loading {name}...")
        p_data = torch.load(src["phys"], map_location="cpu", weights_only=True)
        d_data = torch.load(src["dino"], map_location="cpu", weights_only=True)

        feats = p_data["features"][sl]
        confs = p_data["confidences"][sl]
        cls_tok = d_data["dinov2_cls"][sl]
        reg_tok = d_data["dinov2_regional"][sl]

        # Handle labels
        if "labels" in p_data:
            lbls = p_data["labels"][sl].float()
        else:
            # raise_real is all label 0
            lbls = torch.zeros(len(feats), dtype=torch.float32)

        # Sanity checks
        assert len(feats) == len(cls_tok) == len(lbls), f"Length mismatch in {name}"

        all_phys_feats.append(feats)
        all_phys_confs.append(confs)
        all_dino_cls.append(cls_tok)
        all_dino_reg.append(reg_tok)
        all_labels.append(lbls)
        domain_tags.extend([name] * len(lbls))

        real_count = int((lbls == 0).sum().item())
        fake_count = int((lbls == 1).sum().item())
        print(f"  -> Added {len(lbls)} samples (Real: {real_count}, Fake: {fake_count})")

    # Concatenate all
    concat_feats = torch.cat(all_phys_feats, dim=0)
    concat_confs = torch.cat(all_phys_confs, dim=0)
    concat_cls = torch.cat(all_dino_cls, dim=0)
    concat_reg = torch.cat(all_dino_reg, dim=0)
    concat_labels = torch.cat(all_labels, dim=0)

    total_samples = len(concat_labels)
    total_real = int((concat_labels == 0).sum().item())
    total_fake = int((concat_labels == 1).sum().item())

    print("\n" + "=" * 70)
    print(f"Total Unified Training Set: {total_samples} samples")
    print(f"  Total Real: {total_real} ({total_real/total_samples*100:.1f}%)")
    print(f"  Total Fake: {total_fake} ({total_fake/total_samples*100:.1f}%)")
    print("=" * 70)

    # Save training dataset
    train_cache_path = output_dir / "universal_train_corpus.pt"
    print(f"Saving unified corpus to {train_cache_path}...")
    torch.save(
        {
            "physics_features": concat_feats,
            "physics_confidences": concat_confs,
            "dinov2_cls": concat_cls,
            "dinov2_regional": concat_reg,
            "labels": concat_labels,
            "domain_tags": domain_tags,
            "sample_count": total_samples,
            "real_count": total_real,
            "fake_count": total_fake,
            "timestamp": time.time(),
        },
        train_cache_path,
    )
    print("Unified corpus successfully saved!")

    # 2. Also save Held-out RAISE test partition (remaining 500 samples)
    p_raise = torch.load("G:/Thesis/pipeline_40k/data/cache_synthbuster_regional_v2/raise_real.pt", map_location="cpu", weights_only=True)
    d_raise = torch.load("G:/Thesis/pipeline_dual_stream/data/synthbuster_dino/dinov2_cache_raise_real.pt", map_location="cpu", weights_only=True)
    test_sl = slice(500, None)
    torch.save(
        {
            "physics_features": p_raise["features"][test_sl],
            "physics_confidences": p_raise["confidences"][test_sl],
            "dinov2_cls": d_raise["dinov2_cls"][test_sl],
            "dinov2_regional": d_raise["dinov2_regional"][test_sl],
            "labels": torch.zeros(len(p_raise["features"][test_sl]), dtype=torch.float32),
            "sample_count": len(p_raise["features"][test_sl]),
        },
        output_dir / "raise_held_out_test.pt",
    )
    print(f"Saved 500 held-out RAISE test samples to {output_dir / 'raise_held_out_test.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/data/universal_v3",
    )
    args = parser.parse_args()
    build_unified_corpus(Path(args.output_dir))
