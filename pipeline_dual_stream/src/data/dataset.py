"""Dual-Stream Paired Dataset and DataLoaders.
Loads aligned pairs of vision foundation tokens (DINOv2) and regional physics features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class DualStreamCachedDataset(Dataset):
    """Dataset loading precomputed DINOv2 tokens aligned with physics features."""

    def __init__(
        self,
        physics_cache_path: Union[str, Path],
        dinov2_cache_path: Optional[Union[str, Path]] = None,
        indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        # Load physics cache
        phys_data = torch.load(str(physics_cache_path), map_location="cpu", weights_only=True)
        self.physics_features = phys_data["features"]        # [N, 5, 14]
        self.physics_confidences = phys_data["confidences"]  # [N, 5, 4]
        self.labels = phys_data["labels"].float()             # [N]

        # Load DINOv2 cache if provided
        if dinov2_cache_path is not None and Path(dinov2_cache_path).exists():
            dino_data = torch.load(str(dinov2_cache_path), map_location="cpu", weights_only=True)
            self.dinov2_cls = dino_data["dinov2_cls"]            # [N, 768]
            self.dinov2_regional = dino_data["dinov2_regional"]  # [N, 5, 768]
            if len(self.dinov2_cls) != len(self.labels):
                raise ValueError(
                    f"Mismatch: physics has {len(self.labels)} rows, dinov2 has {len(self.dinov2_cls)}"
                )
        else:
            self.dinov2_cls = None
            self.dinov2_regional = None

        # Filter by indices if specified
        if indices is not None:
            indices = torch.as_tensor(indices, dtype=torch.long)
            self.physics_features = self.physics_features[indices]
            self.physics_confidences = self.physics_confidences[indices]
            self.labels = self.labels[indices]
            if self.dinov2_cls is not None:
                self.dinov2_cls = self.dinov2_cls[indices]
                self.dinov2_regional = self.dinov2_regional[indices]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "physics_features": self.physics_features[idx],
            "physics_confidences": self.physics_confidences[idx],
            "label": self.labels[idx],
        }
        if self.dinov2_cls is not None:
            item["dinov2_cls"] = self.dinov2_cls[idx]
            item["dinov2_regional"] = self.dinov2_regional[idx]
        return item


def create_group_disjoint_split(
    manifest_path: Union[str, Path],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create train and validation split indices disjoint by difference_hash_64 groups."""
    groups = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            grp = record.get("difference_hash_64", str(idx))
            groups.setdefault(grp, []).append(idx)

    rng = np.random.default_rng(seed)
    unique_groups = list(groups.keys())
    rng.shuffle(unique_groups)

    val_count = int(len(unique_groups) * val_ratio)
    val_groups = set(unique_groups[:val_count])
    train_groups = set(unique_groups[val_count:])

    train_indices = [idx for grp in train_groups for idx in groups[grp]]
    val_indices = [idx for grp in val_groups for idx in groups[grp]]

    return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)
