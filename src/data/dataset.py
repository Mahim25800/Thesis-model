"""PyTorch Datasets for cached physics discrepancy features and raw pilot images."""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessor import validate_and_load_image


class PhysicsFeatureDataset(Dataset):
    """Dataset for pre-extracted physics discrepancy features and confidence scores.

    Features: Tensor of shape [N, 14]
    Confidences: Tensor of shape [N, 4]
    Labels: Tensor of shape [N] (0 = real, 1 = fake)
    """

    def __init__(
        self,
        features: torch.Tensor,
        confidences: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.features = features.float()
        self.confidences = confidences.float()
        self.labels = labels.float()

        assert (
            len(self.features) == len(self.confidences) == len(self.labels)
        ), "Features, confidences, and labels must have the same length"

    @classmethod
    def from_cache(cls, cache_path: Union[str, Path]) -> "PhysicsFeatureDataset":
        """Loads dataset from a saved .pt dictionary."""
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found at {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            features=data["features"],
            confidences=data["confidences"],
            labels=data["labels"],
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.confidences[idx], self.labels[idx]


class ImagePilotDataset(Dataset):
    """Traverses a pilot directory structured into real/ and fake/ subdirectories."""

    def __init__(self, root_dir: Union[str, Path], target_size: Tuple[int, int] = (256, 256)):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.samples: List[Tuple[Path, int]] = []

        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        if real_dir.exists():
            for p in real_dir.iterdir():
                if p.suffix.lower() in valid_exts:
                    self.samples.append((p, 0))

        if fake_dir.exists():
            for p in fake_dir.iterdir():
                if p.suffix.lower() in valid_exts:
                    self.samples.append((p, 1))

        self.samples.sort(key=lambda x: str(x[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        img_path, label = self.samples[idx]
        img_arr = validate_and_load_image(img_path)
        return img_arr, label, str(img_path)
