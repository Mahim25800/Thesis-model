"""Abstract Base Class and image preprocessing for Physics-Based Extractors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image
import cv2
import torch


def validate_and_load_image(
    image_input: Union[str, Path, Image.Image, np.ndarray]
) -> np.ndarray:
    """Validates and loads an image as an intact RGB NumPy array uint8 [H, W, 3]."""
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
        with Image.open(img_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            img_arr = np.array(rgb_img, dtype=np.uint8)
    elif isinstance(image_input, Image.Image):
        rgb_img = image_input.convert("RGB")
        img_arr = np.array(rgb_img, dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img_arr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:
                img_arr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.shape[2] == 3:
                img_arr = image_input.copy()
            else:
                raise ValueError(f"Unsupported channel count: {image_input.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {image_input.shape}")
        if img_arr.dtype != np.uint8:
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_arr = img_arr.clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    return img_arr


class BasePhysicsExtractor(ABC):
    """Base class for all physics feature extractors.

    Output format:
    1. A fixed-dimension discrepancy vector: torch.Tensor of shape [feature_dim]
    2. A scalar confidence score: float in [0.0, 1.0]
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    @abstractmethod
    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        pass

    def __call__(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        return self.extract(image)
