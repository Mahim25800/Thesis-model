"""Abstract Base Class for Physics-Based Invariant Discrepancy Extractors."""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
import torch


class BasePhysicsExtractor(ABC):
    """Base class for all physics feature extractors.

    Every extractor receives an RGB image and outputs:
    1. A fixed-dimension discrepancy vector: torch.Tensor of shape [feature_dim]
    2. A scalar confidence score: float in [0.0, 1.0]
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    @abstractmethod
    def extract(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        """Extracts physical discrepancy vector and confidence score.

        Args:
            image: RGB image as np.ndarray uint8 [H, W, 3] or torch.Tensor [3, H, W]

        Returns:
            discrepancy: torch.Tensor [feature_dim]
            confidence: float in [0.0, 1.0]
        """
        pass

    def __call__(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        return self.extract(image)
