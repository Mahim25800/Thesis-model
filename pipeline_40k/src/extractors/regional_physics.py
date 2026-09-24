"""Localized multi-physics descriptors with one shared DSINE prediction."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from .chromatic_shadow import ChromaticShadowExtractor
from .corneal_optics import CornealOpticsExtractor
from .illumination_sh import IlluminationSHExtractor
from .surface_normals import SurfaceNormalsExtractor
from ..data.preprocessor import validate_and_load_image


REGION_NAMES = ("global", "top_left", "top_right", "bottom_left", "bottom_right")


def quadrant_boxes(height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Return full-image and four non-empty quadrants in a stable order."""
    if height < 8 or width < 8:
        raise ValueError("Regional physics requires an image of at least 8x8 pixels")
    middle_y, middle_x = height // 2, width // 2
    return [
        (0, height, 0, width),
        (0, middle_y, 0, middle_x),
        (0, middle_y, middle_x, width),
        (middle_y, height, 0, middle_x),
        (middle_y, height, middle_x, width),
    ]


class RegionalPhysicsExtractor:
    """Extract all four physics entities globally and over four quadrants.

    Output features have shape ``[5, 14]`` and confidence has shape ``[5, 4]``.
    DSINE runs once on the complete image. Its aligned normal and uncertainty
    fields are then summarized inside each region.
    """

    def __init__(self, normals: SurfaceNormalsExtractor):
        self.normals = normals
        self.global_light = IlluminationSHExtractor(grid_size=4)
        self.local_light = IlluminationSHExtractor(grid_size=2)
        self.specular = CornealOpticsExtractor(region_mode="general")
        self.chromatic = ChromaticShadowExtractor()

    def provenance(self) -> dict:
        return {
            "layout": list(REGION_NAMES),
            "feature_dimensions_per_region": [5, 4, 3, 2],
            "normal_estimation": "one full-image prediction, aligned regional summaries",
            "specular_region_selection": "scene-wide high-frequency highlight components; no fixed eye coordinates",
            "normal_extractor": self.normals.provenance(),
        }

    def extract(
        self, image: Union[np.ndarray, torch.Tensor, str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_np = validate_and_load_image(image)
        boxes = quadrant_boxes(*image_np.shape[:2])
        normal_values = self.normals.extract_regions(image_np, boxes)
        feature_rows, confidence_rows = [], []
        for index, (y1, y2, x1, x2) in enumerate(boxes):
            crop = image_np[y1:y2, x1:x2]
            light, c_light = (self.global_light if index == 0 else self.local_light)(crop)
            specular, c_specular = self.specular(crop)
            geometry, c_geometry = normal_values[index]
            chromatic, c_chromatic = self.chromatic(crop)
            feature_rows.append(torch.cat([light, specular, geometry, chromatic]))
            confidence_rows.append(torch.tensor(
                [c_light, c_specular, c_geometry, c_chromatic], dtype=torch.float32
            ))
        features = torch.stack(feature_rows).float()
        confidences = torch.stack(confidence_rows).float()
        if features.shape != (len(REGION_NAMES), 14) or confidences.shape != (len(REGION_NAMES), 4):
            raise RuntimeError("Regional physics extractor emitted an unexpected schema")
        if not torch.isfinite(features).all() or not torch.isfinite(confidences).all():
            raise RuntimeError("Regional physics extractor emitted non-finite values")
        return features, confidences
