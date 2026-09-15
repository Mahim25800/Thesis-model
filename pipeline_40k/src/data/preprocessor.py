"""Image preprocessing, region segmentation, and validation utilities."""

from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
from PIL import Image
import cv2


def validate_and_load_image(
    image_input: Union[str, Path, Image.Image, np.ndarray]
) -> np.ndarray:
    """Validates and loads an image as an intact RGB NumPy array uint8 [H, W, 3]."""
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
        with Image.open(img_path) as pil_img:
            pil_img.verify()
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


def save_verified_jpeg(
    image: Union[Image.Image, np.ndarray], dest_path: Union[str, Path], quality: int = 95
) -> Path:
    """Saves image as verified intact RGB JPEG."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 4:
            pil_img = Image.fromarray(image[:, :, :3]).convert("RGB")
        else:
            pil_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    pil_img.save(dest_path, format="JPEG", quality=quality)

    # Verification step
    with Image.open(dest_path) as check_img:
        check_img.verify()
    with Image.open(dest_path) as check_img:
        assert check_img.mode == "RGB", f"Saved image {dest_path} is not RGB"

    return dest_path


def extract_face_patches(image_np: np.ndarray) -> Dict[str, np.ndarray]:
    """Segments planar regional patches: forehead/top, left, right, ambient/background."""
    h, w = image_np.shape[:2]

    # Forehead / Upper central patch
    top_y1, top_y2 = int(0.12 * h), int(0.35 * h)
    top_x1, top_x2 = int(0.30 * w), int(0.70 * w)
    forehead_patch = image_np[top_y1:top_y2, top_x1:top_x2]

    # Left cheek / regional patch
    left_y1, left_y2 = int(0.40 * h), int(0.70 * h)
    left_x1, left_x2 = int(0.12 * w), int(0.42 * w)
    left_patch = image_np[left_y1:left_y2, left_x1:left_x2]

    # Right cheek / regional patch
    right_y1, right_y2 = int(0.40 * h), int(0.70 * h)
    right_x1, right_x2 = int(0.58 * w), int(0.88 * w)
    right_patch = image_np[right_y1:right_y2, right_x1:right_x2]

    # Ambient / background patch (composite of corners)
    corner_tl = image_np[0:int(0.20 * h), 0:int(0.25 * w)]
    corner_tr = image_np[0:int(0.20 * h), int(0.75 * w):w]
    ambient_patch = np.concatenate([corner_tl, corner_tr], axis=1)

    return {
        "forehead": forehead_patch,
        "left": left_patch,
        "right": right_patch,
        "ambient": ambient_patch,
    }


def extract_eye_crops(image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts left and right eye specular candidate regions."""
    h, w = image_np.shape[:2]

    # Left eye region
    ey1, ey2 = int(0.28 * h), int(0.48 * h)
    lx1, lx2 = int(0.20 * w), int(0.48 * w)
    left_eye = image_np[ey1:ey2, lx1:lx2]

    # Right eye region
    rx1, rx2 = int(0.52 * w), int(0.80 * w)
    right_eye = image_np[ey1:ey2, rx1:rx2]

    return left_eye, right_eye
