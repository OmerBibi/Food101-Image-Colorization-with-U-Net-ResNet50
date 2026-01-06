"""Color space conversion utilities.

Extracted from training_and_eval_v2.py lines 110-117.
"""

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb


def pil_to_rgb01(img: Image.Image) -> np.ndarray:
    """Convert PIL image to float32 RGB in [0,1].

    Args:
        img: PIL Image

    Returns:
        numpy array of shape (H, W, 3) with values in [0, 1]
    """
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def rgb01_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """Convert RGB in [0,1] to LAB color space.

    Args:
        rgb01: RGB array in [0, 1] range

    Returns:
        LAB array with L in [0, 100], ab in ~[-128, 127]
    """
    return rgb2lab(rgb01).astype(np.float32)


def clamp_ab(
    lab: np.ndarray,
    ab_min: float = -110.0,
    ab_max: float = 110.0
) -> np.ndarray:
    """Clamp ab channels to specified range.

    Args:
        lab: LAB array
        ab_min: Minimum value for ab channels
        ab_max: Maximum value for ab channels

    Returns:
        LAB array with clamped ab channels
    """
    lab = lab.copy()
    lab[..., 1] = np.clip(lab[..., 1], ab_min, ab_max)
    lab[..., 2] = np.clip(lab[..., 2], ab_min, ab_max)
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB in [0, 1].

    Wrapper around skimage lab2rgb with clipping.

    Args:
        lab: LAB array

    Returns:
        RGB array clipped to [0, 1]
    """
    return np.clip(lab2rgb(lab), 0, 1)
