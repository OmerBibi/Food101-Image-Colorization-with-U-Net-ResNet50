"""Custom transformations for the colorization dataset.

Extracted from training_and_eval_v2.py lines 119-126.
"""

from PIL import Image
from torchvision import transforms
from typing import Dict


class ResizeShortSide:
    """Resize image maintaining aspect ratio by setting short side to target size.

    Args:
        short_side: Target size for the short side
        interpolation: PIL interpolation method (default: BICUBIC)
    """

    def __init__(self, short_side: int, interpolation=Image.BICUBIC):
        self.short_side = short_side
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.short_side / float(min(w, h))
        return img.resize(
            (int(round(w * scale)), int(round(h * scale))),
            self.interpolation
        )


class ResizeDivisibleBy:
    """Resize image to nearest dimensions divisible by a factor.

    Maintains aspect ratio while ensuring dimensions are valid for
    models with specific downsampling requirements (e.g., U-Net with 32x downsampling).

    Args:
        divisor: Images dimensions must be divisible by this value (default: 32)
        max_size: Optional maximum size for either dimension (prevents OOM)
        interpolation: PIL interpolation method (default: BICUBIC)
    """

    def __init__(self, divisor: int = 32, max_size: int = None, interpolation=Image.BICUBIC):
        self.divisor = divisor
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        # Apply max_size constraint if specified
        if self.max_size is not None:
            scale = min(1.0, self.max_size / max(w, h))
            w = int(w * scale)
            h = int(h * scale)

        # Round to nearest multiple of divisor
        new_w = round(w / self.divisor) * self.divisor
        new_h = round(h / self.divisor) * self.divisor

        # Ensure minimum size (at least one divisor unit)
        new_w = max(new_w, self.divisor)
        new_h = max(new_h, self.divisor)

        return img.resize((new_w, new_h), self.interpolation)


def get_train_transforms(config: Dict) -> transforms.Compose:
    """Build training transformation pipeline.

    Args:
        config: Configuration dictionary

    Returns:
        Composed transforms for training
    """
    short_side = config['data']['resize_short_side']
    crop_size = config['data']['crop_size']

    transform_list = [
        ResizeShortSide(short_side),
        transforms.RandomCrop(crop_size),
    ]

    if config['data'].get('train_horizontal_flip', True):
        transform_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict) -> transforms.Compose:
    """Build validation transformation pipeline.

    Args:
        config: Configuration dictionary

    Returns:
        Composed transforms for validation
    """
    short_side = config['data']['resize_short_side']
    crop_size = config['data']['crop_size']

    return transforms.Compose([
        ResizeShortSide(short_side),
        transforms.CenterCrop(crop_size),
    ])


def get_inference_transforms(config: Dict, use_full_size: bool = True) -> transforms.Compose:
    """Build inference transformation pipeline.

    Args:
        config: Configuration dictionary
        use_full_size: If True, resize to nearest size divisible by 32 without cropping.
                      If False, use validation transforms (with cropping).

    Returns:
        Composed transforms for inference
    """
    if use_full_size:
        # Full-size inference: resize to divisible by 32
        divisor = config.get('inference', {}).get('resize_divisor', 32)
        max_size = config.get('inference', {}).get('max_inference_size', None)

        return transforms.Compose([
            ResizeDivisibleBy(divisor=divisor, max_size=max_size),
        ])
    else:
        # Standard inference: use validation transforms (with cropping)
        return get_val_transforms(config)
