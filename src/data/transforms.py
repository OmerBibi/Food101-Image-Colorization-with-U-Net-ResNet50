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
