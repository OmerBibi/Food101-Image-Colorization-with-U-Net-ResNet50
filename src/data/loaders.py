"""DataLoader creation for Food101 colorization task.

Extracted from training_and_eval_v2.py lines 242-257.
"""

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from typing import Tuple
from pathlib import Path

from .datasets import ColorizationFood101
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    config: dict,
    centers: np.ndarray
) -> Tuple[DataLoader, DataLoader, ColorizationFood101]:
    """Create training and validation dataloaders.

    Args:
        config: Configuration dictionary
        centers: Color space centers array

    Returns:
        (train_loader, val_loader, val_dataset)
    """
    data_root = Path(config['paths']['data_root'])
    seed = config['seed']
    val_split = config['data']['val_split']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pin_memory = config['training']['pin_memory']

    # Load base dataset
    train_base = Food101(root=str(data_root), split="train", download=True)

    # Create train/val split
    n_total = len(train_base)
    idx = np.arange(n_total)
    np.random.default_rng(seed).shuffle(idx)
    n_val = int(round(val_split * n_total))
    trn_idx, val_idx = idx[n_val:].tolist(), idx[:n_val].tolist()

    # Create datasets
    train_ds = ColorizationFood101(
        train_base,
        trn_idx,
        get_train_transforms(config),
        centers,
        soft_knn=config['soft_encoding']['k_neighbors'],
        sigma_soft=config['soft_encoding']['sigma_soft'],
        ab_min=config['color']['ab_min'],
        ab_max=config['color']['ab_max']
    )

    val_ds = ColorizationFood101(
        train_base,
        val_idx,
        get_val_transforms(config),
        centers,
        soft_knn=config['soft_encoding']['k_neighbors'],
        sigma_soft=config['soft_encoding']['sigma_soft'],
        ab_min=config['color']['ab_min'],
        ab_max=config['color']['ab_max']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, val_ds
