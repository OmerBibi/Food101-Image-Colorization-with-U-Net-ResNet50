#!/usr/bin/env python3
"""
Training script for Food101 image colorization.

Refactored from training_and_eval_v2.py main() function.
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

from src.config import Config
from src.data.loaders import create_dataloaders
from src.models.unet_resnet50 import UNetResNet50
from src.training.trainer import Trainer


def ensure_preprocessing_artifacts(cfg):
    """Run preprocessing if artifacts are missing.

    Args:
        cfg: Config instance
    """
    if cfg.centers_path.exists() and cfg.weights_path.exists():
        return

    print("\n" + "=" * 70)
    print("Preprocessing artifacts not found!")
    print("Running automatic preprocessing...")
    print("=" * 70 + "\n")

    # Import preprocessing functions
    from src.preprocessing.color_grids import build_color_centers
    from src.preprocessing.prior_weights import compute_rebalancing_weights
    from torchvision.datasets import Food101
    from src.data.transforms import get_val_transforms

    # Load dataset
    train_base = Food101(root=str(cfg.data_root), split="train", download=True)

    # Create train/val split (consistent with training)
    seed = cfg.config['seed']
    n = len(train_base)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_val = int(round(cfg.config['data']['val_split'] * n))
    trn_idx = idx[n_val:].tolist()

    # Get preprocessing transform
    preprocess_tf = get_val_transforms(cfg.config)

    # Build centers if needed
    if not cfg.centers_path.exists():
        print("Building color centers...")
        centers, K = build_color_centers(
            dataset=train_base,
            indices=trn_idx,
            transform=preprocess_tf,
            K=cfg.config['preprocessing']['target_bins'],
            output_path=cfg.centers_path,
            config=cfg.config
        )
    else:
        centers = np.load(cfg.centers_path).astype(np.float32)

    # Build weights if needed
    if not cfg.weights_path.exists():
        print("\nComputing rebalancing weights...")
        # Use all training images for prior computation
        prior_use_all = cfg.config['preprocessing'].get('prior_use_all_train', True)
        prior_indices = trn_idx if prior_use_all else trn_idx[:cfg.config['preprocessing']['prune_images']]

        compute_rebalancing_weights(
            dataset=train_base,
            indices=prior_indices,
            transform=preprocess_tf,
            centers=centers,
            output_path=cfg.weights_path,
            config=cfg.config
        )

    print("\n" + "=" * 70)
    print("Automatic preprocessing complete!")
    print("=" * 70 + "\n")


def main(args):
    # Load configuration
    cfg = Config(args.config)
    config = cfg.config

    # Set random seeds
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Starting training on {cfg.device}")
    print(f"Config: {args.config}")

    # Ensure preprocessing artifacts exist (run if missing)
    ensure_preprocessing_artifacts(cfg)

    # Load centers and weights
    centers = np.load(cfg.centers_path).astype(np.float32)
    ab_weights = np.load(cfg.weights_path).astype(np.float32)
    K = centers.shape[0]

    print(f"Loaded {K} color bins from {cfg.centers_path}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, val_ds = create_dataloaders(config, centers)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = UNetResNet50(
        num_classes=K,
        model_config=config['model']
    ).to(cfg.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"Encoder: {config['model']['encoder']}")
    print(f"Decoder channels: {config['model']['decoder_channels']}")

    # Setup output directories
    cfg.create_output_dirs()
    output_dirs = cfg.get_output_dirs()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        centers=centers,
        ab_weights=ab_weights,
        val_dataset=val_ds,
        output_dirs=output_dirs
    )

    # Train
    print(f"\nStarting training for {config['training']['epochs']} epochs")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Freeze encoder for first {config['training']['freeze_epochs']} epochs")
    print("-" * 60)
    trainer.train()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dirs[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train colorization model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
