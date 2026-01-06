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
