#!/usr/bin/env python3
"""
Preprocessing script for Food101 image colorization.

Generates color bin centers and rebalancing weights from training data.
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path
from torchvision.datasets import Food101

from src.config import Config
from src.data.transforms import get_val_transforms
from src.preprocessing.color_grids import build_color_centers
from src.preprocessing.prior_weights import compute_rebalancing_weights


def main(args):
    # Load configuration
    cfg = Config(args.config)
    config = cfg.config

    # Set random seeds for reproducibility
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("=" * 70)
    print("Food101 Colorization Preprocessing")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Seed: {seed}")
    print(f"Output directory: {cfg.artifact_dir}")
    print()

    # Setup paths
    centers_path = cfg.centers_path
    weights_path = cfg.weights_path

    # Ensure artifact directory exists
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    # Load Food101 dataset
    print("Loading Food101 dataset...")
    train_base = Food101(root=str(cfg.data_root), split="train", download=True)
    print(f"  Total training images: {len(train_base)}")

    # Create train/val split (consistent with training)
    n = len(train_base)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_val = int(round(config['data']['val_split'] * n))
    trn_idx = idx[n_val:].tolist()
    val_idx = idx[:n_val].tolist()

    print(f"  Train split: {len(trn_idx)} images")
    print(f"  Val split: {len(val_idx)} images")
    print()

    # Get preprocessing transform (center crop for stability)
    preprocess_tf = get_val_transforms(config)

    # Step 1: Build color centers
    print("-" * 70)
    print("Step 1: Building color bin centers")
    print("-" * 70)

    if centers_path.exists() and not args.force:
        print(f"Centers already exist at: {centers_path}")
        print("  Use --force to regenerate")
        centers = np.load(centers_path).astype(np.float32)
        K = centers.shape[0]
    else:
        centers, K = build_color_centers(
            dataset=train_base,
            indices=trn_idx,
            transform=preprocess_tf,
            K=config['preprocessing']['target_bins'],
            output_path=centers_path,
            config=config
        )

    print()

    # Step 2: Compute rebalancing weights
    print("-" * 70)
    print("Step 2: Computing rebalancing weights")
    print("-" * 70)

    if weights_path.exists() and not args.force:
        print(f"Weights already exist at: {weights_path}")
        print("  Use --force to regenerate")
        weights = np.load(weights_path).astype(np.float32)
    else:
        # Use all training images for prior computation
        prior_use_all = config['preprocessing'].get('prior_use_all_train', True)
        prior_indices = trn_idx if prior_use_all else trn_idx[:config['preprocessing']['prune_images']]

        weights = compute_rebalancing_weights(
            dataset=train_base,
            indices=prior_indices,
            transform=preprocess_tf,
            centers=centers,
            output_path=weights_path,
            config=config
        )

    print()

    # Summary
    print("=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"Color bins (K): {K}")
    print(f"Centers file: {centers_path}")
    print(f"Weights file: {weights_path}")
    print()
    print("You can now run training with:")
    print(f"  python scripts/train.py --config {args.config}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Food101 dataset for colorization training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if files exist"
    )
    args = parser.parse_args()
    main(args)
