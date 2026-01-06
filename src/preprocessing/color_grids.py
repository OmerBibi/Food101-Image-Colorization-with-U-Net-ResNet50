"""Color grid building and K-means clustering for bin center generation.
"""

import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..utils.color_utils import pil_to_rgb01, rgb01_to_lab, clamp_ab


def build_ab_grid(
    step: float = 10.0,
    ab_min: float = -110.0,
    ab_max: float = 110.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Build regular grid in ab color space.

    Args:
        step: Grid spacing
        ab_min: Minimum ab value
        ab_max: Maximum ab value

    Returns:
        grid_points: (G, 2) array of grid coordinates
        grid_vals: 1D array of grid values along each axis
    """
    vals = np.arange(ab_min, ab_max + 1e-6, step, dtype=np.float32)
    aa, bb = np.meshgrid(vals, vals, indexing="xy")
    grid = np.stack([aa.reshape(-1), bb.reshape(-1)], axis=1)  # (G, 2)
    return grid, vals


def snap_to_grid(
    ab: np.ndarray,
    step: float = 10.0,
    ab_min: float = -110.0,
    ab_max: float = 110.0
) -> np.ndarray:
    """Snap ab values to nearest grid point.

    Args:
        ab: (N, 2) array of ab values
        step: Grid spacing
        ab_min: Minimum ab value
        ab_max: Maximum ab value

    Returns:
        snapped: (N, 2) array of snapped ab values
    """
    x = np.clip(ab, ab_min, ab_max)
    snapped = np.round((x - ab_min) / step) * step + ab_min
    snapped = np.clip(snapped, ab_min, ab_max)
    return snapped.astype(np.float32)


def grid_index(
    snapped_ab: np.ndarray,
    step: float = 10.0,
    ab_min: float = -110.0,
    ab_max: float = 110.0
) -> np.ndarray:
    """Map snapped ab coordinates to flat grid indices.

    Args:
        snapped_ab: (N, 2) array of grid-aligned ab values
        step: Grid spacing
        ab_min: Minimum ab value
        ab_max: Maximum ab value

    Returns:
        indices: (N,) array of flat grid indices
    """
    coord = np.round((snapped_ab - ab_min) / step).astype(np.int32)
    size = int(round((ab_max - ab_min) / step)) + 1
    a_i = coord[:, 0]
    b_i = coord[:, 1]
    return b_i * size + a_i


def build_color_centers(
    dataset,
    indices: List[int],
    transform,
    K: int,
    output_path: Path,
    config: Dict
) -> Tuple[np.ndarray, int]:
    """Build color bin centers using weighted K-means on grid-pruned data.

    This function:
    1. Samples images from the dataset
    2. Converts to LAB and snaps ab values to a regular grid
    3. Counts frequency of observed grid points
    4. Runs weighted MiniBatchKMeans on observed points
    5. Saves centers and metadata

    Args:
        dataset: PyTorch dataset (e.g., Food101)
        indices: List of dataset indices to use
        transform: Torchvision transform for preprocessing
        K: Target number of color bins
        output_path: Path to save centers .npy file
        config: Configuration dictionary

    Returns:
        centers: (K_final, 2) array of cluster centers
        K_final: Actual number of bins (may be < K if fewer observed)
    """
    # Check if centers already exist
    if output_path.exists():
        print(f"Centers already exist: {output_path}")
        centers = np.load(output_path).astype(np.float32)
        return centers, centers.shape[0]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract parameters from config
    preprocessing = config.get('preprocessing', {})
    color_config = config.get('color', {})
    seed = config.get('seed', 42)

    ab_min = color_config.get('ab_min', -110.0)
    ab_max = color_config.get('ab_max', 110.0)
    grid_step = preprocessing.get('grid_step', 10.0)
    prune_images = preprocessing.get('prune_images', 30000)

    print(f"Building color centers:")
    print(f"  Target bins: {K}")
    print(f"  Grid step: {grid_step}")
    print(f"  ab range: [{ab_min}, {ab_max}]")
    print(f"  Pruning with {min(prune_images, len(indices))} images")

    # Build grid
    grid_points, grid_vals = build_ab_grid(grid_step, ab_min, ab_max)
    G = grid_points.shape[0]
    print(f"  Grid size: {G} points")

    # Count grid point frequencies
    prune_n = min(prune_images, len(indices))
    prune_ids = indices[:prune_n]
    counts = np.zeros(G, dtype=np.int64)

    for j, i in enumerate(prune_ids, 1):
        # Load and transform image
        img, _ = dataset[i]
        img = transform(img)

        # Convert to LAB
        rgb01 = pil_to_rgb01(img)
        lab = clamp_ab(rgb01_to_lab(rgb01), ab_min, ab_max)
        ab = lab[..., 1:3].reshape(-1, 2)

        # Snap to grid and count
        snapped = snap_to_grid(ab, grid_step, ab_min, ab_max)
        idxs = grid_index(snapped, grid_step, ab_min, ab_max)
        counts += np.bincount(idxs, minlength=G)

        if j % 2000 == 0:
            observed = int((counts > 0).sum())
            print(f"  Processed {j}/{prune_n} images | Observed grid points: {observed}/{G}")

    # Get observed grid points
    observed_mask = counts > 0
    obs_points = grid_points[observed_mask]
    obs_counts = counts[observed_mask].astype(np.float64)

    k_chosen = min(K, obs_points.shape[0])
    print(f"  Observed grid points: {obs_points.shape[0]}")
    print(f"  Using K = {k_chosen}")

    # Weighted K-means
    kmeans = MiniBatchKMeans(
        n_clusters=k_chosen,
        random_state=seed,
        batch_size=2048,
        n_init=10,
        max_iter=300,
        init_size=20000,
        reassignment_ratio=0.01,
        verbose=0,
    )

    try:
        # Try using sample_weight (preferred)
        kmeans.fit(obs_points, sample_weight=obs_counts)
        print("  KMeans fitted with sample_weight")
    except TypeError:
        # Fallback: repeat points based on counts
        print("  KMeans sample_weight not supported; using repetition")
        rep = np.clip((obs_counts / obs_counts.mean()).round().astype(np.int32), 1, 200)
        rep_points = np.repeat(obs_points, rep, axis=0)
        kmeans.fit(rep_points)

    # Save centers
    centers = kmeans.cluster_centers_.astype(np.float32)
    np.save(output_path, centers)

    # Save metadata
    meta = {
        "dataset": "Food101",
        "K": int(k_chosen),
        "ab_min": float(ab_min),
        "ab_max": float(ab_max),
        "grid_step": float(grid_step),
        "prune_images": int(prune_n),
        "resize_policy": "short_side=256, center_crop=224",
        "nearest_grid": True,
        "weighted_kmeans": True,
        "seed": int(seed),
    }

    meta_path = output_path.with_suffix('').with_suffix('.json')  # Replace .npy with .json
    meta_path = output_path.parent / f"{output_path.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"  Saved centers: {output_path}")
    print(f"  Saved metadata: {meta_path}")
    print(f"  Centers range - a: [{centers[:,0].min():.1f}, {centers[:,0].max():.1f}] "
          f"| b: [{centers[:,1].min():.1f}, {centers[:,1].max():.1f}]")

    return centers, k_chosen
