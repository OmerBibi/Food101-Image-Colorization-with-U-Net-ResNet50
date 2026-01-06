"""Prior estimation and rebalancing weight computation.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..utils.color_utils import pil_to_rgb01, rgb01_to_lab, clamp_ab


class SoftEncoder:
    """Soft encoding using K-nearest neighbors with Gaussian weights."""

    def __init__(
        self,
        centers: np.ndarray,
        k_neighbors: int = 5,
        sigma_soft: float = 5.0
    ):
        """Initialize soft encoder.

        Args:
            centers: (K, 2) array of color bin centers
            k_neighbors: Number of nearest neighbors for soft encoding
            sigma_soft: Gaussian sigma for distance weighting
        """
        self.centers = centers
        self.K = centers.shape[0]
        self.k_neighbors = min(k_neighbors, self.K)
        self.sigma_soft = sigma_soft

        # Build KNN index
        self.nn_model = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            algorithm="auto"
        ).fit(centers)

    def encode(self, ab_hw2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Soft encode ab values to nearest color bins.

        Args:
            ab_hw2: (N, 2) array of ab values

        Returns:
            indices: (N, k_neighbors) array of bin indices
            weights: (N, k_neighbors) array of soft weights (rows sum to 1)
        """
        dists, idx = self.nn_model.kneighbors(ab_hw2, return_distance=True)

        # Gaussian weighting
        w = np.exp(-(dists**2) / (2.0 * self.sigma_soft * self.sigma_soft))
        w = w.astype(np.float32)
        w /= (w.sum(axis=1, keepdims=True) + 1e-12)

        return idx.astype(np.int64), w


class PriorSmoother:
    """Gaussian smoothing of prior distribution in color space."""

    def __init__(
        self,
        centers: np.ndarray,
        smooth_neighbors: int = 60,
        sigma_smooth: float = 5.0
    ):
        """Initialize prior smoother.

        Args:
            centers: (K, 2) array of color bin centers
            smooth_neighbors: Number of neighbors for smoothing
            sigma_smooth: Gaussian sigma for smoothing
        """
        self.centers = centers
        self.K = centers.shape[0]
        self.smooth_neighbors = min(smooth_neighbors, self.K)
        self.sigma_smooth = sigma_smooth

        # Build KNN index
        self.nn_model = NearestNeighbors(
            n_neighbors=self.smooth_neighbors,
            algorithm="auto"
        ).fit(centers)

    def smooth(self, prior: np.ndarray) -> np.ndarray:
        """Smooth prior across color space using Gaussian kernel.

        Args:
            prior: (K,) array of prior probabilities

        Returns:
            smoothed: (K,) array of smoothed prior probabilities
        """
        # Get neighbors for each center
        dists, nbrs = self.nn_model.kneighbors(
            self.centers,
            return_distance=True
        )

        # Compute Gaussian weights
        W = np.exp(-(dists**2) / (2.0 * self.sigma_smooth * self.sigma_smooth))
        W = W.astype(np.float64)
        W /= (W.sum(axis=1, keepdims=True) + 1e-12)

        # Distribute each center's prior to its neighbors
        p_smooth = np.zeros_like(prior, dtype=np.float64)
        for q in range(self.K):
            p_smooth[nbrs[q]] += prior[q] * W[q]

        # Normalize
        p_smooth = p_smooth / (p_smooth.sum() + 1e-12)
        return p_smooth.astype(np.float64)


def compute_rebalancing_weights(
    dataset,
    indices: List[int],
    transform,
    centers: np.ndarray,
    output_path: Path,
    config: Dict
) -> np.ndarray:
    """Compute class rebalancing weights using empirical prior.

    This function:
    1. Accumulates soft-encoded color histogram over training images
    2. Normalizes to empirical prior p
    3. Smooths prior in color space: p -> p_s
    4. Mixes with uniform distribution: p_tilde = (1-λ)p_s + λ/K
    5. Inverts to get class weights: w = 1/p_tilde (normalized to mean=1)

    Args:
        dataset: PyTorch dataset (e.g., Food101)
        indices: List of dataset indices to use
        transform: Torchvision transform for preprocessing
        centers: (K, 2) array of color bin centers
        output_path: Path to save weights .npy file
        config: Configuration dictionary

    Returns:
        weights: (K,) array of class rebalancing weights
    """
    # Check if weights already exist
    if output_path.exists():
        print(f"Weights already exist: {output_path}")
        weights = np.load(output_path).astype(np.float32)
        return weights

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract parameters from config
    preprocessing = config.get('preprocessing', {})
    color_config = config.get('color', {})
    soft_encoding = config.get('soft_encoding', {})

    ab_min = color_config.get('ab_min', -110.0)
    ab_max = color_config.get('ab_max', 110.0)
    lambda_uniform = preprocessing.get('lambda_uniform', 0.5)
    sigma_soft = soft_encoding.get('sigma_soft', 5.0)
    sigma_smooth = preprocessing.get('sigma_smooth', 5.0)
    smooth_neighbors = preprocessing.get('smooth_neighbors', 60)
    batch_size = preprocessing.get('prior_batch_size', 256)

    K = centers.shape[0]

    print(f"Computing rebalancing weights:")
    print(f"  Number of bins: {K}")
    print(f"  Prior images: {len(indices)}")
    print(f"  Sigma soft: {sigma_soft}")
    print(f"  Sigma smooth: {sigma_smooth}")
    print(f"  Lambda uniform: {lambda_uniform}")

    # Initialize soft encoder and smoother
    encoder = SoftEncoder(centers, k_neighbors=5, sigma_soft=sigma_soft)
    smoother = PriorSmoother(centers, smooth_neighbors=smooth_neighbors, sigma_smooth=sigma_smooth)

    # Accumulate soft-encoded histogram
    hist = np.zeros(K, dtype=np.float64)

    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start:start + batch_size]

        for i in batch_ids:
            # Load and transform image
            img, _ = dataset[i]
            img = transform(img)

            # Convert to LAB
            rgb01 = pil_to_rgb01(img)
            lab = clamp_ab(rgb01_to_lab(rgb01), ab_min, ab_max)
            ab = lab[..., 1:3].reshape(-1, 2)

            # Soft encode
            idx5, w5 = encoder.encode(ab)

            # Accumulate soft counts
            flat_idx = idx5.reshape(-1)
            flat_w = w5.reshape(-1)
            hist += np.bincount(flat_idx, weights=flat_w, minlength=K)

        # Progress logging
        if (start // batch_size) % 20 == 0:
            done = min(start + batch_size, len(indices))
            print(f"  Progress: {done}/{len(indices)} images")

    # Normalize to prior
    p = hist / (hist.sum() + 1e-12)

    # Smooth prior
    p_s = smoother.smooth(p)

    # Mix with uniform distribution
    p_tilde = (1.0 - lambda_uniform) * p_s + lambda_uniform * (1.0 / K)

    # Compute weights (inverse of prior)
    w = 1.0 / (p_tilde + 1e-12)
    w = w / w.mean()  # Normalize to mean=1

    # Save weights
    weights = w.astype(np.float32)
    np.save(output_path, weights)

    # Save metadata
    meta = {
        "dataset": "Food101",
        "K": int(K),
        "lambda_uniform": float(lambda_uniform),
        "sigma_soft": float(sigma_soft),
        "sigma_smooth": float(sigma_smooth),
        "smooth_neighbors": int(smooth_neighbors),
        "prior_soft_counts": True,
        "prior_transform": "short_side=256, center_crop=224",
        "prior_images_used": len(indices),
        "centers_file": str(output_path.parent / f"ab_centers_k{K}.npy"),
    }

    meta_path = output_path.parent / f"{output_path.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"  Saved weights: {output_path}")
    print(f"  Saved metadata: {meta_path}")
    print(f"  Weight stats - mean: {weights.mean():.3f}, min: {weights.min():.3f}, max: {weights.max():.3f}")
    print(f"  Prior sum: {p.sum():.6f}")

    return weights
