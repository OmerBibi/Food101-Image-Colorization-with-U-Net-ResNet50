"""PyTorch Dataset for colorization task with soft-encoded ab targets."""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List

from ..utils.color_utils import pil_to_rgb01, rgb01_to_lab, clamp_ab


class ColorizationFood101(Dataset):
    """Food101 Dataset for colorization task with soft-encoded ab targets.

    Args:
        base_ds: Base Food101 dataset
        indices: List of indices to use from base dataset
        transform: Image transformations
        centers: ab color space centers (K, 2)
        soft_knn: Number of nearest neighbors for soft encoding
        sigma_soft: Gaussian sigma for soft encoding weights
        ab_min: Minimum ab value for clamping
        ab_max: Maximum ab value for clamping
    """

    def __init__(
        self,
        base_ds,
        indices: List[int],
        transform,
        centers: np.ndarray,
        soft_knn: int = 5,
        sigma_soft: float = 5.0,
        ab_min: float = -110.0,
        ab_max: float = 110.0
    ):
        self.base = base_ds
        self.indices = list(indices)
        self.tf = transform
        self.centers = centers.astype(np.float32)
        self.soft_knn = soft_knn
        self.sigma_soft = sigma_soft
        self.ab_min = ab_min
        self.ab_max = ab_max

        # Build KNN model for soft encoding
        self.nn5 = NearestNeighbors(n_neighbors=soft_knn).fit(self.centers)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> Dict[str, torch.Tensor]:
        i = self.indices[k]
        img, _ = self.base[i]
        img = self.tf(img)

        # Convert PIL to RGB array [0, 1]
        rgb01 = pil_to_rgb01(img)

        # Store ground truth RGB for metric computation
        rgb_gt = torch.from_numpy(rgb01.astype(np.float32)).permute(2, 0, 1)

        # Convert to LAB and clamp ab channels
        lab = clamp_ab(
            rgb01_to_lab(rgb01),
            ab_min=self.ab_min,
            ab_max=self.ab_max
        )
        L, ab = lab[..., 0:1], lab[..., 1:3]
        L01 = (L / 100.0).astype(np.float32)

        H, W, _ = ab.shape

        # Soft encoding of ab values using K-nearest neighbors
        dists, idx5 = self.nn5.kneighbors(
            ab.reshape(-1, 2),
            return_distance=True
        )
        w5 = np.exp(-(dists**2) / (2.0 * self.sigma_soft**2)).astype(np.float32)
        w5 /= (w5.sum(axis=1, keepdims=True) + 1e-12)
        qstar = idx5[:, 0].reshape(H, W)

        return {
            "L": torch.from_numpy(L01).permute(2, 0, 1),
            "idx5": torch.from_numpy(idx5.reshape(H, W, 5)),
            "w5": torch.from_numpy(w5.reshape(H, W, 5)),
            "qstar": torch.from_numpy(qstar.astype(np.int64)),
            "rgb_gt": rgb_gt
        }
