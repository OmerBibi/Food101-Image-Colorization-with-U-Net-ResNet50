"""Visualization utilities for colorization inference.

Extracted from training_and_eval_v2.py lines 217-235.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.color import lab2rgb
from typing import List, Tuple


@torch.no_grad()
def get_visuals(
    logits: torch.Tensor,
    L01: torch.Tensor,
    centers_np: np.ndarray,
    T: float = 0.42
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate RGB predictions and entropy maps.

    Args:
        logits: Model predictions (B, K, H, W)
        L01: Grayscale L channel normalized to [0, 1] (B, 1, H, W)
        centers_np: Color space centers (K, 2)
        T: Annealing temperature

    Returns:
        (list of RGB images, entropy maps)
    """
    # RGB prediction (Annealed Mean)
    centers_t = torch.from_numpy(centers_np).to(logits.device).to(logits.dtype)
    p = F.softmax(logits / T, dim=1)
    ab = torch.einsum("bkhw,kc->bchw", p, centers_t)

    L = (L01 * 100.0).cpu().numpy()
    ab_np = ab.cpu().numpy()
    B, _, H, W = L.shape

    rgbs = []
    for i in range(B):
        lab = np.zeros((H, W, 3), dtype=np.float32)
        lab[..., 0] = L[i, 0]
        lab[..., 1:] = ab_np[i].transpose(1, 2, 0)
        rgbs.append(np.clip(lab2rgb(lab), 0, 1))

    # Entropy Map
    p_full = F.softmax(logits, dim=1)
    entropy = -torch.sum(
        p_full * torch.log(p_full + 1e-10),
        dim=1
    ).cpu().numpy()

    return rgbs, entropy


@torch.no_grad()
def decode_annealed_mean(
    logits: torch.Tensor,
    centers_np: np.ndarray,
    T: float = 0.42
) -> torch.Tensor:
    """Decode logits to ab predictions using annealed mean.

    Args:
        logits: Model predictions (B, K, H, W)
        centers_np: Color space centers (K, 2)
        T: Annealing temperature

    Returns:
        Predicted ab channels (B, 2, H, W)
    """
    centers_t = torch.from_numpy(centers_np).to(logits.device).to(logits.dtype)
    p = F.softmax(logits / T, dim=1)
    ab = torch.einsum("bkhw,kc->bchw", p, centers_t)
    return ab
