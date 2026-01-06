"""Loss functions for colorization.

Extracted from training_and_eval_v2.py lines 209-215.
"""

import torch
import torch.nn.functional as F


def weighted_soft_ce_loss(
    logits: torch.Tensor,
    idx5: torch.Tensor,
    w5: torch.Tensor,
    qstar: torch.Tensor,
    ab_weights_t: torch.Tensor
) -> torch.Tensor:
    """Weighted soft cross-entropy loss for color prediction.

    Args:
        logits: Model predictions (B, K, H, W)
        idx5: Top-5 nearest bin indices (B, H, W, 5)
        w5: Soft encoding weights (B, H, W, 5)
        qstar: Nearest bin index (B, H, W)
        ab_weights_t: Per-bin rebalancing weights (K,)

    Returns:
        Scalar loss value
    """
    logp = F.log_softmax(logits, dim=1)
    idx = idx5.permute(0, 3, 1, 2).contiguous()
    w = w5.permute(0, 3, 1, 2).contiguous()
    gathered = torch.gather(logp, dim=1, index=idx)
    per_pix = -(w * gathered).sum(dim=1)
    return (ab_weights_t[qstar] * per_pix).mean()
