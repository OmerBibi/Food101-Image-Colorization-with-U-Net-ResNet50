"""Model checkpoint management.

Manages top-K best model checkpoints based on validation metrics.
"""

import torch
from pathlib import Path
from typing import List, Tuple


class TopModelManager:
    """Manages top-K best model checkpoints.

    Maintains a ranked list of the K best checkpoints based on a metric value,
    automatically deleting checkpoints that fall outside the top-K.

    Args:
        ckpt_dir: Checkpoint directory
        max_keep: Maximum number of checkpoints to keep
        metric_name: Name of the metric for filename (e.g., 'loss', 'lpips')
    """

    def __init__(self, ckpt_dir: Path, max_keep: int = 3, metric_name: str = 'loss'):
        self.ckpt_dir = ckpt_dir
        self.max_keep = max_keep
        self.metric_name = metric_name
        self.best_models: List[Tuple[float, Path]] = []

    def save_if_best(self, metric_value: float, state_dict: dict, epoch: int):
        """Save checkpoint if it's among the top-K best models.

        Args:
            metric_value: Metric value for this checkpoint (lower is better)
            state_dict: Model state dictionary to save
            epoch: Current epoch number
        """
        path = self.ckpt_dir / f"best_ep{epoch:03d}_{self.metric_name}{metric_value:.4f}.pt"
        self.best_models.append((metric_value, path))
        torch.save(state_dict, path)

        # Sort by metric value (ascending - lower is better)
        self.best_models.sort(key=lambda x: x[0])

        # Remove worst model if exceeding max_keep
        if len(self.best_models) > self.max_keep:
            _, worst_path = self.best_models.pop(-1)
            if worst_path.exists():
                worst_path.unlink()
