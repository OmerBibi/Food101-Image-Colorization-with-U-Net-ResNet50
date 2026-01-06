"""Preprocessing module for color bin generation and weight computation.
"""

from .color_grids import build_color_centers
from .prior_weights import compute_rebalancing_weights

__all__ = ['build_color_centers', 'compute_rebalancing_weights']
