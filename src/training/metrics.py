"""Perceptual quality metrics for image colorization evaluation.

Computes LPIPS, SSIM, and PSNR metrics for comparing predicted and ground truth images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PerceptualMetrics:
    """Compute perceptual quality metrics for colorization.

    Supports three complementary metrics:
    - LPIPS: Learned perceptual similarity using deep features
    - SSIM: Structural similarity index
    - PSNR: Peak signal-to-noise ratio

    Args:
        device: Device to run computations on
        use_lpips: Enable LPIPS computation
        use_ssim: Enable SSIM computation
        use_psnr: Enable PSNR computation
        lpips_net: Network for LPIPS ('alex' or 'vgg')
    """

    def __init__(
        self,
        device: str,
        use_lpips: bool = True,
        use_ssim: bool = True,
        use_psnr: bool = True,
        lpips_net: str = 'alex'
    ):
        self.device = device
        self.use_lpips = use_lpips
        self.use_ssim = use_ssim
        self.use_psnr = use_psnr

        self.lpips_model = None
        if use_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net=lpips_net).to(device)
                self.lpips_model.eval()
                for param in self.lpips_model.parameters():
                    param.requires_grad = False
                print(f"LPIPS model ({lpips_net}) loaded successfully")
            except ImportError:
                print("Warning: lpips library not found. Install with: pip install lpips")
                self.use_lpips = False

    @torch.no_grad()
    def compute_batch_metrics(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all enabled metrics for a batch of images.

        Args:
            pred_rgb: Predicted RGB images (B, 3, H, W) in [0, 1]
            target_rgb: Ground truth RGB images (B, 3, H, W) in [0, 1]

        Returns:
            Dictionary with metric names as keys and average values as floats
        """
        metrics = {}

        if self.use_lpips and self.lpips_model is not None:
            metrics['lpips'] = self._compute_lpips(pred_rgb, target_rgb)

        if self.use_ssim:
            metrics['ssim'] = self._compute_ssim(pred_rgb, target_rgb)

        if self.use_psnr:
            metrics['psnr'] = self._compute_psnr(pred_rgb, target_rgb)

        return metrics

    def _compute_lpips(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> float:
        """Compute LPIPS metric.

        LPIPS expects inputs in [-1, 1] range.
        """
        # Normalize to [-1, 1]
        pred_norm = pred_rgb * 2.0 - 1.0
        target_norm = target_rgb * 2.0 - 1.0

        # Compute LPIPS
        lpips_value = self.lpips_model(pred_norm, target_norm)
        return lpips_value.mean().item()

    def _compute_ssim(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> float:
        """Compute SSIM metric.

        Uses pytorch-msssim implementation if available, falls back to basic computation.
        """
        try:
            from pytorch_msssim import ssim
            ssim_value = ssim(pred_rgb, target_rgb, data_range=1.0, size_average=True)
            return ssim_value.item()
        except ImportError:
            # Fallback to basic SSIM computation
            return self._compute_ssim_basic(pred_rgb, target_rgb)

    def _compute_ssim_basic(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor
    ) -> float:
        """Basic SSIM computation fallback."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(pred_rgb, 3, 1, padding=1)
        mu2 = F.avg_pool2d(target_rgb, 3, 1, padding=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred_rgb * pred_rgb, 3, 1, padding=1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target_rgb * target_rgb, 3, 1, padding=1) - mu2_sq
        sigma12 = F.avg_pool2d(pred_rgb * target_rgb, 3, 1, padding=1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def _compute_psnr(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> float:
        """Compute PSNR metric.

        PSNR = 10 * log10(MAX^2 / MSE)
        For images in [0, 1], MAX = 1.0
        """
        mse = F.mse_loss(pred_rgb, target_rgb)
        if mse == 0:
            return float('inf')

        psnr = 10.0 * torch.log10(1.0 / mse)
        return psnr.item()

    def __repr__(self):
        enabled = []
        if self.use_lpips:
            enabled.append('LPIPS')
        if self.use_ssim:
            enabled.append('SSIM')
        if self.use_psnr:
            enabled.append('PSNR')
        return f"PerceptualMetrics(enabled={enabled})"
