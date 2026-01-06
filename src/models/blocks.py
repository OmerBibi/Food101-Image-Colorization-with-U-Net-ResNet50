"""Reusable building blocks for the U-Net decoder.

Extracted from training_and_eval_v2.py lines 167-182.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNReLU(nn.Module):
    """Convolution + GroupNorm + ReLU block.

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        groups: Number of groups for GroupNorm (default: 32)
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.gn = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class UpBlock(nn.Module):
    """Upsampling block with skip connection.

    Args:
        in_ch: Number of input channels
        skip_ch: Number of skip connection channels
        out_ch: Number of output channels
        groups: Number of groups for GroupNorm (default: 32)
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, groups: int = 32):
        super().__init__()
        self.c1 = ConvGNReLU(in_ch + skip_ch, out_ch, groups)
        self.c2 = ConvGNReLU(out_ch, out_ch, groups)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False
        )
        return self.c2(self.c1(torch.cat([x, skip], dim=1)))
