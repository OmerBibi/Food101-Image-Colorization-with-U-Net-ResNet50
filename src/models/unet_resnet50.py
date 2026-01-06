"""U-Net with ResNet encoder for colorization.

Extracted from training_and_eval_v2.py lines 158-204.
Made configurable via model_config dictionary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
)
from typing import Dict

from .blocks import ConvGNReLU, UpBlock


def adapt_resnet_first_conv_to_1ch(m: nn.Module) -> nn.Module:
    """Modify ResNet to accept 1-channel grayscale input.

    Args:
        m: ResNet module

    Returns:
        Modified module with 1-channel input
    """
    conv1 = m.conv1
    new_conv = nn.Conv2d(
        1, conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=False
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
    m.conv1 = new_conv
    return m


class UNetResNet50(nn.Module):
    """U-Net with ResNet encoder for colorization.

    Supports configurable encoder backbones and decoder channel sizes.

    Args:
        num_classes: Number of color bins (K)
        model_config: Model configuration dictionary with keys:
            - encoder: resnet architecture (resnet18/34/50/101)
            - encoder_pretrained: whether to use pretrained weights
            - bottleneck_channels: list of bottleneck channel sizes
            - decoder_channels: list of decoder channel sizes (5 elements)
            - groupnorm_groups: number of groups for GroupNorm
            - skip_connections: whether to use skip connections
    """

    # Feature channels for each ResNet architecture [after conv1, layer1, layer2, layer3, layer4]
    ENCODER_CHANNELS = {
        'resnet18': [64, 64, 128, 256, 512],
        'resnet34': [64, 64, 128, 256, 512],
        'resnet50': [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
    }

    def __init__(self, num_classes: int, model_config: Dict):
        super().__init__()

        encoder_name = model_config['encoder']
        encoder_pretrained = model_config['encoder_pretrained']
        bottleneck_ch = model_config['bottleneck_channels']
        decoder_ch = model_config['decoder_channels']
        groups = model_config.get('groupnorm_groups', 32)
        self.use_skip = model_config.get('skip_connections', True)

        # Get encoder channels
        if encoder_name not in self.ENCODER_CHANNELS:
            raise ValueError(
                f"Unsupported encoder: {encoder_name}. "
                f"Supported: {list(self.ENCODER_CHANNELS.keys())}"
            )
        enc_ch = self.ENCODER_CHANNELS[encoder_name]

        # Create encoder
        if encoder_name == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if encoder_pretrained else None
            enc = resnet18(weights=weights)
        elif encoder_name == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if encoder_pretrained else None
            enc = resnet34(weights=weights)
        elif encoder_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if encoder_pretrained else None
            enc = resnet50(weights=weights)
        elif encoder_name == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V2 if encoder_pretrained else None
            enc = resnet101(weights=weights)

        self.enc = adapt_resnet_first_conv_to_1ch(enc)

        # Bottleneck layers
        self.b1 = ConvGNReLU(enc_ch[4], bottleneck_ch[1], groups)
        self.b2 = ConvGNReLU(bottleneck_ch[1], bottleneck_ch[2], groups)

        # Decoder with skip connections
        # up4: from bottleneck to match layer3
        # up3: from up4 to match layer2
        # up2: from up3 to match layer1
        # up1: from up2 to match conv1
        skip_ch = enc_ch if self.use_skip else [0, 0, 0, 0, 0]

        self.up4 = UpBlock(bottleneck_ch[2], skip_ch[3], decoder_ch[0], groups)
        self.up3 = UpBlock(decoder_ch[0], skip_ch[2], decoder_ch[1], groups)
        self.up2 = UpBlock(decoder_ch[1], skip_ch[1], decoder_ch[2], groups)
        self.up1 = UpBlock(decoder_ch[2], skip_ch[0], decoder_ch[3], groups)

        self.final_up = ConvGNReLU(decoder_ch[3], decoder_ch[4], groups)
        self.head = nn.Conv2d(decoder_ch[4], num_classes, 1)

    def forward(self, L):
        """Forward pass.

        Args:
            L: Grayscale L channel (B, 1, H, W)

        Returns:
            Logits for ab bins (B, K, H, W)
        """
        # Encoder
        x = self.enc.conv1(L)
        x = self.enc.bn1(x)
        x0 = self.enc.relu(x)
        x = self.enc.maxpool(x0)
        x1 = self.enc.layer1(x)
        x2 = self.enc.layer2(x1)
        x3 = self.enc.layer3(x2)
        x4 = self.enc.layer4(x3)

        # Bottleneck
        b = self.b2(self.b1(x4))

        # Decoder with optional skip connections
        if self.use_skip:
            d3 = self.up4(b, x3)
            d2 = self.up3(d3, x2)
            d1 = self.up2(d2, x1)
            d0 = self.up1(d1, x0)
        else:
            # Without skip connections, pass zeros
            d3 = self.up4(b, torch.zeros_like(x3))
            d2 = self.up3(d3, torch.zeros_like(x2))
            d1 = self.up2(d2, torch.zeros_like(x1))
            d0 = self.up1(d1, torch.zeros_like(x0))

        u = F.interpolate(
            d0,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False
        )
        return self.head(self.final_up(u))
