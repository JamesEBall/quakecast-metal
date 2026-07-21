"""Compact attention U-Net used by the earthquake forecasting playground.

The architecture follows SmaAt-UNet: depthwise-separable convolutions, CBAM
attention at each encoder level, and U-Net skip connections.

Author: James Edward Ball
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernels_per_layer: int = 2):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + maximum)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.norm(self.conv(pooled)))


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat((skip, x), dim=1))


class SmaAtUNet(nn.Module):
    """Three aggregated input maps to one log-rate forecast map."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.att1 = CBAM(64)
        self.down1, self.att2 = Down(64, 128), CBAM(128)
        self.down2, self.att3 = Down(128, 256), CBAM(256)
        self.down3, self.att4 = Down(256, 512), CBAM(512)
        self.down4, self.att5 = Down(512, 512), CBAM(512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.att1(self.inc(x))
        x2 = self.att2(self.down1(x1))
        x3 = self.att3(self.down2(x2))
        x4 = self.att4(self.down3(x3))
        x5 = self.att5(self.down4(x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.output(x)
