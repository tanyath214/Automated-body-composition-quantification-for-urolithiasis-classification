"""VB-Net-style encoder blocks for the reconstructed VBS-Net model.

This is a reconstructed implementation based on the manuscript description,
not the original proprietary training code.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Module):
    """A small 2D conv block matching the VB-Net-style encoder/decoder spirit."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """Two convolution layers with a residual projection when channels change."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + residual)


class DownBlock(nn.Module):
    """Down-sampling block used by the VB-Net-style encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VBNetEncoder(nn.Module):
    """Four-level VB-Net-style encoder for 2D CT slices."""

    def __init__(self, in_channels: int = 1, channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512)) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("channels must contain five entries: stem plus four down-sampling levels.")

        self.out_channels = channels
        self.stem = ResidualConvBlock(in_channels, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        self.down4 = DownBlock(channels[3], channels[4])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x4, [x0, x1, x2, x3]
