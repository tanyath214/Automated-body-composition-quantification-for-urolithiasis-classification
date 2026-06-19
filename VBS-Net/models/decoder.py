"""VB-Net-style decoder blocks for reconstructed VBS-Net."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .vbnet_blocks import ResidualConvBlock


class UpBlock(nn.Module):
    """Upsample, concatenate a skip connection, then refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


class VBNetDecoder(nn.Module):
    """Four-stage decoder mirroring the encoder resolution hierarchy."""

    def __init__(self, channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512)) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.up0 = UpBlock(c1, c0, c0)
        self.out_channels = c0

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x0, x1, x2, x3 = skips
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        return x
