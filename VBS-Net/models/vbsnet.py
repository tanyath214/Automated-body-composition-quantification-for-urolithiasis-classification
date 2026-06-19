"""Reconstructed VBS-Net segmentation architecture.

This module follows the manuscript-level architecture: VB-Net encoder, SAM/ViT
image encoder, bottleneck feature concatenation, 1x1 compression, VB-Net decoder,
and a configurable segmentation head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import VBNetDecoder
from .sam_encoder import SAMImageEncoderWrapper
from .vbnet_blocks import VBNetEncoder


class VBSNet(nn.Module):
    """VBS-Net for 2D L1/L3 CT slice segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        input_size: int = 96,
        encoder_channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
        sam_channels: int = 256,
        use_sam: bool = False,
        sam_checkpoint: str | None = None,
        sam_model_type: str = "vit_b",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = VBNetEncoder(in_channels=in_channels, channels=encoder_channels)
        self.sam_encoder = SAMImageEncoderWrapper(
            in_channels=in_channels,
            image_size=input_size,
            output_channels=sam_channels,
            use_sam=use_sam,
            checkpoint=sam_checkpoint,
            model_type=sam_model_type,
        )
        bottleneck_channels = encoder_channels[-1]
        self.fusion = nn.Sequential(
            nn.Conv2d(bottleneck_channels + sam_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = VBNetDecoder(channels=encoder_channels)
        self.head = nn.Conv2d(self.decoder.out_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vb_features, skips = self.encoder(x)
        sam_features = self.sam_encoder(x)
        if sam_features.shape[-2:] != vb_features.shape[-2:]:
            sam_features = F.interpolate(sam_features, size=vb_features.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat([vb_features, sam_features], dim=1))
        decoded = self.decoder(fused, skips)
        return self.head(decoded)


def build_vbsnet(config: dict | None = None) -> VBSNet:
    """Build VBSNet from a nested config dictionary."""

    config = config or {}
    model_cfg = config.get("model", config)
    return VBSNet(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 4),
        input_size=model_cfg.get("input_size", 96),
        encoder_channels=tuple(model_cfg.get("encoder_channels", [32, 64, 128, 256, 512])),
        sam_channels=model_cfg.get("sam_channels", 256),
        use_sam=model_cfg.get("use_sam", False),
        sam_checkpoint=model_cfg.get("sam_checkpoint"),
        sam_model_type=model_cfg.get("sam_model_type", "vit_b"),
    )
