"""SAM image encoder wrapper with a lightweight ViT fallback.

The manuscript describes a pretrained SAM image encoder. This repository keeps
that interface, while providing a local ViT-like fallback so the reconstructed
implementation runs even when segment-anything or SAM checkpoints are absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class LightweightViTEncoder(nn.Module):
    """Small ViT-like encoder returning spatial bottleneck feature maps."""

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 96,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for the fallback ViT encoder.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_out", nonlinearity="relu")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        x = self.patch_embed(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.norm(self.encoder(x))
        return x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, h, w)


class SAMImageEncoderWrapper(nn.Module):
    """Optional SAM image encoder, falling back to LightweightViTEncoder.

    Args:
        use_sam: Try to instantiate segment-anything's SAM image encoder.
        checkpoint: Path to a SAM checkpoint. If omitted or unavailable, fallback
            mode is used.
        model_type: SAM model registry key, for example "vit_b".
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 96,
        output_channels: int = 256,
        use_sam: bool = False,
        checkpoint: str | None = None,
        model_type: str = "vit_b",
        fallback_depth: int = 4,
        fallback_heads: int = 8,
        fallback_patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.uses_segment_anything = False
        self.output_channels = output_channels

        sam_checkpoint = Path(checkpoint) if checkpoint else None
        if use_sam and sam_checkpoint is not None and sam_checkpoint.exists():
            try:
                from segment_anything import sam_model_registry

                sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
                self.encoder = sam.image_encoder
                self.uses_segment_anything = True
                self.sam_input_size = getattr(self.encoder, "img_size", 1024)
                self.input_projection = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
                self.channel_projection = nn.Identity()
                if output_channels != 256:
                    self.channel_projection = nn.Conv2d(256, output_channels, kernel_size=1)
                return
            except Exception:
                self.uses_segment_anything = False

        self.encoder = LightweightViTEncoder(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=fallback_patch_size,
            embed_dim=output_channels,
            depth=fallback_depth,
            num_heads=fallback_heads,
        )
        self.input_projection = nn.Identity()
        self.channel_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_segment_anything:
            x = self.input_projection(x)
            x = F.interpolate(x, size=(self.sam_input_size, self.sam_input_size), mode="bilinear", align_corners=False)
            features = self.encoder(x)
            return self.channel_projection(features)
        return self.encoder(x)
