"""Hybrid Dice + Focal loss used by the reconstructed VBS-Net training script."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HybridLoss(nn.Module):
    """L_hybrid = 0.5 * L_DSC + 0.5 * L_Focal."""

    def __init__(self, num_classes: int = 4, dice_weight: float = 0.5, focal_weight: float = 0.5, gamma: float = 2.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.monai_dice = None
        self.monai_focal = None
        try:
            from monai.losses import DiceLoss, FocalLoss

            self.monai_dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
            self.monai_focal = FocalLoss(to_onehot_y=True, use_softmax=True)
        except Exception:
            self.monai_dice = None
            self.monai_focal = None

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_1h, dims)
        denominator = torch.sum(probs + target_1h, dims)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()

    def _focal_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long(), reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.monai_dice is not None and self.monai_focal is not None:
            target_ch = target.long().unsqueeze(1)
            dice = self.monai_dice(logits, target_ch)
            focal = self.monai_focal(logits, target_ch)
        else:
            dice = self._dice_loss(logits, target)
            focal = self._focal_loss(logits, target)
        return self.dice_weight * dice + self.focal_weight * focal
