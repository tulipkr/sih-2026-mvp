"""
Combined BCE + Dice loss (spec §13). Weighting configurable via
config['loss']['bce_weight'/'dice_weight']. Plain BCE alone collapses to
all-zero predictions on class-imbalanced oil-spill masks — Dice is
required, not optional.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one of bce_weight/dice_weight must be > 0.")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        return loss


def build_loss(config: dict) -> BCEDiceLoss:
    l = config["loss"]
    return BCEDiceLoss(bce_weight=l.get("bce_weight", 0.5), dice_weight=l.get("dice_weight", 0.5))
