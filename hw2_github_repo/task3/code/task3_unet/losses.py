from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: int = -1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        target_one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        target_one_hot = target_one_hot * valid
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        cardinality = torch.sum(probs + target_one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(name: str, num_classes: int) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=-1)
    if name == "dice":
        return DiceLoss(num_classes)
    if name == "ce_dice":
        return CombinedLoss(num_classes)
    raise ValueError(f"Unknown loss: {name}")
