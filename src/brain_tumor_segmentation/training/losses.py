"""Loss functions for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from typing import Optional


class CombinedLoss(nn.Module):
    """
    Combined loss function for segmentation.
    Combines Dice loss and Cross-Entropy loss.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        include_background: bool = False,
        to_onehot_y: bool = False,
        softmax: bool = True,
    ):
        """
        Args:
            dice_weight: Weight for Dice loss
            ce_weight: Weight for Cross-Entropy loss
            include_background: Whether to include background in loss calculation
            to_onehot_y: Whether to convert labels to one-hot encoding
            softmax: Whether to apply softmax to predictions
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D] or [B, C, H, W, D]
        
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        
        # CE loss expects target as [B, H, W, D] with class indices
        if target.ndim == 5:  # One-hot encoded
            target_ce = torch.argmax(target, dim=1)
        else:
            target_ce = target.squeeze(1) if target.ndim == 5 else target
        
        ce = self.ce_loss(pred, target_ce.long())
        
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        return total_loss


def get_loss_function(
    loss_type: str = "dice_ce",
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
    include_background: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: Type of loss ("dice", "dice_ce", "focal", "ce")
        dice_weight: Weight for Dice loss in combined loss
        ce_weight: Weight for CE loss in combined loss
        include_background: Whether to include background class
        **kwargs: Additional loss-specific arguments
    
    Returns:
        Loss function module
    """
    if loss_type == "dice":
        return DiceLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
        )
    
    elif loss_type == "dice_ce":
        return CombinedLoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
        )
    
    elif loss_type == "focal":
        return FocalLoss(
            include_background=include_background,
            to_onehot_y=True,
        )
    
    elif loss_type == "ce":
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
