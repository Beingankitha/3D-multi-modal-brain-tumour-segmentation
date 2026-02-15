"""Metrics for segmentation evaluation."""

from typing import Dict, List, Optional

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from scipy.spatial.distance import directed_hausdorff


class SegmentationMetrics:
    """
    Collection of metrics for segmentation evaluation.
    Supports multi-class and region-based evaluation.
    """

    def __init__(
        self,
        num_classes: int = 4,
        include_background: bool = False,
        reduction: str = "mean_batch",
    ):
        """
        Args:
            num_classes: Number of classes
            include_background: Whether to include background in metrics
            reduction: Reduction method for metrics
        """
        self.num_classes = num_classes
        self.include_background = include_background

        # MONAI metrics
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction=reduction,
        )

    def compute_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice coefficient.

        Args:
            pred: Predicted logits or probabilities [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D] or [B, 1, H, W, D]

        Returns:
            Dice scores per class
        """
        # Convert predictions to one-hot
        if pred.shape[1] == self.num_classes:
            pred_onehot = torch.argmax(pred, dim=1, keepdim=True)
            pred_onehot = torch.nn.functional.one_hot(
                pred_onehot.squeeze(1), num_classes=self.num_classes
            )
            pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()
        else:
            pred_onehot = pred

        # Convert target to one-hot
        if target.ndim == 4 or (target.ndim == 5 and target.shape[1] == 1):
            target = target.squeeze(1) if target.ndim == 5 else target
            target_onehot = torch.nn.functional.one_hot(
                target.long(), num_classes=self.num_classes
            )
            target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
        else:
            target_onehot = target

        dice_scores = self.dice_metric(pred_onehot, target_onehot)

        return dice_scores

    def compute_iou(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Intersection over Union (IoU).

        Args:
            pred: Predicted segmentation [B, H, W, D] or [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D]

        Returns:
            IoU scores per class
        """
        if pred.ndim == 5:
            pred = torch.argmax(pred, dim=1)

        iou_scores = []

        for class_idx in range(self.num_classes):
            if not self.include_background and class_idx == 0:
                continue

            pred_class = pred == class_idx
            target_class = target == class_idx

            intersection = (pred_class & target_class).sum().float()
            union = (pred_class | target_class).sum().float()

            if union > 0:
                iou = intersection / union
            else:
                iou = torch.tensor(float("nan"))

            iou_scores.append(iou)

        return torch.stack(iou_scores)

    def compute_region_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        regions: Dict[str, List[int]],
    ) -> Dict[str, float]:
        """
        Compute metrics for specific tumor regions (WT, TC, ET).

        Args:
            pred: Predicted segmentation [B, C, H, W, D] or [B, H, W, D]
            target: Ground truth labels [B, H, W, D]
            regions: Dictionary mapping region names to class indices
                    e.g., {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}

        Returns:
            Dictionary of region-wise Dice scores
        """
        if pred.ndim == 5 and pred.shape[1] > 1:
            pred = torch.argmax(pred, dim=1)

        region_metrics = {}

        for region_name, class_indices in regions.items():
            # Create binary masks for the region
            pred_region = torch.zeros_like(pred, dtype=torch.bool)
            target_region = torch.zeros_like(target, dtype=torch.bool)

            for class_idx in class_indices:
                pred_region |= pred == class_idx
                target_region |= target == class_idx

            # Compute Dice for this region
            intersection = (pred_region & target_region).sum().float()
            pred_sum = pred_region.sum().float()
            target_sum = target_region.sum().float()

            if pred_sum + target_sum > 0:
                dice = (2.0 * intersection) / (pred_sum + target_sum)
            else:
                dice = torch.tensor(float("nan"))

            region_metrics[f"dice_{region_name}"] = dice.item()

        return region_metrics

    def reset(self):
        """Reset metric accumulators."""
        self.dice_metric.reset()


def compute_hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: float = 95,
) -> float:
    """
    Compute Hausdorff distance between predicted and target segmentations.

    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        percentile: Percentile for robust Hausdorff distance

    Returns:
        Hausdorff distance
    """
    # Get surface points
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float("inf")

    # Compute directed Hausdorff distances
    dist_1 = directed_hausdorff(pred_points, target_points)[0]
    dist_2 = directed_hausdorff(target_points, pred_points)[0]

    # Return max (standard Hausdorff) or percentile (robust)
    if percentile == 100:
        return max(dist_1, dist_2)
    else:
        # For percentile-based, compute all distances and take percentile
        from scipy.spatial.distance import cdist

        distances = cdist(pred_points, target_points)
        hd = np.percentile(distances, percentile)
        return hd
