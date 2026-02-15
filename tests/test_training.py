"""Tests for training components."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_tumor_segmentation.training import (
    SegmentationMetrics,
    get_loss_function,
)


def test_dice_loss():
    """Test Dice loss function."""
    loss_fn = get_loss_function("dice", include_background=False)

    # Create dummy data
    pred = torch.randn(2, 4, 32, 32, 32)
    target = torch.randint(0, 4, (2, 1, 32, 32, 32))  # Need channel dimension

    loss = loss_fn(pred, target)

    assert isinstance(loss.item(), float)
    assert loss >= 0


def test_combined_loss():
    """Test combined Dice + CE loss."""
    loss_fn = get_loss_function(
        "dice_ce",
        dice_weight=0.5,
        ce_weight=0.5,
        include_background=False,
    )

    # Create dummy data
    pred = torch.randn(2, 4, 32, 32, 32)
    target = torch.randint(0, 4, (2, 1, 32, 32, 32))  # Need channel dimension

    loss = loss_fn(pred, target)

    assert isinstance(loss.item(), float)
    assert loss >= 0


def test_segmentation_metrics():
    """Test segmentation metrics."""
    metrics = SegmentationMetrics(num_classes=4, include_background=False)

    # Create dummy data
    pred = torch.randn(2, 4, 32, 32, 32)
    target = torch.randint(0, 4, (2, 1, 32, 32, 32))  # Need channel dimension

    # Compute Dice
    dice_scores = metrics.compute_dice(pred, target)
    assert dice_scores.shape == (2, 3)  # batch_size=2, 3 non-background classes

    # Compute IoU
    iou_scores = metrics.compute_iou(pred, target)
    assert len(iou_scores) == 3


def test_region_metrics():
    """Test region-based metrics."""
    metrics = SegmentationMetrics(num_classes=4, include_background=False)

    # Create dummy data
    pred = torch.randint(0, 4, (2, 32, 32, 32))
    target = torch.randint(0, 4, (2, 32, 32, 32))

    regions = {
        "WT": [1, 2, 3],
        "TC": [1, 3],
        "ET": [3],
    }

    region_metrics = metrics.compute_region_metrics(pred, target, regions)

    assert "dice_WT" in region_metrics
    assert "dice_TC" in region_metrics
    assert "dice_ET" in region_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
