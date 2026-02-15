"""Tests for model architecture."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_tumor_segmentation.models import build_model, BrainTumorSegmentationModel


def test_build_unet():
    """Test building U-Net model."""
    model = build_model(
        model_name="unet",
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=[16, 32, 64],
        strides=[2, 2],
        num_res_units=2,
        dropout=0.1,
    )
    
    assert model is not None
    
    # Test forward pass
    x = torch.randn(1, 4, 64, 64, 64)
    output = model(x)
    
    assert output.shape == (1, 4, 64, 64, 64)


def test_model_wrapper():
    """Test BrainTumorSegmentationModel wrapper."""
    backbone = build_model(
        model_name="unet",
        in_channels=4,
        out_channels=4,
        channels=[16, 32],
        strides=[2],
    )
    
    model = BrainTumorSegmentationModel(
        backbone=backbone,
        num_classes=4,
    )
    
    # Test forward pass
    x = torch.randn(1, 4, 64, 64, 64)
    output = model(x)
    
    assert output.shape == (1, 4, 64, 64, 64)
    
    # Test prediction
    pred = model.predict(x)
    assert pred.shape == (1, 64, 64, 64)
    assert pred.dtype == torch.int64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
