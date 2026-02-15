"""Tests for inference and explainability."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_tumor_segmentation.models import build_model
from brain_tumor_segmentation.inference import SegmentationInference, post_process_prediction
from brain_tumor_segmentation.explainability import SaliencyMap


def test_segmentation_inference():
    """Test inference pipeline."""
    model = build_model(
        model_name="unet",
        in_channels=4,
        out_channels=4,
        channels=[16, 32],
        strides=[2],
    )
    
    device = torch.device("cpu")
    predictor = SegmentationInference(
        model=model,
        device=device,
        roi_size=(64, 64, 64),
        overlap=0.5,
    )
    
    # Test prediction
    image = torch.randn(1, 4, 64, 64, 64)
    pred, probs = predictor.predict(image, use_sliding_window=False)
    
    assert pred.shape == (64, 64, 64)
    assert probs.shape == (4, 64, 64, 64)
    assert pred.dtype == torch.int64


def test_post_processing():
    """Test post-processing."""
    pred = torch.randint(0, 4, (64, 64, 64))
    
    processed = post_process_prediction(
        pred,
        min_size=10,
        remove_small_objects=True,
    )
    
    assert processed.shape == pred.shape


def test_saliency_map():
    """Test saliency map generation."""
    model = build_model(
        model_name="unet",
        in_channels=4,
        out_channels=4,
        channels=[16, 32],
        strides=[2],
    )
    
    device = torch.device("cpu")
    saliency = SaliencyMap(model=model, device=device)
    
    image = torch.randn(1, 4, 32, 32, 32)
    saliency_map = saliency.compute_saliency(image, target_class=1)
    
    assert saliency_map.shape == (32, 32, 32)
    assert saliency_map.min() >= 0
    assert saliency_map.max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
