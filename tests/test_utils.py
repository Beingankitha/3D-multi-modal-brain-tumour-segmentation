"""Tests for utility functions."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_tumor_segmentation.utils import (
    get_device,
    set_seed,
    count_parameters,
)


def test_get_device():
    """Test device selection."""
    # Test auto device
    device = get_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "mps", "cpu"]
    
    # Test CPU device
    device = get_device("cpu")
    assert device.type == "cpu"


def test_set_seed():
    """Test random seed setting."""
    set_seed(42)
    
    # Test that random operations are reproducible
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    
    assert torch.allclose(a, b)


def test_count_parameters():
    """Test parameter counting."""
    model = torch.nn.Linear(10, 5)
    num_params = count_parameters(model)
    
    # Linear layer has 10*5 + 5 = 55 parameters
    assert num_params == 55


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
