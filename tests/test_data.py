"""Tests for data loading and transforms."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_tumor_segmentation.data import BrainTumorDataset, simple_transform


def test_simple_transform():
    """Test simple transform function."""
    # Create dummy data
    image = np.random.rand(4, 128, 128, 128).astype(np.float32)
    label = np.random.randint(0, 4, size=(128, 128, 128)).astype(np.int64)

    sample = {
        "image": image,
        "label": label,
    }

    # Apply transform
    transformed = simple_transform(sample)

    # Check output types and shapes
    assert isinstance(transformed["image"], torch.Tensor)
    assert isinstance(transformed["label"], torch.Tensor)
    assert transformed["image"].shape == (4, 128, 128, 128)
    assert transformed["label"].shape == (128, 128, 128)
    assert transformed["image"].dtype == torch.float32
    assert transformed["label"].dtype == torch.int64


def test_brain_tumor_dataset():
    """Test BrainTumorDataset initialization."""
    # Create dummy data list
    data_list = [
        {
            "image": "dummy_image.nii.gz",
            "label": "dummy_label.nii.gz",
        }
    ]

    # Initialize dataset (without loading actual files)
    dataset = BrainTumorDataset(data_list, transform=None, cache_data=False)

    # Check properties
    assert len(dataset) == 1
    assert dataset.data_list == data_list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
