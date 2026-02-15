"""Transform functions for brain tumor segmentation data."""

from typing import Dict

import numpy as np
import torch


def simple_transform(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Simple transform to convert numpy arrays to PyTorch tensors.

    Args:
        sample: Dictionary with 'image' and 'label' numpy arrays

    Returns:
        Dictionary with 'image' and 'label' as PyTorch tensors
    """
    image = torch.from_numpy(sample["image"]).float()
    label = torch.from_numpy(sample["label"]).long()

    return {"image": image, "label": label}
