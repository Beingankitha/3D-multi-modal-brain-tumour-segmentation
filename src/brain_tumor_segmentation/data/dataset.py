"""Dataset class for brain tumor segmentation."""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    """
    Dataset class for brain tumor segmentation.

    Args:
        data_list: List of dictionaries containing file paths for images and labels
        transform: Optional transform to apply to the data
        cache_data: Whether to cache data in memory
    """

    def __init__(
        self,
        data_list: List[Dict[str, str]],
        transform: Optional[Callable] = None,
        cache_data: bool = False,
    ):
        self.data_list = data_list
        self.transform = transform
        self.cache_data = cache_data
        self._cache = {} if cache_data else None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing the image and label data
        """
        if self.cache_data and idx in self._cache:
            data = self._cache[idx]
        else:
            # For now, just return the file paths
            # In a real implementation, this would load the NIfTI files
            data = self.data_list[idx].copy()

            if self.cache_data:
                self._cache[idx] = data

        if self.transform:
            data = self.transform(data)

        return data
