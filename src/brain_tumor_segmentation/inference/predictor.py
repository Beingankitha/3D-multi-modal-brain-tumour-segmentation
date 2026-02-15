"""Inference utilities for brain tumor segmentation."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer


class SegmentationInference:
    """
    Inference pipeline for 3D brain tumor segmentation.
    Supports sliding window inference for large volumes.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
        sw_batch_size: int = 4,
        mode: str = "gaussian",
    ):
        """
        Args:
            model: Trained segmentation model
            device: Device for inference
            roi_size: ROI size for sliding window
            overlap: Overlap ratio between windows
            sw_batch_size: Batch size for sliding window inference
            mode: Blending mode ("constant", "gaussian")
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Sliding window inferer
        self.inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=mode,
        )

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        use_sliding_window: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict segmentation for an input image.

        Args:
            image: Input image tensor [1, C, H, W, D] or [C, H, W, D]
            use_sliding_window: Whether to use sliding window inference

        Returns:
            Tuple of (predictions, probabilities)
            - predictions: Predicted class indices [H, W, D]
            - probabilities: Class probabilities [C, H, W, D]
        """
        # Ensure batch dimension
        if image.ndim == 4:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Run inference
        if use_sliding_window:
            logits = self.inferer(image, self.model)
        else:
            logits = self.model(image)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)

        # Get predictions
        pred = torch.argmax(probs, dim=1)

        # Remove batch dimension
        pred = pred.squeeze(0)
        probs = probs.squeeze(0)

        return pred, probs

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict segmentation for a batch of images.

        Args:
            images: Batch of images [B, C, H, W, D]

        Returns:
            Tuple of (predictions, probabilities)
        """
        images = images.to(self.device)

        logits = self.model(images)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        return pred, probs

    def predict_with_tta(
        self,
        image: torch.Tensor,
        num_augmentations: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with Test-Time Augmentation (TTA).

        Args:
            image: Input image [1, C, H, W, D]
            num_augmentations: Number of augmented predictions to average

        Returns:
            Tuple of (predictions, probabilities)
        """
        # Ensure batch dimension
        if image.ndim == 4:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        all_probs = []

        # Original prediction
        logits = self.inferer(image, self.model)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs)

        # Augmented predictions (flips)
        for flip_dims in [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
            if len(all_probs) >= num_augmentations:
                break

            # Flip image
            image_flip = torch.flip(image, dims=flip_dims)

            # Predict
            logits_flip = self.inferer(image_flip, self.model)
            probs_flip = torch.softmax(logits_flip, dim=1)

            # Flip back
            probs_flip = torch.flip(probs_flip, dims=flip_dims)
            all_probs.append(probs_flip)

        # Average probabilities
        mean_probs = torch.stack(all_probs).mean(dim=0)
        pred = torch.argmax(mean_probs, dim=1)

        # Remove batch dimension
        pred = pred.squeeze(0)
        mean_probs = mean_probs.squeeze(0)

        return pred, mean_probs


def post_process_prediction(
    pred: torch.Tensor,
    min_size: int = 64,
    remove_small_objects: bool = True,
) -> torch.Tensor:
    """
    Post-process segmentation prediction.

    Args:
        pred: Predicted segmentation [H, W, D]
        min_size: Minimum object size to keep (in voxels)
        remove_small_objects: Whether to remove small objects

    Returns:
        Post-processed prediction
    """
    if not remove_small_objects:
        return pred

    # Convert to numpy for processing
    pred_np = pred.cpu().numpy() if torch.is_tensor(pred) else pred

    # Process each class separately
    from scipy.ndimage import label
    from scipy.ndimage import sum as ndi_sum

    result = np.zeros_like(pred_np)

    for class_idx in range(1, pred_np.max() + 1):
        # Get binary mask for this class
        binary_mask = pred_np == class_idx

        # Label connected components
        labeled, num_features = label(binary_mask)

        # Compute sizes
        if num_features > 0:
            sizes = ndi_sum(binary_mask, labeled, range(1, num_features + 1))

            # Keep only large enough components
            for i, size in enumerate(sizes, 1):
                if size >= min_size:
                    result[labeled == i] = class_idx

    # Convert back to tensor if needed
    if torch.is_tensor(pred):
        result = torch.from_numpy(result).to(pred.device)

    return result
