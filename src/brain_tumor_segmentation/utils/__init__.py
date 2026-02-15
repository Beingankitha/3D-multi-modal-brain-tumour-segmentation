"""Utility functions for device management, seeding, and configuration."""

import os
import random
import torch
import numpy as np
from typing import Optional


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    Args:
        device: Device specification. Options:
            - "auto": Auto-detect best available device (cuda > mps > cpu)
            - "cuda": Use CUDA GPU
            - "mps": Use Apple Silicon GPU
            - "cpu": Use CPU
            - "cuda:0", "cuda:1", etc.: Specific CUDA device
    
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = "cpu"
            print("Using CPU")
    
    return torch.device(device)


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
        benchmark: If True, use cudnn benchmark (faster but not deterministic)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set PYTHONHASHSEED for full reproducibility
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        torch.backends.cudnn.benchmark = benchmark


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_info(device: torch.device) -> dict:
    """Get memory information for the given device."""
    info = {}
    
    if device.type == "cuda":
        info["allocated"] = torch.cuda.memory_allocated(device) / 1024**3  # GB
        info["reserved"] = torch.cuda.memory_reserved(device) / 1024**3  # GB
        info["max_allocated"] = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    elif device.type == "mps":
        # MPS doesn't provide detailed memory stats yet
        info["device"] = "mps"
    else:
        info["device"] = "cpu"
    
    return info


def create_directory(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_class_weights(label_counts: dict, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        label_counts: Dictionary mapping class indices to counts
        num_classes: Total number of classes
    
    Returns:
        Tensor of class weights
    """
    weights = torch.zeros(num_classes)
    total = sum(label_counts.values())
    
    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 1)
        weights[class_idx] = total / (num_classes * count)
    
    return weights
