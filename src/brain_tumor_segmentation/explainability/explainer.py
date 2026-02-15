"""Explainability methods for brain tumor segmentation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm


class OcclusionSensitivity:
    """
    Occlusion sensitivity analysis for 3D segmentation.
    Shows which regions are important for the model's predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        stride: Tuple[int, int, int] = (8, 8, 8),
        occlusion_value: float = 0.0,
    ):
        """
        Args:
            model: Trained segmentation model
            device: Device for computation
            patch_size: Size of occlusion patch
            stride: Stride for sliding the occlusion patch
            occlusion_value: Value to use for occluded regions
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.occlusion_value = occlusion_value
    
    @torch.no_grad()
    def compute_sensitivity_map(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute occlusion sensitivity map.
        
        Args:
            image: Input image [1, C, H, W, D] or [C, H, W, D]
            target_class: Target class to analyze (None for predicted class)
        
        Returns:
            Sensitivity map [H, W, D]
        """
        # Ensure batch dimension
        if image.ndim == 4:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Get baseline prediction
        baseline_logits = self.model(image)
        baseline_probs = torch.softmax(baseline_logits, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(baseline_probs, dim=1).item()
        
        baseline_score = baseline_probs[0, target_class].mean().item()
        
        # Initialize sensitivity map
        _, _, H, W, D = image.shape
        sensitivity_map = np.zeros((H, W, D))
        count_map = np.zeros((H, W, D))
        
        # Slide occlusion patch
        ph, pw, pd = self.patch_size
        sh, sw, sd = self.stride
        
        positions = []
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                for d in range(0, D - pd + 1, sd):
                    positions.append((h, w, d))
        
        # Compute sensitivity for each position
        for h, w, d in tqdm(positions, desc="Occlusion sensitivity"):
            # Create occluded image
            occluded_image = image.clone()
            occluded_image[:, :, h:h+ph, w:w+pw, d:d+pd] = self.occlusion_value
            
            # Predict with occluded image
            occluded_logits = self.model(occluded_image)
            occluded_probs = torch.softmax(occluded_logits, dim=1)
            occluded_score = occluded_probs[0, target_class].mean().item()
            
            # Compute sensitivity (drop in probability)
            sensitivity = baseline_score - occluded_score
            
            # Update sensitivity map
            sensitivity_map[h:h+ph, w:w+pw, d:d+pd] += sensitivity
            count_map[h:h+ph, w:w+pw, d:d+pd] += 1
        
        # Average overlapping regions
        sensitivity_map = np.divide(
            sensitivity_map, count_map, where=count_map > 0
        )
        
        return sensitivity_map


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for 3D segmentation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        target_layer: str,
    ):
        """
        Args:
            model: Trained segmentation model
            device: Device for computation
            target_layer: Name of the layer to compute Grad-CAM for
        """
        self.model = model.to(device)
        self.device = device
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                return
        
        raise ValueError(f"Layer {self.target_layer} not found in model")
    
    def compute_gradcam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            image: Input image [1, C, H, W, D] or [C, H, W, D]
            target_class: Target class (None for predicted class)
        
        Returns:
            Grad-CAM heatmap [H, W, D]
        """
        # Ensure batch dimension
        if image.ndim == 4:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad = True
        
        # Forward pass
        self.model.eval()
        logits = self.model(image)
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).flatten()[0]
        
        # Backward pass
        self.model.zero_grad()
        logits[0, target_class].mean().backward()
        
        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        
        # Weighted combination of activation maps
        gradcam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU
        gradcam = torch.relu(gradcam)
        
        # Normalize
        gradcam = gradcam - gradcam.min()
        gradcam = gradcam / (gradcam.max() + 1e-8)
        
        # Resize to input size
        gradcam = torch.nn.functional.interpolate(
            gradcam,
            size=image.shape[2:],
            mode='trilinear',
            align_corners=False,
        )
        
        gradcam = gradcam.squeeze().cpu().numpy()
        
        return gradcam


class SaliencyMap:
    """
    Saliency map visualization using gradients.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        """
        Args:
            model: Trained segmentation model
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
    
    def compute_saliency(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute saliency map.
        
        Args:
            image: Input image [1, C, H, W, D] or [C, H, W, D]
            target_class: Target class (None for predicted class)
        
        Returns:
            Saliency map [H, W, D]
        """
        # Ensure batch dimension
        if image.ndim == 4:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad = True
        
        # Forward pass
        self.model.eval()
        logits = self.model(image)
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).flatten()[0]
        
        # Backward pass
        self.model.zero_grad()
        logits[0, target_class].mean().backward()
        
        # Get gradients
        saliency = image.grad.abs().max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
