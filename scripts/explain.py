#!/usr/bin/env python
"""Explainability analysis script."""

import argparse
import sys
from pathlib import Path
import torch
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_tumor_segmentation.utils import get_device
from brain_tumor_segmentation.utils.config import load_config
from brain_tumor_segmentation.models import build_model
from brain_tumor_segmentation.explainability import (
    OcclusionSensitivity,
    GradCAM,
    SaliencyMap,
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def save_slice_visualization(image_3d, output_path, title="", cmap="hot"):
    """Save a middle slice visualization."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed. Skipping visualization.")
        return
        
    mid_slice = image_3d.shape[2] // 2
    plt.figure(figsize=(10, 8))
    plt.imshow(image_3d[:, :, mid_slice].T, cmap=cmap, origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate explainability visualizations"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input NIfTI image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["occlusion", "saliency"],
        choices=["occlusion", "gradcam", "saliency"],
        help="Explainability methods to use",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class for analysis (default: predicted class)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Build model
    print("Building model...")
    model = build_model(
        model_name=config.model.name,
        spatial_dims=config.model.spatial_dims,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        channels=config.model.channels,
        strides=config.model.strides,
        num_res_units=config.model.num_res_units,
        dropout=config.model.dropout,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully")
    
    # Load input image
    print(f"Loading input from {args.input}")
    img = nib.load(args.input)
    image_data = img.get_fdata()
    
    # Ensure 4D (C, H, W, D)
    if image_data.ndim == 3:
        image_data = np.expand_dims(image_data, axis=0)
    elif image_data.ndim == 4 and image_data.shape[-1] == 4:
        image_data = np.transpose(image_data, (3, 0, 1, 2))
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0)
    for c in range(image_tensor.shape[1]):
        channel = image_tensor[0, c]
        mean = channel.mean()
        std = channel.std()
        if std > 0:
            image_tensor[0, c] = (channel - mean) / std
    
    print(f"Input shape: {image_tensor.shape}")
    
    # Get prediction
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        pred = torch.argmax(logits, dim=1)
    
    target_class = args.target_class if args.target_class is not None else pred.flatten()[0].item()
    print(f"Analyzing target class: {target_class}")
    
    # Run explainability methods
    if "occlusion" in args.methods:
        print("\nComputing occlusion sensitivity...")
        occlusion = OcclusionSensitivity(
            model=model,
            device=device,
            patch_size=config.explainability.occlusion.patch_size,
            stride=config.explainability.occlusion.stride,
        )
        
        sensitivity_map = occlusion.compute_sensitivity_map(
            image_tensor.to(device),
            target_class=target_class,
        )
        
        # Save visualization
        output_path = output_dir / f"occlusion_class_{target_class}.png"
        save_slice_visualization(
            sensitivity_map,
            output_path,
            title=f"Occlusion Sensitivity - Class {target_class}",
            cmap="hot",
        )
        
        # Save as NIfTI
        nifti_path = output_dir / f"occlusion_class_{target_class}.nii.gz"
        nifti_img = nib.Nifti1Image(sensitivity_map, img.affine)
        nib.save(nifti_img, nifti_path)
    
    if "saliency" in args.methods:
        print("\nComputing saliency map...")
        saliency = SaliencyMap(model=model, device=device)
        
        saliency_map = saliency.compute_saliency(
            image_tensor.to(device),
            target_class=target_class,
        )
        
        # Save visualization
        output_path = output_dir / f"saliency_class_{target_class}.png"
        save_slice_visualization(
            saliency_map,
            output_path,
            title=f"Saliency Map - Class {target_class}",
            cmap="hot",
        )
        
        # Save as NIfTI
        nifti_path = output_dir / f"saliency_class_{target_class}.nii.gz"
        nifti_img = nib.Nifti1Image(saliency_map, img.affine)
        nib.save(nifti_img, nifti_path)
    
    if "gradcam" in args.methods:
        print("\nComputing Grad-CAM...")
        try:
            target_layer = config.explainability.gradcam.target_layers[0]
            gradcam = GradCAM(
                model=model,
                device=device,
                target_layer=target_layer,
            )
            
            gradcam_map = gradcam.compute_gradcam(
                image_tensor.to(device),
                target_class=target_class,
            )
            
            # Save visualization
            output_path = output_dir / f"gradcam_class_{target_class}.png"
            save_slice_visualization(
                gradcam_map,
                output_path,
                title=f"Grad-CAM - Class {target_class}",
                cmap="hot",
            )
            
            # Save as NIfTI
            nifti_path = output_dir / f"gradcam_class_{target_class}.nii.gz"
            nifti_img = nib.Nifti1Image(gradcam_map, img.affine)
            nib.save(nifti_img, nifti_path)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
    
    print(f"\nExplainability analysis completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
