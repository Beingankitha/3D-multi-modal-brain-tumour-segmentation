#!/usr/bin/env python
"""Inference script for 3D brain tumor segmentation."""

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
from brain_tumor_segmentation.inference import SegmentationInference, post_process_prediction


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on brain tumor MRI scans"
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
        help="Path to input NIfTI image (4D with 4 modalities) or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output segmentation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )
    parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Use test-time augmentation",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="Disable sliding window inference",
    )
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="Apply post-processing to remove small objects",
    )
    
    args = parser.parse_args()
    
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
    
    # Create inference pipeline
    roi_size = config.inference.sliding_window.roi_size
    overlap = config.inference.sliding_window.overlap
    
    predictor = SegmentationInference(
        model=model,
        device=device,
        roi_size=roi_size,
        overlap=overlap,
        mode=config.inference.sliding_window.mode,
    )
    
    # Load input image
    print(f"Loading input from {args.input}")
    img = nib.load(args.input)
    image_data = img.get_fdata()
    
    # Ensure 4D (C, H, W, D)
    if image_data.ndim == 3:
        # Single modality, expand to 4D
        image_data = np.expand_dims(image_data, axis=0)
    elif image_data.ndim == 4:
        # Check if it's (H, W, D, C) and transpose
        if image_data.shape[-1] == 4:
            image_data = np.transpose(image_data, (3, 0, 1, 2))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0)
    
    # Normalize
    for c in range(image_tensor.shape[1]):
        channel = image_tensor[0, c]
        mean = channel.mean()
        std = channel.std()
        if std > 0:
            image_tensor[0, c] = (channel - mean) / std
    
    print(f"Input shape: {image_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    use_sliding_window = not args.no_sliding_window
    
    if args.use_tta:
        pred, probs = predictor.predict_with_tta(image_tensor)
    else:
        pred, probs = predictor.predict(image_tensor, use_sliding_window=use_sliding_window)
    
    # Post-process
    if args.post_process:
        print("Applying post-processing...")
        min_size = config.inference.post_processing.get("min_size", 64)
        pred = post_process_prediction(
            pred,
            min_size=min_size,
            remove_small_objects=True,
        )
    
    # Save output
    print(f"Saving output to {args.output}")
    pred_np = pred.cpu().numpy().astype(np.uint8)
    
    # Create NIfTI image with same affine as input
    output_img = nib.Nifti1Image(pred_np, img.affine, img.header)
    nib.save(output_img, args.output)
    
    print("Inference completed successfully!")
    
    # Print label statistics
    unique, counts = np.unique(pred_np, return_counts=True)
    print("\nSegmentation statistics:")
    label_names = ["Background", "Necrotic/Non-enhancing", "Edema", "Enhancing"]
    for label, count in zip(unique, counts):
        if label < len(label_names):
            print(f"  {label_names[label]}: {count} voxels")


if __name__ == "__main__":
    main()
