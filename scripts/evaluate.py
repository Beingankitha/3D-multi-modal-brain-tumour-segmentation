#!/usr/bin/env python
"""Evaluation script for 3D brain tumor segmentation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_tumor_segmentation.data import (
    create_data_loaders,
    load_msd_task01_data,
    simple_transform,
)
from brain_tumor_segmentation.models import build_model
from brain_tumor_segmentation.training import SegmentationMetrics
from brain_tumor_segmentation.utils import get_device, set_seed
from brain_tumor_segmentation.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate brain tumor segmentation model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.data_root:
        config.data.root_dir = args.data_root

    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(config.reproducibility.seed)

    # Load dataset
    print(f"\nLoading dataset from {config.data.root_dir}")
    train_data, val_data, test_data = load_msd_task01_data(
        data_root=config.data.root_dir,
        seed=config.data.seed,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
    )

    # Select split
    if args.split == "train":
        eval_data = train_data
    elif args.split == "val":
        eval_data = val_data
    else:
        eval_data = test_data

    print(f"Evaluating on {args.split} split: {len(eval_data)} samples")

    # Create data loader
    from torch.utils.data import DataLoader

    from brain_tumor_segmentation.data import BrainTumorDataset

    eval_dataset = BrainTumorDataset(eval_data, transform=simple_transform)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # Build model
    print("\nBuilding model...")
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

    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Create metrics
    metrics = SegmentationMetrics(
        num_classes=config.data.num_classes,
        include_background=False,
    )

    # Define tumor regions
    regions = {
        "WT": [1, 2, 3],  # Whole Tumor
        "TC": [1, 3],  # Tumor Core
        "ET": [3],  # Enhancing Tumor
    }

    # Evaluation loop
    print("\nRunning evaluation...")
    all_dice_scores = []
    all_region_metrics = {name: [] for name in regions.keys()}

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images)

            # Compute Dice scores
            dice_scores = metrics.compute_dice(outputs, labels)
            all_dice_scores.append(dice_scores)

            # Compute region-based metrics
            region_metrics = metrics.compute_region_metrics(outputs, labels, regions)
            for region_name, dice_value in region_metrics.items():
                if not np.isnan(dice_value):
                    all_region_metrics[region_name.split("_")[1]].append(dice_value)

    # Aggregate results
    all_dice_scores = torch.cat(all_dice_scores, dim=0)
    mean_dice_per_class = all_dice_scores.mean(dim=0)
    overall_mean_dice = all_dice_scores.mean()

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Mean Dice: {overall_mean_dice:.4f}")

    print("\nPer-Class Dice Scores:")
    label_names = ["Necrotic/Non-enhancing", "Edema", "Enhancing"]
    for i, (name, dice) in enumerate(zip(label_names, mean_dice_per_class)):
        print(f"  Class {i+1} ({name}): {dice:.4f}")

    print("\nRegion-Based Dice Scores:")
    for region_name, scores in all_region_metrics.items():
        if len(scores) > 0:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {region_name}: {mean_score:.4f} ± {std_score:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
