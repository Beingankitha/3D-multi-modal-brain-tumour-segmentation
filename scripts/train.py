#!/usr/bin/env python
"""Training script for 3D brain tumor segmentation."""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_tumor_segmentation.utils import get_device, set_seed, create_directory
from brain_tumor_segmentation.utils.config import load_config, save_config
from brain_tumor_segmentation.data import (
    load_msd_task01_data,
    create_data_loaders,
    simple_transform,
)
from brain_tumor_segmentation.models import build_model
from brain_tumor_segmentation.training import (
    get_loss_function,
    SegmentationMetrics,
    Trainer,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def main():
    parser = argparse.ArgumentParser(
        description="Train 3D brain tumor segmentation model"
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
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides (e.g., training.batch_size=4)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config, args.overrides)
    
    # Override from command line
    if args.data_root:
        config.data.root_dir = args.data_root
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    
    # Set device
    if args.device != "auto":
        config.training.device = args.device
    device = get_device(config.training.device)
    
    # Set random seed
    set_seed(
        config.reproducibility.seed,
        config.reproducibility.deterministic,
        config.reproducibility.benchmark,
    )
    
    # Create output directories
    create_directory(config.logging.output_dir)
    create_directory(config.logging.checkpoint_dir)
    create_directory(config.logging.log_dir)
    
    # Save configuration
    config_save_path = Path(config.logging.output_dir) / "config.yaml"
    save_config(config, str(config_save_path))
    print(f"Configuration saved to {config_save_path}")
    
    # Load dataset
    print(f"\nLoading dataset from {config.data.root_dir}")
    try:
        train_data, val_data, test_data = load_msd_task01_data(
            data_root=config.data.root_dir,
            seed=config.data.seed,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
            test_split=config.data.test_split,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download Task01_BrainTumour dataset:")
        print("1. Download from: http://medicaldecathlon.com/")
        print("2. Extract to: ./data/Task01_BrainTumour/")
        print("3. Or specify path with --data-root")
        return
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_transform=simple_transform,
        val_transform=simple_transform,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
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
        pretrained=config.model.pretrained,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    loss_fn = get_loss_function(
        loss_type=config.training.loss.type,
        dice_weight=config.training.loss.dice_weight,
        ce_weight=config.training.loss.ce_weight,
        include_background=config.training.loss.include_background,
    )
    
    # Create optimizer
    if config.training.optimizer.type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
        )
    elif config.training.optimizer.type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
        )
    elif config.training.optimizer.type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.optimizer.lr,
            momentum=0.9,
            weight_decay=config.training.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer.type}")
    
    # Create scheduler
    scheduler = None
    if config.training.scheduler.type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.scheduler.min_lr,
        )
    elif config.training.scheduler.type.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
    
    # Create metrics
    metrics = SegmentationMetrics(
        num_classes=config.data.num_classes,
        include_background=False,
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        metrics=metrics,
        scheduler=scheduler,
        amp=config.training.amp and device.type == "cuda",
        grad_clip_max_norm=config.training.grad_clip.max_norm if config.training.grad_clip.enabled else None,
        checkpoint_dir=config.logging.checkpoint_dir,
        log_dir=config.logging.log_dir,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # TensorBoard writer
    writer = None
    if config.logging.use_tensorboard and HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=config.logging.log_dir)
        print(f"TensorBoard logs: {config.logging.log_dir}")
    elif config.logging.use_tensorboard and not HAS_TENSORBOARD:
        print("Warning: TensorBoard not installed. Install with: pip install tensorboard")
    
    # Training loop
    print("\nStarting training...")
    print(f"Device: {device}")
    print(f"Mixed precision: {trainer.use_amp}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print("-" * 60)
    
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(trainer.epoch, config.training.num_epochs):
        trainer.epoch = epoch + 1
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        if (epoch + 1) % config.training.val_interval == 0:
            val_metrics = trainer.validate(val_loader)
            
            # Logging
            print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Dice: {val_metrics['val_dice']:.4f}")
            
            if writer:
                writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch + 1)
                writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch + 1)
                writer.add_scalar("Dice/val", val_metrics["val_dice"], epoch + 1)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % config.training.save_interval == 0:
                trainer.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Save best model
            is_best = val_metrics["val_dice"] > best_val_dice
            if is_best:
                best_val_dice = val_metrics["val_dice"]
                trainer.best_metric = best_val_dice
                trainer.save_checkpoint("checkpoint_latest.pth", is_best=True)
                patience_counter = 0
                print(f"New best model! Val Dice: {best_val_dice:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if config.training.early_stopping.enabled:
                if patience_counter >= config.training.early_stopping.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Checkpoints saved to: {config.logging.checkpoint_dir}")
    print("=" * 60)
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
