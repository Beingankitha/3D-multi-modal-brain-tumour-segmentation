"""Training and validation loops."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
from tqdm import tqdm
import os
from pathlib import Path

from .metrics import SegmentationMetrics


class Trainer:
    """
    Trainer class for 3D brain tumor segmentation.
    Supports multi-device training (CUDA, MPS, CPU) and mixed precision.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        metrics: SegmentationMetrics,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        amp: bool = True,
        grad_clip_max_norm: Optional[float] = None,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        """
        Args:
            model: Segmentation model
            optimizer: Optimizer
            loss_fn: Loss function
            device: Device to train on
            metrics: Metrics calculator
            scheduler: Learning rate scheduler (optional)
            amp: Use automatic mixed precision (only for CUDA)
            grad_clip_max_norm: Max gradient norm for clipping (optional)
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.metrics = metrics
        self.scheduler = scheduler
        self.grad_clip_max_norm = grad_clip_max_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision (only for CUDA)
        self.use_amp = amp and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.epoch = 0
        self.best_metric = 0.0
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip_max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_max_norm
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_max_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Compute Dice score
            with torch.no_grad():
                dice_scores = self.metrics.compute_dice(outputs, labels)
                mean_dice = dice_scores.mean().item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{mean_dice:.4f}",
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        
        return {
            "train_loss": avg_loss,
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        all_dice_scores = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                dice_scores = self.metrics.compute_dice(outputs, labels)
                all_dice_scores.append(dice_scores)
                
                # Update progress bar
                mean_dice = dice_scores.mean().item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{mean_dice:.4f}",
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(val_loader)
        all_dice_scores = torch.cat(all_dice_scores, dim=0)
        mean_dice = all_dice_scores.mean().item()
        
        return {
            "val_loss": avg_loss,
            "val_dice": mean_dice,
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
