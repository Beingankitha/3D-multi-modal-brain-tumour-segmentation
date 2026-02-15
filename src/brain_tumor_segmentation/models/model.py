"""3D U-Net model for brain tumor segmentation."""

import torch
import torch.nn as nn
from monai.networks.nets import UNet, SwinUNETR
from typing import Sequence, Optional


def build_model(
    model_name: str = "unet",
    spatial_dims: int = 3,
    in_channels: int = 4,
    out_channels: int = 4,
    channels: Sequence[int] = (32, 64, 128, 256, 512),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.1,
    pretrained: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """
    Build a 3D segmentation model.
    
    Args:
        model_name: Name of the model architecture ("unet", "swinunetr")
        spatial_dims: Number of spatial dimensions (3 for 3D)
        in_channels: Number of input channels (4 modalities for MSD Task01)
        out_channels: Number of output channels (4 classes)
        channels: Sequence of channel counts for each level
        strides: Sequence of stride values for each level
        num_res_units: Number of residual units per block
        dropout: Dropout rate
        pretrained: Path to pretrained weights (optional)
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    """
    if model_name.lower() == "unet":
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
        )
    elif model_name.lower() == "swinunetr":
        # Swin UNETR - Transformer-based architecture
        img_size = kwargs.get("img_size", (128, 128, 128))
        feature_size = kwargs.get("feature_size", 48)
        
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=dropout,
            spatial_dims=spatial_dims,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Load pretrained weights if provided
    if pretrained is not None and pretrained != "":
        print(f"Loading pretrained weights from {pretrained}")
        state_dict = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    return model


class BrainTumorSegmentationModel(nn.Module):
    """
    Wrapper model for brain tumor segmentation with additional functionality.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 4,
        deep_supervision: bool = False,
    ):
        """
        Args:
            backbone: Base segmentation model (e.g., UNet)
            num_classes: Number of output classes
            deep_supervision: Whether to use deep supervision
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction with post-processing.
        
        Args:
            x: Input tensor [B, C, H, W, D]
        
        Returns:
            Predicted segmentation mask [B, H, W, D]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
        return pred
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Extract feature maps from a specific layer.
        Useful for visualization and explainability.
        
        Args:
            x: Input tensor
            layer_name: Name of the layer to extract features from
        
        Returns:
            Feature maps from the specified layer
        """
        features = {}
        
        def hook_fn(module, input, output):
            features["output"] = output
        
        # Register hook
        for name, module in self.backbone.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove hook
        handle.remove()
        
        return features.get("output", None)


def initialize_weights(model: nn.Module, init_type: str = "kaiming"):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        init_type: Initialization method ("kaiming", "xavier", "normal")
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "normal":
                nn.init.normal_(m.weight, 0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
