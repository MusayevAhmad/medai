"""
X-Ray Image Classification Model

Implements transfer learning with pre-trained CNN backbones for
chest X-ray classification (COVID-19, Pneumonia, Normal).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models


# Available backbone architectures
BACKBONES = {
    "densenet121": (models.densenet121, 1024),
    "densenet169": (models.densenet169, 1664),
    "resnet50": (models.resnet50, 2048),
    "resnet101": (models.resnet101, 2048),
    "efficientnet_b0": (models.efficientnet_b0, 1280),
    "efficientnet_b1": (models.efficientnet_b1, 1280),
}


class XRayClassifier(nn.Module):
    """
    X-Ray image classifier using transfer learning.
    
    Supports multiple backbone architectures:
    - DenseNet121/169 (default, similar to CheXNet)
    - ResNet50/101
    - EfficientNet-B0/B1
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for classifier head
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from: {list(BACKBONES.keys())}")
        
        model_fn, feature_dim = BACKBONES[backbone]
        
        # Load pretrained backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = model_fn(weights=weights)
        
        # Replace classifier head based on architecture
        if "densenet" in backbone:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
        elif "resnet" in backbone:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
        elif "efficientnet" in backbone:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier head."""
        if "densenet" in self.backbone_name:
            features = self.backbone.features(x)
            out = nn.functional.relu(features, inplace=True)
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
            return torch.flatten(out, 1)
        elif "resnet" in self.backbone_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)
        elif "efficientnet" in self.backbone_name:
            return self.backbone.features(x)
        return x


def create_model(
    num_classes: int = 3,
    backbone: str = "densenet121",
    pretrained: bool = True,
    dropout: float = 0.3,
    device: Optional[str] = None,
) -> Tuple[XRayClassifier, str]:
    """
    Create an X-Ray classifier model.
    
    Args:
        num_classes: Number of output classes
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        dropout: Dropout rate
        device: Device to load model on (auto-detect if None)
    
    Returns:
        Tuple of (model, device)
    """
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Creating {backbone} model with {num_classes} classes")
    print(f"Device: {device}")
    
    model = XRayClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, device


def save_model(
    model: XRayClassifier,
    save_path: str,
    class_names: List[str],
    config: Optional[Dict] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        save_path: Path to save the model
        class_names: List of class names
        config: Optional training configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "backbone": model.backbone_name,
        "class_names": class_names,
        "config": config or {},
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(
    model_path: str,
    device: Optional[str] = None,
) -> Tuple[XRayClassifier, List[str], str]:
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to the saved model
        device: Device to load model on
    
    Returns:
        Tuple of (model, class_names, device)
    """
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = XRayClassifier(
        num_classes=checkpoint["num_classes"],
        backbone=checkpoint["backbone"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    class_names = checkpoint.get("class_names", [])
    
    print(f"Model loaded. Classes: {class_names}")
    
    return model, class_names, device


def freeze_backbone(model: XRayClassifier, freeze: bool = True) -> None:
    """
    Freeze or unfreeze the backbone layers.
    
    Useful for fine-tuning: first train only the head,
    then unfreeze and fine-tune the whole model.
    
    Args:
        model: The model
        freeze: Whether to freeze (True) or unfreeze (False)
    """
    for name, param in model.backbone.named_parameters():
        # Don't freeze the classifier head
        if "classifier" in name or "fc" in name:
            continue
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Backbone layers {status}")


if __name__ == "__main__":
    # Test model creation
    model, device = create_model(num_classes=3, backbone="densenet121")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with different backbones
    for backbone in ["resnet50", "efficientnet_b0"]:
        print(f"\nTesting {backbone}...")
        model, _ = create_model(num_classes=3, backbone=backbone)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
