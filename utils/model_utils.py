"""
Model utilities for loading and preparing models.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
import torchvision.models as models


def load_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: Optional[int] = None,
    device: Optional[str] = None
) -> nn.Module:
    """
    Load a PyTorch model.
    
    Args:
        model_name: Name of the model (e.g., 'resnet50', 'vit_b_16')
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes (for fine-tuned models)
        device: Device to load model on
        
    Returns:
        PyTorch model
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load from torchvision
    if hasattr(models, model_name):
        if pretrained:
            weights = 'DEFAULT'
            model = getattr(models, model_name)(weights=weights)
        else:
            model = getattr(models, model_name)()
    else:
        raise ValueError(f"Model {model_name} not found")
    
    # Modify output layer if needed
    if num_classes is not None:
        model = modify_output_layer(model, num_classes)
    
    model = model.to(device)
    model.eval()
    
    return model


def modify_output_layer(
    model: nn.Module,
    num_classes: int
) -> nn.Module:
    """
    Modify the final layer of a model for different number of classes.
    
    Args:
        model: PyTorch model
        num_classes: New number of output classes
        
    Returns:
        Modified model
    """
    # Get model type
    model_type = type(model).__name__.lower()
    
    if 'resnet' in model_type or 'resnext' in model_type:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif 'densenet' in model_type:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif 'efficientnet' in model_type:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif 'vit' in model_type or 'vision_transformer' in model_type:
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        print(f"Warning: Unknown model type {model_type}. Output layer not modified.")
    
    return model


def get_target_layer(
    model: nn.Module,
    model_name: Optional[str] = None
) -> nn.Module:
    """
    Get the recommended target layer for GradCAM-based methods.
    
    Args:
        model: PyTorch model
        model_name: Optional model name to help identify the layer
        
    Returns:
        Target layer module
    """
    model_type = type(model).__name__.lower()
    
    # ResNet family
    if 'resnet' in model_type:
        return model.layer4[-1]
    
    # DenseNet family
    elif 'densenet' in model_type:
        return model.features.denseblock4
    
    # EfficientNet family
    elif 'efficientnet' in model_type:
        return model.features[-1]
    
    # VGG family
    elif 'vgg' in model_type:
        return model.features[-1]
    
    # MobileNet
    elif 'mobilenet' in model_type:
        return model.features[-1]
    
    # Vision Transformer
    elif 'vit' in model_type or 'vision_transformer' in model_type:
        # For ViT, use attention extraction instead
        return None
    
    else:
        # Default: try to find last convolutional layer
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise ValueError(
                f"Could not automatically determine target layer for {model_type}. "
                "Please specify manually."
            )
        
        return last_conv


def get_target_layer_name(
    model: nn.Module,
    model_name: Optional[str] = None
) -> str:
    """
    Get the name of the recommended target layer.
    
    Args:
        model: PyTorch model
        model_name: Optional model name
        
    Returns:
        Target layer name
    """
    model_type = type(model).__name__.lower()
    
    # ResNet family
    if 'resnet' in model_type:
        return 'layer4'
    
    # DenseNet family
    elif 'densenet' in model_type:
        return 'features.denseblock4'
    
    # EfficientNet family
    elif 'efficientnet' in model_type:
        # Find the last feature layer
        for name, _ in model.named_modules():
            if 'features' in name:
                last_feature = name
        return last_feature
    
    # VGG family
    elif 'vgg' in model_type:
        return 'features.28'  # Usually the last conv layer
    
    else:
        raise ValueError(
            f"Could not determine target layer name for {model_type}. "
            "Please inspect model and specify manually."
        )


def prepare_model_for_explanation(
    model: nn.Module,
    requires_grad: bool = True
) -> nn.Module:
    """
    Prepare model for explanation by setting appropriate modes.
    
    Args:
        model: PyTorch model
        requires_grad: Whether to enable gradients
        
    Returns:
        Prepared model
    """
    model.eval()
    
    if requires_grad:
        # Enable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = True
    
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[str] = None
) -> nn.Module:
    """
    Load model weights from checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Model with loaded weights
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model
