"""
GradCAM: Gradient-weighted Class Activation Mapping
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
via Gradient-based Localization", ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from .base import BaseExplainer


class GradCAM(BaseExplainer):
    """
    GradCAM implementation for generating class activation maps.
    
    Args:
        model: PyTorch model to explain
        target_layer: Name of the target layer (e.g., 'layer4' for ResNet)
                     or the layer module itself
        device: Device to run computations on
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: Union[str, nn.Module],
        device: Optional[str] = None
    ):
        super().__init__(model, device)
        
        # Get the target layer
        if isinstance(target_layer, str):
            self.target_layer = self._get_layer_by_name(target_layer)
        else:
            self.target_layer = target_layer
            
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Get layer module by its name."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def explain(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Generate GradCAM explanation.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            normalize: Whether to normalize the heatmap
            
        Returns:
            GradCAM heatmap of shape (H, W)
        """
        input_tensor = self._preprocess_input(input_tensor)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        target_class = self._get_target_class(input_tensor, target_class)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Generate CAM
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Apply ReLU
        
        # Upsample to input size
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        cam = cam.squeeze()  # (H, W)
        
        if normalize:
            cam = self._normalize_heatmap(cam)
        
        return cam.cpu()
