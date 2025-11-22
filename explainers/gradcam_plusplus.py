"""
GradCAM++: Improved version of GradCAM with better localization
Chattopadhay et al., "Grad-CAM++: Improved Visual Explanations for 
Deep Convolutional Networks", WACV 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from .base import BaseExplainer


class GradCAMPlusPlus(BaseExplainer):
    """
    GradCAM++ implementation with weighted gradient pooling.
    
    Args:
        model: PyTorch model to explain
        target_layer: Name of the target layer or the layer module itself
        device: Device to run computations on
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: Union[str, nn.Module],
        device: Optional[str] = None
    ):
        super().__init__(model, device)
        
        if isinstance(target_layer, str):
            self.target_layer = self._get_layer_by_name(target_layer)
        else:
            self.target_layer = target_layer
            
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Get layer module by its name."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
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
        Generate GradCAM++ explanation with improved weighting.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            normalize: Whether to normalize the heatmap
            
        Returns:
            GradCAM++ heatmap of shape (H, W)
        """
        input_tensor = self._preprocess_input(input_tensor)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        target_class = self._get_target_class(input_tensor, target_class)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')
        
        # Calculate alpha weights (GradCAM++ improvement)
        grad_squared = gradients ** 2
        grad_cubed = grad_squared * gradients
        
        # Compute alpha
        numerator = grad_squared
        denominator = 2 * grad_squared + torch.sum(
            activations * grad_cubed, 
            dim=(2, 3), 
            keepdim=True
        ) + 1e-8
        
        alpha = numerator / denominator
        
        # Apply ReLU to gradients (only positive influence)
        positive_gradients = F.relu(target.exp() * gradients)
        
        # Weight by alpha
        weights = torch.sum(alpha * positive_gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        cam = cam.squeeze()
        
        if normalize:
            cam = self._normalize_heatmap(cam)
        
        return cam.cpu()
