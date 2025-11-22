"""
Concept Bottleneck Model (CBM) Attribution
Extracts concept importance for CBM-style models
Koh et al., "Concept Bottleneck Models", ICML 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union
from .base import BaseExplainer


class CBMAttribution(BaseExplainer):
    """
    Extract concept attributions from Concept Bottleneck Models.
    
    CBMs have an interpretable bottleneck layer where each unit
    represents a human-understandable concept.
    
    Args:
        model: CBM model with concept bottleneck
        concept_layer: Name of the concept bottleneck layer
        concept_names: Optional list of concept names
        device: Device to run computations on
    """
    
    def __init__(
        self,
        model: nn.Module,
        concept_layer: str,
        concept_names: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        super().__init__(model, device)
        self.concept_layer_name = concept_layer
        self.concept_layer = self._get_layer_by_name(concept_layer)
        self.concept_names = concept_names
        
        self.concept_activations = None
        self.concept_gradients = None
        
        self._register_hooks()
    
    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Get layer module by its name."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def _register_hooks(self):
        """Register hooks to capture concept activations and gradients."""
        def forward_hook(module, input, output):
            self.concept_activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.concept_gradients = grad_output[0].detach()
        
        self.concept_layer.register_forward_hook(forward_hook)
        self.concept_layer.register_full_backward_hook(backward_hook)
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, float]]:
        """
        Generate concept attribution for the prediction.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            normalize: Whether to normalize attributions
            return_dict: Return as dictionary with concept names
            
        Returns:
            Concept attributions (as tensor or dict)
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
        
        # Get concept attributions using gradient * activation
        attributions = self.concept_activations[0] * self.concept_gradients[0]
        
        if normalize:
            attributions = attributions / (attributions.abs().sum() + 1e-8)
        
        attributions = attributions.cpu()
        
        if return_dict and self.concept_names:
            return {
                name: float(attr) 
                for name, attr in zip(self.concept_names, attributions)
            }
        
        return attributions
    
    def get_concept_activations(
        self,
        input_tensor: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, float]]:
        """
        Get raw concept activations without attribution.
        
        Args:
            input_tensor: Input image tensor
            
        Returns:
            Concept activations
        """
        input_tensor = self._preprocess_input(input_tensor)
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        activations = self.concept_activations[0].cpu()
        
        if self.concept_names:
            return {
                name: float(act)
                for name, act in zip(self.concept_names, activations)
            }
        
        return activations
    
    def get_top_concepts(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        k: int = 5,
        absolute: bool = True
    ) -> List[tuple]:
        """
        Get top-k most important concepts.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            k: Number of top concepts to return
            absolute: Use absolute values for ranking
            
        Returns:
            List of (concept_name/index, attribution) tuples
        """
        attributions = self.explain(
            input_tensor, 
            target_class, 
            normalize=True,
            return_dict=False
        )
        
        if absolute:
            values, indices = torch.abs(attributions).topk(k)
        else:
            values, indices = attributions.topk(k)
        
        results = []
        for idx, val in zip(indices, values):
            idx = int(idx)
            name = self.concept_names[idx] if self.concept_names else idx
            results.append((name, float(val)))
        
        return results
    
    def spatial_concept_map(
        self,
        input_tensor: torch.Tensor,
        concept_idx: int,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate spatial map showing where a concept is activated.
        Requires that concept layer has spatial dimensions.
        
        Args:
            input_tensor: Input image tensor
            concept_idx: Index of the concept
            target_class: Target class index
            
        Returns:
            Spatial map of shape (H, W)
        """
        input_tensor = self._preprocess_input(input_tensor)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        target_class = self._get_target_class(input_tensor, target_class)
        
        # Check if concept layer has spatial dimensions
        if self.concept_activations.dim() < 4:
            raise ValueError(
                "Concept layer does not have spatial dimensions. "
                f"Shape: {self.concept_activations.shape}"
            )
        
        # Extract spatial activation for the concept
        concept_activation = self.concept_activations[0, concept_idx]  # (H', W')
        
        # Upsample to input size
        spatial_map = F.interpolate(
            concept_activation.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        spatial_map = self._normalize_heatmap(spatial_map)
        
        return spatial_map.cpu()
