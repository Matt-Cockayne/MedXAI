"""
Attention Map Extraction for Vision Transformers
Extracts and visualizes attention weights from ViT models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
from .base import BaseExplainer


class AttentionExtractor(BaseExplainer):
    """
    Extract attention maps from Vision Transformer models.
    
    Args:
        model: Vision Transformer model (e.g., from timm or transformers)
        device: Device to run computations on
        attention_layer: Which layer to extract attention from (-1 for last)
        head_fusion: How to fuse multiple attention heads ('mean', 'max', 'min')
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        attention_layer: int = -1,
        head_fusion: str = 'mean'
    ):
        super().__init__(model, device)
        self.attention_layer = attention_layer
        self.head_fusion = head_fusion
        self.attention_weights = []
        
        # Register hooks to capture attention
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        def attention_hook(module, input, output):
            # Store attention weights
            # Format depends on the model architecture
            if isinstance(output, tuple):
                # Some models return (output, attention_weights)
                self.attention_weights.append(output[1] if len(output) > 1 else None)
            else:
                self.attention_weights.append(None)
        
        # Try to find attention layers
        # This is a generic approach and may need customization
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if hasattr(module, 'register_forward_hook'):
                    module.register_forward_hook(attention_hook)
    
    def _extract_attention_map(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract and process attention map.
        
        Args:
            attention_weights: Raw attention weights from model
            layer_idx: Which layer to use
            
        Returns:
            Processed attention map
        """
        if len(self.attention_weights) == 0:
            raise ValueError("No attention weights captured. Model may not support attention extraction.")
        
        # Get attention from specified layer
        attn = self.attention_weights[layer_idx]
        
        if attn is None:
            raise ValueError(f"No attention weights found at layer {layer_idx}")
        
        # Attention shape is typically (batch, heads, tokens, tokens)
        # We want attention to CLS token or average across queries
        
        # Remove batch dimension if present
        if attn.dim() == 4:
            attn = attn[0]  # (heads, tokens, tokens)
        
        # Fuse attention heads
        if self.head_fusion == 'mean':
            attn = attn.mean(dim=0)  # (tokens, tokens)
        elif self.head_fusion == 'max':
            attn = attn.max(dim=0)[0]
        elif self.head_fusion == 'min':
            attn = attn.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown head fusion method: {self.head_fusion}")
        
        # Get attention to CLS token (first token) or average
        # Use attention from CLS token to all patches
        attn_map = attn[0, 1:]  # Exclude CLS token itself
        
        return attn_map
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate attention map visualization.
        
        Args:
            input_tensor: Input image tensor
            target_class: Not used for attention (kept for interface consistency)
            normalize: Whether to normalize the attention map
            layer_idx: Which attention layer to visualize
            
        Returns:
            Attention map of shape (H, W)
        """
        input_tensor = self._preprocess_input(input_tensor)
        layer_idx = layer_idx or self.attention_layer
        
        # Clear previous attention weights
        self.attention_weights = []
        
        # Forward pass to capture attention
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Extract attention map
        attn_map = self._extract_attention_map(self.attention_weights, layer_idx)
        
        # Reshape to spatial dimensions
        # Assuming square patch grid
        num_patches = attn_map.shape[0]
        grid_size = int(num_patches ** 0.5)
        
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Cannot reshape {num_patches} patches to square grid")
        
        attn_map = attn_map.reshape(grid_size, grid_size)
        
        # Upsample to input size
        attn_map = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        if normalize:
            attn_map = self._normalize_heatmap(attn_map)
        
        return attn_map.cpu()
    
    def explain_rollout(
        self,
        input_tensor: torch.Tensor,
        normalize: bool = True,
        start_layer: int = 0,
        discard_ratio: float = 0.9
    ) -> torch.Tensor:
        """
        Attention rollout: accumulate attention across layers.
        Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020
        
        Args:
            input_tensor: Input image tensor
            normalize: Whether to normalize the attention map
            start_layer: Layer to start rollout from
            discard_ratio: Ratio of lowest attention values to discard
            
        Returns:
            Rolled out attention map
        """
        input_tensor = self._preprocess_input(input_tensor)
        
        self.attention_weights = []
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Initialize with identity matrix
        num_tokens = self.attention_weights[0].shape[-1]
        result = torch.eye(num_tokens, device=self.device)
        
        # Roll out attention across layers
        for attn in self.attention_weights[start_layer:]:
            if attn is None:
                continue
            
            # Average across heads
            attn = attn[0].mean(dim=0)  # (tokens, tokens)
            
            # Discard lowest attention values
            flat = attn.view(-1)
            threshold = flat.kthvalue(int(flat.shape[0] * discard_ratio))[0]
            attn = torch.where(attn < threshold, torch.zeros_like(attn), attn)
            
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            # Multiply with result
            result = torch.matmul(attn, result)
        
        # Extract attention map
        attn_map = result[0, 1:]  # From CLS to patches
        
        # Reshape and upsample
        num_patches = attn_map.shape[0]
        grid_size = int(num_patches ** 0.5)
        attn_map = attn_map.reshape(grid_size, grid_size)
        
        attn_map = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        if normalize:
            attn_map = self._normalize_heatmap(attn_map)
        
        return attn_map.cpu()
