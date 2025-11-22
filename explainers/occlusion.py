"""
Occlusion Sensitivity: Sliding window occlusion analysis
Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks", ECCV 2014
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import BaseExplainer


class Occlusion(BaseExplainer):
    """
    Occlusion sensitivity analysis.
    
    Systematically occludes parts of the input and measures 
    the impact on the model's prediction.
    
    Args:
        model: PyTorch model to explain
        device: Device to run computations on
        window_size: Size of the occlusion window (height, width)
        stride: Stride for sliding the window
        occlusion_value: Value to use for occlusion (default: 0)
        batch_size: Batch size for processing occlusions
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        window_size: Tuple[int, int] = (16, 16),
        stride: int = 8,
        occlusion_value: float = 0.0,
        batch_size: int = 32
    ):
        super().__init__(model, device)
        self.window_size = window_size
        self.stride = stride
        self.occlusion_value = occlusion_value
        self.batch_size = batch_size
    
    def _get_occlusion_positions(self, h: int, w: int) -> list:
        """
        Get all positions for sliding window.
        
        Args:
            h: Image height
            w: Image width
            
        Returns:
            List of (y, x) positions
        """
        positions = []
        win_h, win_w = self.window_size
        
        for y in range(0, h - win_h + 1, self.stride):
            for x in range(0, w - win_w + 1, self.stride):
                positions.append((y, x))
        
        return positions
    
    def _create_occluded_images(
        self,
        input_tensor: torch.Tensor,
        positions: list
    ) -> torch.Tensor:
        """
        Create batch of occluded images.
        
        Args:
            input_tensor: Original input tensor
            positions: List of occlusion positions
            
        Returns:
            Batch of occluded images
        """
        n_positions = len(positions)
        occluded = input_tensor.repeat(n_positions, 1, 1, 1)
        win_h, win_w = self.window_size
        
        for i, (y, x) in enumerate(positions):
            occluded[i, :, y:y+win_h, x:x+win_w] = self.occlusion_value
        
        return occluded
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True,
        window_size: Optional[Tuple[int, int]] = None,
        stride: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate occlusion sensitivity map.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            normalize: Whether to normalize the sensitivity map
            window_size: Custom window size (overrides default)
            stride: Custom stride (overrides default)
            
        Returns:
            Sensitivity map of shape (H, W)
        """
        input_tensor = self._preprocess_input(input_tensor)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            target_class = self._get_target_class(input_tensor, target_class)
            baseline_prob = F.softmax(baseline_output, dim=1)[0, target_class].item()
        
        window_size = window_size or self.window_size
        stride = stride or self.stride
        
        _, _, h, w = input_tensor.shape
        positions = self._get_occlusion_positions(h, w)
        
        # Initialize sensitivity map
        sensitivity_map = torch.zeros((h, w), device=self.device)
        counts = torch.zeros((h, w), device=self.device)
        
        # Process in batches
        n_batches = (len(positions) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(positions))
                batch_positions = positions[start_idx:end_idx]
                
                # Create occluded images
                occluded = self._create_occluded_images(
                    input_tensor, 
                    batch_positions
                )
                
                # Get predictions
                outputs = self.model(occluded)
                probs = F.softmax(outputs, dim=1)[:, target_class]
                
                # Calculate sensitivity (drop in probability)
                sensitivities = baseline_prob - probs
                
                # Accumulate sensitivity values
                win_h, win_w = window_size
                for j, (y, x) in enumerate(batch_positions):
                    sensitivity_map[y:y+win_h, x:x+win_w] += sensitivities[j]
                    counts[y:y+win_h, x:x+win_w] += 1
        
        # Average overlapping regions
        sensitivity_map = sensitivity_map / (counts + 1e-8)
        
        if normalize:
            sensitivity_map = self._normalize_heatmap(sensitivity_map)
        
        return sensitivity_map.cpu()
