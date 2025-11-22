"""
RISE: Randomized Input Sampling for Explanation
Petsiuk et al., "RISE: Randomized Input Sampling for Explanation of 
Black-box Models", BMVC 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .base import BaseExplainer


class RISE(BaseExplainer):
    """
    RISE implementation using random masking.
    
    Generates importance map by randomly masking the input and 
    measuring the effect on the model's output.
    
    Args:
        model: PyTorch model to explain
        device: Device to run computations on
        n_masks: Number of random masks to generate
        mask_probability: Probability of keeping each cell in mask
        cell_size: Size of each cell in the mask grid
        batch_size: Batch size for processing masks
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        n_masks: int = 8000,
        mask_probability: float = 0.5,
        cell_size: int = 8,
        batch_size: int = 128
    ):
        super().__init__(model, device)
        self.n_masks = n_masks
        self.mask_probability = mask_probability
        self.cell_size = cell_size
        self.batch_size = batch_size
    
    def _generate_masks(self, input_shape: tuple) -> torch.Tensor:
        """
        Generate random binary masks.
        
        Args:
            input_shape: Shape of input (H, W)
            
        Returns:
            Masks of shape (n_masks, H, W)
        """
        _, _, h, w = input_shape
        
        # Calculate grid size
        grid_h = h // self.cell_size
        grid_w = w // self.cell_size
        
        # Generate random masks on grid
        masks = np.random.binomial(
            1, self.mask_probability, 
            size=(self.n_masks, grid_h, grid_w)
        )
        
        # Upsample to image size
        masks = torch.from_numpy(masks).float().to(self.device)
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return masks
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True,
        n_masks: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate RISE explanation.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            normalize: Whether to normalize the saliency map
            n_masks: Number of masks (overrides default)
            batch_size: Batch size (overrides default)
            
        Returns:
            Saliency map of shape (H, W)
        """
        input_tensor = self._preprocess_input(input_tensor)
        target_class = self._get_target_class(input_tensor, target_class)
        
        n_masks = n_masks or self.n_masks
        batch_size = batch_size or self.batch_size
        
        # Generate masks
        masks = self._generate_masks(input_tensor.shape)
        
        # Initialize saliency map
        saliency = torch.zeros(
            input_tensor.shape[2:], 
            device=self.device
        )
        
        # Process in batches
        n_batches = (n_masks + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_masks)
                batch_masks = masks[start_idx:end_idx]
                
                # Apply masks to input
                masked_inputs = input_tensor * batch_masks.unsqueeze(1)
                
                # Get predictions
                outputs = self.model(masked_inputs)
                probs = F.softmax(outputs, dim=1)
                target_probs = probs[:, target_class]
                
                # Accumulate weighted masks
                for j, prob in enumerate(target_probs):
                    saliency += prob * batch_masks[j]
        
        # Normalize by number of masks
        saliency /= n_masks
        
        if normalize:
            saliency = self._normalize_heatmap(saliency)
        
        return saliency.cpu()
