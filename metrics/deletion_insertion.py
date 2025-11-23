"""
Deletion and Insertion metrics for faithfulness evaluation
Petsiuk et al., "RISE: Randomized Input Sampling for Explanation 
of Black-box Models", BMVC 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
from tqdm import tqdm


class DeletionInsertion:
    """
    Deletion and Insertion metrics for evaluating explanation faithfulness.
    
    Deletion: Progressively removes most important pixels and measures drop in confidence
    Insertion: Progressively adds most important pixels and measures rise in confidence
    
    Args:
        model: PyTorch model
        device: Device to run computations on
        substrate: Value for deleted/uninserted pixels ('blur', 'zero', 'mean')
        n_steps: Number of deletion/insertion steps
        batch_size: Batch size for processing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        substrate: str = 'blur',
        n_steps: int = 100,
        batch_size: int = 10
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.substrate = substrate
        self.n_steps = n_steps
        self.batch_size = batch_size
    
    def _get_substrate(
        self,
        input_tensor: torch.Tensor,
        method: str
    ) -> torch.Tensor:
        """Generate substrate for deletion/insertion."""
        if method == 'zero':
            return torch.zeros_like(input_tensor)
        elif method == 'mean':
            mean = input_tensor.mean(dim=(2, 3), keepdim=True)
            return mean.expand_as(input_tensor)
        elif method == 'blur':
            # Gaussian blur
            kernel_size = 11
            sigma = 10
            from torchvision.transforms import GaussianBlur
            blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            return blur(input_tensor)
        else:
            raise ValueError(f"Unknown substrate method: {method}")
    
    def deletion_score(
        self,
        input_tensor: torch.Tensor,
        saliency_map: torch.Tensor,
        target_class: int,
        return_curve: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute deletion metric (lower is better).
        
        Args:
            input_tensor: Input image (1, C, H, W)
            saliency_map: Explanation map (H, W)
            target_class: Target class index
            return_curve: Whether to return the full curve
            
        Returns:
            Area Under Curve (AUC) and optionally the curve
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.detach().to(self.device)
        saliency_map = saliency_map.detach().to(self.device) if isinstance(saliency_map, torch.Tensor) else torch.tensor(saliency_map).to(self.device)
        
        # Get substrate
        substrate = self._get_substrate(input_tensor, self.substrate)
        
        # Flatten and sort by importance
        h, w = saliency_map.shape
        flat_saliency = saliency_map.flatten()
        sorted_indices = torch.argsort(flat_saliency, descending=True)
        
        # Calculate step size
        total_pixels = h * w
        step_size = max(1, total_pixels // self.n_steps)
        
        probs = []
        
        with torch.no_grad():
            # Initial prediction
            output = self.model(input_tensor)
            prob = F.softmax(output, dim=1)[0, target_class].item()
            probs.append(prob)
            
            # Progressively delete pixels
            modified = input_tensor.clone()
            
            for i in range(0, total_pixels, step_size):
                # Get indices to delete
                end_idx = min(i + step_size, total_pixels)
                indices_to_delete = sorted_indices[i:end_idx]
                
                # Convert flat indices to 2D
                y_coords = indices_to_delete // w
                x_coords = indices_to_delete % w
                
                # Delete pixels
                modified[:, :, y_coords, x_coords] = substrate[:, :, y_coords, x_coords]
                
                # Get prediction
                output = self.model(modified)
                prob = F.softmax(output, dim=1)[0, target_class].item()
                probs.append(prob)
        
        # Calculate AUC
        probs = np.array(probs)
        auc = np.trapz(probs, dx=1.0 / len(probs))
        
        if return_curve:
            return auc, probs
        
        return auc
    
    def insertion_score(
        self,
        input_tensor: torch.Tensor,
        saliency_map: torch.Tensor,
        target_class: int,
        return_curve: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute insertion metric (higher is better).
        
        Args:
            input_tensor: Input image (1, C, H, W)
            saliency_map: Explanation map (H, W)
            target_class: Target class index
            return_curve: Whether to return the full curve
            
        Returns:
            Area Under Curve (AUC) and optionally the curve
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.detach().to(self.device)
        saliency_map = saliency_map.detach().to(self.device) if isinstance(saliency_map, torch.Tensor) else torch.tensor(saliency_map).to(self.device)
        
        # Get substrate
        substrate = self._get_substrate(input_tensor, self.substrate)
        
        # Flatten and sort by importance
        h, w = saliency_map.shape
        flat_saliency = saliency_map.flatten()
        sorted_indices = torch.argsort(flat_saliency, descending=True)
        
        # Calculate step size
        total_pixels = h * w
        step_size = max(1, total_pixels // self.n_steps)
        
        probs = []
        
        with torch.no_grad():
            # Start with substrate
            modified = substrate.clone()
            
            # Initial prediction (should be low)
            output = self.model(modified)
            prob = F.softmax(output, dim=1)[0, target_class].item()
            probs.append(prob)
            
            # Progressively insert pixels
            for i in range(0, total_pixels, step_size):
                # Get indices to insert
                end_idx = min(i + step_size, total_pixels)
                indices_to_insert = sorted_indices[i:end_idx]
                
                # Convert flat indices to 2D
                y_coords = indices_to_insert // w
                x_coords = indices_to_insert % w
                
                # Insert pixels
                modified[:, :, y_coords, x_coords] = input_tensor[:, :, y_coords, x_coords]
                
                # Get prediction
                output = self.model(modified)
                prob = F.softmax(output, dim=1)[0, target_class].item()
                probs.append(prob)
        
        # Calculate AUC
        probs = np.array(probs)
        auc = np.trapz(probs, dx=1.0 / len(probs))
        
        if return_curve:
            return auc, probs
        
        return auc
    
    def evaluate(
        self,
        input_tensor: torch.Tensor,
        saliency_map: torch.Tensor,
        target_class: int
    ) -> dict:
        """
        Evaluate both deletion and insertion metrics.
        
        Returns:
            Dictionary with both scores and curves
        """
        del_auc, del_curve = self.deletion_score(
            input_tensor, saliency_map, target_class, return_curve=True
        )
        
        ins_auc, ins_curve = self.insertion_score(
            input_tensor, saliency_map, target_class, return_curve=True
        )
        
        return {
            'deletion_auc': del_auc,
            'insertion_auc': ins_auc,
            'deletion_curve': del_curve,
            'insertion_curve': ins_curve
        }


def deletion_insertion_curves(
    model: nn.Module,
    input_tensor: torch.Tensor,
    saliency_map: torch.Tensor,
    target_class: int,
    **kwargs
) -> dict:
    """
    Convenience function for deletion/insertion evaluation.
    
    Returns:
        Dictionary with metrics
    """
    evaluator = DeletionInsertion(model, **kwargs)
    return evaluator.evaluate(input_tensor, saliency_map, target_class)
