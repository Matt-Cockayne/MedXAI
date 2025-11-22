"""
Pointing Game metric for evaluating localization accuracy
Zhang et al., "Top-down Neural Attention by Excitation Backprop", ECCV 2016
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, Union


class PointingGame:
    """
    Pointing Game metric for evaluating explanation localization.
    
    Checks if the maximum of the saliency map falls within the ground truth mask.
    
    Args:
        tolerance: Pixel tolerance around the maximum point
    """
    
    def __init__(self, tolerance: int = 15):
        self.tolerance = tolerance
    
    def evaluate(
        self,
        saliency_map: torch.Tensor,
        ground_truth_mask: torch.Tensor,
        return_point: bool = False
    ) -> Union[float, Tuple[float, Tuple[int, int]]]:
        """
        Evaluate pointing game accuracy.
        
        Args:
            saliency_map: Explanation map of shape (H, W)
            ground_truth_mask: Binary mask of shape (H, W)
            return_point: Whether to return the pointed location
            
        Returns:
            Hit rate (1.0 if hit, 0.0 if miss) and optionally the point
        """
        if saliency_map.shape != ground_truth_mask.shape:
            raise ValueError(
                f"Shape mismatch: saliency {saliency_map.shape} vs "
                f"ground truth {ground_truth_mask.shape}"
            )
        
        # Find maximum point in saliency map
        flat_idx = torch.argmax(saliency_map.flatten())
        max_y, max_x = divmod(int(flat_idx), saliency_map.shape[1])
        
        # Check if within tolerance of ground truth
        hit = self._check_hit(max_y, max_x, ground_truth_mask)
        
        if return_point:
            return float(hit), (max_y, max_x)
        
        return float(hit)
    
    def _check_hit(
        self,
        y: int,
        x: int,
        mask: torch.Tensor
    ) -> bool:
        """Check if point (y, x) hits the mask within tolerance."""
        h, w = mask.shape
        
        y_min = max(0, y - self.tolerance)
        y_max = min(h, y + self.tolerance + 1)
        x_min = max(0, x - self.tolerance)
        x_max = min(w, x + self.tolerance + 1)
        
        region = mask[y_min:y_max, x_min:x_max]
        
        return bool(region.any())
    
    def evaluate_batch(
        self,
        saliency_maps: torch.Tensor,
        ground_truth_masks: torch.Tensor
    ) -> dict:
        """
        Evaluate over a batch of samples.
        
        Args:
            saliency_maps: Batch of saliency maps (N, H, W)
            ground_truth_masks: Batch of masks (N, H, W)
            
        Returns:
            Dictionary with accuracy and counts
        """
        hits = 0
        total = len(saliency_maps)
        
        for saliency, mask in zip(saliency_maps, ground_truth_masks):
            hits += self.evaluate(saliency, mask)
        
        return {
            'accuracy': hits / total,
            'hits': hits,
            'total': total
        }


def pointing_game(
    saliency_map: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    tolerance: int = 15
) -> float:
    """
    Convenience function for pointing game evaluation.
    
    Args:
        saliency_map: Explanation map of shape (H, W)
        ground_truth_mask: Binary mask of shape (H, W)
        tolerance: Pixel tolerance
        
    Returns:
        Hit rate (1.0 or 0.0)
    """
    pg = PointingGame(tolerance=tolerance)
    return pg.evaluate(saliency_map, ground_truth_mask)
