"""
Plausibility metrics for evaluating human-interpretable quality
"""

import torch
import numpy as np
from typing import Optional
from scipy.stats import spearmanr


class PlausibilityMetrics:
    """
    Plausibility metrics evaluate how well explanations align with 
    human understanding and annotations.
    
    These metrics require ground truth annotations (e.g., segmentation masks,
    bounding boxes, or human attention maps).
    """
    
    def __init__(self):
        pass
    
    def iou_score(
        self,
        saliency_map: torch.Tensor,
        ground_truth_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Intersection over Union between thresholded saliency and ground truth.
        
        Args:
            saliency_map: Explanation map (H, W), normalized to [0, 1]
            ground_truth_mask: Binary ground truth mask (H, W)
            threshold: Threshold for binarizing saliency map
            
        Returns:
            IoU score
        """
        # Binarize saliency map
        binary_saliency = (saliency_map > threshold).float()
        
        # Calculate intersection and union
        intersection = (binary_saliency * ground_truth_mask).sum()
        union = ((binary_saliency + ground_truth_mask) > 0).float().sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def precision_recall_f1(
        self,
        saliency_map: torch.Tensor,
        ground_truth_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> dict:
        """
        Precision, Recall, and F1 score.
        
        Args:
            saliency_map: Explanation map (H, W)
            ground_truth_mask: Binary ground truth mask (H, W)
            threshold: Threshold for binarizing saliency map
            
        Returns:
            Dictionary with precision, recall, and F1
        """
        binary_saliency = (saliency_map > threshold).float()
        
        tp = (binary_saliency * ground_truth_mask).sum().item()
        fp = (binary_saliency * (1 - ground_truth_mask)).sum().item()
        fn = ((1 - binary_saliency) * ground_truth_mask).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def relevance_rank_correlation(
        self,
        saliency_map: torch.Tensor,
        ground_truth_map: torch.Tensor
    ) -> float:
        """
        Spearman rank correlation between saliency and ground truth importance.
        
        Useful when ground truth is a continuous importance map (e.g., from eye tracking).
        
        Args:
            saliency_map: Explanation map (H, W)
            ground_truth_map: Ground truth importance map (H, W)
            
        Returns:
            Spearman correlation coefficient
        """
        sal_flat = saliency_map.flatten().cpu().numpy()
        gt_flat = ground_truth_map.flatten().cpu().numpy()
        
        # Remove constant values
        if len(set(sal_flat)) <= 1 or len(set(gt_flat)) <= 1:
            return 0.0
        
        correlation, _ = spearmanr(sal_flat, gt_flat)
        return correlation
    
    def top_k_intersection(
        self,
        saliency_map: torch.Tensor,
        ground_truth_mask: torch.Tensor,
        k: int = 100
    ) -> float:
        """
        Intersection of top-k most salient pixels with ground truth.
        
        Args:
            saliency_map: Explanation map (H, W)
            ground_truth_mask: Binary ground truth mask (H, W)
            k: Number of top pixels to consider
            
        Returns:
            Intersection ratio
        """
        # Get top-k pixels from saliency
        flat_saliency = saliency_map.flatten()
        _, top_indices = torch.topk(flat_saliency, k)
        
        # Create binary mask for top-k
        top_k_mask = torch.zeros_like(flat_saliency)
        top_k_mask[top_indices] = 1
        top_k_mask = top_k_mask.reshape(saliency_map.shape)
        
        # Calculate intersection
        intersection = (top_k_mask * ground_truth_mask).sum().item()
        
        return intersection / k
    
    def mass_accuracy(
        self,
        saliency_map: torch.Tensor,
        ground_truth_mask: torch.Tensor,
        threshold: float = 0.9
    ) -> float:
        """
        Fraction of saliency mass that falls within ground truth region.
        
        Args:
            saliency_map: Explanation map (H, W)
            ground_truth_mask: Binary ground truth mask (H, W)
            threshold: What fraction of mass to consider
            
        Returns:
            Mass accuracy score
        """
        # Sort pixels by saliency
        flat_saliency = saliency_map.flatten()
        flat_mask = ground_truth_mask.flatten()
        
        sorted_indices = torch.argsort(flat_saliency, descending=True)
        
        # Accumulate mass until threshold
        total_mass = flat_saliency.sum().item()
        target_mass = total_mass * threshold
        
        accumulated_mass = 0
        hits = 0
        total = 0
        
        for idx in sorted_indices:
            idx = int(idx)
            accumulated_mass += flat_saliency[idx].item()
            total += 1
            
            if flat_mask[idx] > 0:
                hits += 1
            
            if accumulated_mass >= target_mass:
                break
        
        return hits / total if total > 0 else 0.0
    
    def evaluate_all(
        self,
        saliency_map: torch.Tensor,
        ground_truth_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> dict:
        """
        Evaluate all plausibility metrics.
        
        Returns:
            Dictionary with all metrics
        """
        pr_results = self.precision_recall_f1(saliency_map, ground_truth_mask, threshold)
        
        return {
            'iou': self.iou_score(saliency_map, ground_truth_mask, threshold),
            'precision': pr_results['precision'],
            'recall': pr_results['recall'],
            'f1': pr_results['f1'],
            'rank_correlation': self.relevance_rank_correlation(saliency_map, ground_truth_mask),
            'top_100_intersection': self.top_k_intersection(saliency_map, ground_truth_mask, k=100),
            'mass_accuracy': self.mass_accuracy(saliency_map, ground_truth_mask)
        }
