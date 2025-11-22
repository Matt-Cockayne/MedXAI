"""
Faithfulness metrics for evaluating explanation quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from scipy.stats import spearmanr, pearsonr


class FaithfulnessMetrics:
    """
    Collection of faithfulness metrics for explanation evaluation.
    
    Faithfulness measures how well the explanation reflects the model's 
    actual decision-making process.
    
    Args:
        model: PyTorch model
        device: Device to run computations on
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def sensitivity_n(
        self,
        input_tensor: torch.Tensor,
        saliency_map: torch.Tensor,
        target_class: int,
        n_samples: int = 100,
        perturbation_size: float = 0.1
    ) -> float:
        """
        Sensitivity-n: Correlation between changes in explanation and output.
        Ancona et al., "Towards better understanding of gradient-based 
        attribution methods", ICLR 2018
        
        Args:
            input_tensor: Input image
            saliency_map: Explanation map
            target_class: Target class
            n_samples: Number of perturbed samples
            perturbation_size: Size of perturbations
            
        Returns:
            Correlation coefficient
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        saliency_map = saliency_map.to(self.device)
        
        # Get baseline output
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            baseline_prob = F.softmax(baseline_output, dim=1)[0, target_class].item()
        
        explanation_diffs = []
        output_diffs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Create perturbation
                noise = torch.randn_like(input_tensor) * perturbation_size
                perturbed = input_tensor + noise
                
                # Get perturbed output
                perturbed_output = self.model(perturbed)
                perturbed_prob = F.softmax(perturbed_output, dim=1)[0, target_class].item()
                
                # Calculate output difference
                output_diff = abs(perturbed_prob - baseline_prob)
                
                # Calculate explanation difference (L2 norm of perturbation weighted by saliency)
                weighted_noise = noise.squeeze() * saliency_map.unsqueeze(0)
                explanation_diff = torch.norm(weighted_noise).item()
                
                explanation_diffs.append(explanation_diff)
                output_diffs.append(output_diff)
        
        # Calculate correlation
        if len(set(explanation_diffs)) > 1 and len(set(output_diffs)) > 1:
            correlation, _ = spearmanr(explanation_diffs, output_diffs)
            return correlation
        else:
            return 0.0
    
    def infidelity(
        self,
        input_tensor: torch.Tensor,
        saliency_map: torch.Tensor,
        target_class: int,
        n_samples: int = 100,
        perturbation_size: float = 0.1
    ) -> float:
        """
        Infidelity: Expected MSE between explanation and true input contribution.
        Yeh et al., "On the (In)fidelity and Sensitivity of Explanations", NeurIPS 2019
        
        Lower is better.
        
        Args:
            input_tensor: Input image
            saliency_map: Explanation map
            target_class: Target class
            n_samples: Number of samples
            perturbation_size: Perturbation magnitude
            
        Returns:
            Infidelity score
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        saliency_map = saliency_map.to(self.device)
        
        infidelities = []
        
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            baseline_logit = baseline_output[0, target_class].item()
            
            for _ in range(n_samples):
                # Generate perturbation
                perturbation = torch.randn_like(input_tensor) * perturbation_size
                
                # Perturbed input
                perturbed = input_tensor - perturbation
                perturbed_output = self.model(perturbed)
                perturbed_logit = perturbed_output[0, target_class].item()
                
                # True contribution
                true_contrib = baseline_logit - perturbed_logit
                
                # Explanation-based contribution
                expl_contrib = torch.sum(
                    saliency_map.unsqueeze(0) * perturbation.squeeze()
                ).item()
                
                # Squared error
                infidelities.append((true_contrib - expl_contrib) ** 2)
        
        return np.mean(infidelities)
    
    def monotonicity(
        self,
        input_tensor: torch.Tensor,
        saliency_map: torch.Tensor,
        target_class: int,
        n_steps: int = 50
    ) -> float:
        """
        Monotonicity: Measures if removing features decreases confidence monotonically.
        
        Higher is better (1.0 is perfect monotonicity).
        
        Args:
            input_tensor: Input image
            saliency_map: Explanation map
            target_class: Target class
            n_steps: Number of steps
            
        Returns:
            Monotonicity score
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        saliency_map = saliency_map.to(self.device)
        
        # Get pixel ordering by importance
        h, w = saliency_map.shape
        flat_saliency = saliency_map.flatten()
        sorted_indices = torch.argsort(flat_saliency, descending=True)
        
        total_pixels = h * w
        step_size = max(1, total_pixels // n_steps)
        
        probs = []
        modified = input_tensor.clone()
        
        with torch.no_grad():
            # Get initial probability
            output = self.model(modified)
            prob = F.softmax(output, dim=1)[0, target_class].item()
            probs.append(prob)
            
            # Progressively mask pixels
            for i in range(0, total_pixels, step_size):
                end_idx = min(i + step_size, total_pixels)
                indices = sorted_indices[i:end_idx]
                
                y_coords = indices // w
                x_coords = indices % w
                
                modified[:, :, y_coords, x_coords] = 0
                
                output = self.model(modified)
                prob = F.softmax(output, dim=1)[0, target_class].item()
                probs.append(prob)
        
        # Count monotonic decreases
        probs = np.array(probs)
        diffs = np.diff(probs)
        monotonic_decreases = np.sum(diffs <= 0)
        
        return monotonic_decreases / len(diffs) if len(diffs) > 0 else 0.0
    
    def evaluate_all(
        self,
        input_tensor: torch.Tensor,
        saliency_map: torch.Tensor,
        target_class: int
    ) -> dict:
        """
        Evaluate all faithfulness metrics.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'sensitivity_n': self.sensitivity_n(input_tensor, saliency_map, target_class),
            'infidelity': self.infidelity(input_tensor, saliency_map, target_class),
            'monotonicity': self.monotonicity(input_tensor, saliency_map, target_class)
        }
