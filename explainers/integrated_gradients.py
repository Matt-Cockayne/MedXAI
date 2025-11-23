"""
Integrated Gradients: Path-based attribution method
Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from .base import BaseExplainer


class IntegratedGradients(BaseExplainer):
    """
    Integrated Gradients implementation for attribution.
    
    Computes gradients along a path from a baseline to the input,
    integrating them to get pixel-wise attributions.
    
    Args:
        model: PyTorch model to explain
        device: Device to run computations on
        baseline: Baseline input (default: zeros)
        steps: Number of interpolation steps
    """
    
    def __init__(
        self, 
        model: nn.Module,
        device: Optional[str] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ):
        super().__init__(model, device)
        self.baseline = baseline
        self.steps = steps
    
    def explain(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        normalize: bool = True,
        baseline: Optional[torch.Tensor] = None,
        steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Integrated Gradients attribution.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            normalize: Whether to normalize the attribution map
            baseline: Custom baseline (overrides default)
            steps: Number of integration steps (overrides default)
            
        Returns:
            Attribution map of shape (H, W)
        """
        input_tensor = self._preprocess_input(input_tensor)
        target_class = self._get_target_class(input_tensor, target_class)
        
        # Use provided baseline or default
        if baseline is None:
            baseline = self.baseline
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = self._preprocess_input(baseline)
        
        steps = steps or self.steps
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps + 1, device=self.device)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)
        
        interpolated_inputs = torch.cat(interpolated_inputs, dim=0)
        # Clone and detach to create a new leaf variable
        interpolated_inputs = interpolated_inputs.clone().detach().requires_grad_(True)
        
        # Forward pass on all interpolated inputs
        outputs = self.model(interpolated_inputs)
        
        # Get gradients for target class
        self.model.zero_grad()
        target_outputs = outputs[:, target_class]
        target_outputs.sum().backward()
        
        gradients = interpolated_inputs.grad  # (steps+1, C, H, W)
        
        # Approximate the integral using trapezoidal rule
        avg_gradients = torch.mean(gradients, dim=0, keepdim=True)  # (1, C, H, W)
        
        # Multiply by input difference
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        # Sum across channels to get pixel-wise attributions
        attribution = torch.sum(torch.abs(integrated_gradients), dim=1)  # (1, H, W)
        attribution = attribution.squeeze()  # (H, W)
        
        if normalize:
            attribution = self._normalize_heatmap(attribution)
        
        return attribution.cpu()
    
    def explain_smooth(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_samples: int = 50,
        noise_level: float = 0.15,
        **kwargs
    ) -> torch.Tensor:
        """
        SmoothGrad version of Integrated Gradients for reduced noise.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            n_samples: Number of noisy samples
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Smoothed attribution map
        """
        input_tensor = self._preprocess_input(input_tensor)
        
        attributions = []
        for _ in range(n_samples):
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise
            attribution = self.explain(noisy_input, target_class, **kwargs)
            attributions.append(attribution)
        
        # Average attributions
        smooth_attribution = torch.stack(attributions).mean(dim=0)
        
        return smooth_attribution
