"""
Base Explainer class that all explainability methods inherit from.
Provides a unified interface for generating explanations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Union, List


class BaseExplainer(ABC):
    """
    Abstract base class for all explainability methods.
    
    Args:
        model: PyTorch model to explain
        device: Device to run computations on ('cuda' or 'cpu')
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    @abstractmethod
    def explain(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate explanation for the input.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W) or (C, H, W)
            target_class: Target class index. If None, uses predicted class
            **kwargs: Method-specific parameters
            
        Returns:
            Explanation heatmap of shape (H, W)
        """
        pass
    
    def _preprocess_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Ensure input is in correct format (1, C, H, W) and on correct device."""
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return input_tensor.to(self.device)
    
    def _get_target_class(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> int:
        """Get target class, using model prediction if not specified."""
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        return target_class
    
    def _normalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Normalize heatmap to [0, 1] range."""
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__}, device={self.device})"
