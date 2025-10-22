"""
Base classes for activation extraction
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch


class BaseActivationExtractor(ABC):
    """Abstract base class for activation extractors"""
    
    @abstractmethod
    def register_hooks(self, layer_names: List[str]) -> None:
        """
        Register forward hooks on specified layers
        
        Args:
            layer_names: List of layer names to hook
        """
        pass
    
    @abstractmethod
    def extract_activations(
        self,
        prompts: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations for given prompts
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        pass
    
    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        pass
    
    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Get all available layer names"""
        pass