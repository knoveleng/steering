"""
Base classes for feature direction calculation
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch


class BaseFeatureDirectionCalculator(ABC):
    """Abstract base class for feature direction calculators"""
    
    @abstractmethod
    def compute_candidate_directions(
        self,
        positive_activations: Dict[str, torch.Tensor],
        negative_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute candidate feature directions at each layer
        
        Args:
            positive_activations: Activations from positive class
            negative_activations: Activations from negative class
            
        Returns:
            Dictionary mapping layer names to direction vectors
        """
        pass
    
    @abstractmethod
    def select_best_direction(
        self,
        candidates: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, str]:
        """
        Select the best feature direction from candidates
        
        Args:
            candidates: Candidate directions
            
        Returns:
            Tuple of (best_direction, layer_name)
        """
        pass