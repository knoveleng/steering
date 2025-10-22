"""
Base classes for steering operators
"""

from abc import ABC, abstractmethod
import torch


class BaseSteeringOperator(ABC):
    """Abstract base class for steering operators"""
    
    @abstractmethod
    def steer_activation(
        self,
        activation: torch.Tensor,
        theta: float
    ) -> torch.Tensor:
        """
        Apply steering to activation
        
        Args:
            activation: Input activation tensor
            theta: Steering parameter (e.g., angle in degrees)
            
        Returns:
            Steered activation tensor
        """
        pass
    
    @abstractmethod
    def reset_cache(self) -> None:
        """Clear any cached computations"""
        pass