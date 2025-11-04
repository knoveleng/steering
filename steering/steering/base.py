"""
Base classes for steering operators
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseSteeringOperator(ABC):
    """Abstract base class for steering operators"""

    @abstractmethod
    def steer_activation(
        self,
        activation: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Apply steering to activation

        Args:
            activation: Input activation tensor
            theta: Steering parameter (e.g., angle in degrees)
            layer_name: Optional layer name for selective steering

        Returns:
            Steered activation tensor
        """
        pass
    
    @abstractmethod
    def reset_cache(self) -> None:
        """Clear any cached computations"""
        pass