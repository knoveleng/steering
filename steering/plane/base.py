"""
Base classes for steering plane construction
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch


class BasePlaneConstructor(ABC):
    """Abstract base class for steering plane constructors"""
    
    @abstractmethod
    def construct_plane(
        self,
        feature_direction: torch.Tensor,
        candidates: Dict[str, torch.Tensor],
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
    ) -> None:
        """
        Construct steering plane from feature direction and candidates
        
        Args:
            feature_direction: The selected feature direction
            candidates: All candidate directions
        """
        pass
    
    @abstractmethod
    def get_basis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get orthonormal basis of the plane
        
        Returns:
            Tuple of (b1, b2) basis vectors
        """
        pass
    
    @abstractmethod
    def get_projection_matrix(self) -> torch.Tensor:
        """
        Get projection matrix onto the plane
        
        Returns:
            Projection matrix P = b1*b1^T + b2*b2^T
        """
        pass