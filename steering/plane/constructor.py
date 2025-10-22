"""
Steering plane construction implementation
"""

import torch
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
import numpy as np

from .base import BasePlaneConstructor


class SteeringPlaneConstructor(BasePlaneConstructor):
    """
    Construct 2D steering plane using PCA
    """
    
    def __init__(self, feature_direction: Optional[torch.Tensor] = None):
        """
        Initialize constructor
        
        Args:
            feature_direction: Optional pre-selected feature direction
        """
        self.feature_direction = feature_direction
        self.b1 = None
        self.b2 = None
        self.projection_matrix = None
        self.original_dtype = None
    
    def construct_plane(
        self,
        feature_direction: torch.Tensor,
        candidates: Dict[str, torch.Tensor]
    ) -> None:
        """
        Construct plane using feature direction and PCA
        
        Args:
            feature_direction: Selected feature direction (unit vector)
            candidates: All candidate directions
        """
        # Store original dtype
        self.original_dtype = feature_direction.dtype
        
        self.feature_direction = feature_direction
        
        # Stack candidates into matrix
        candidate_matrix = torch.stack(list(candidates.values()))
        
        # Convert to float32 for sklearn compatibility
        candidate_matrix_np = candidate_matrix.float().cpu().numpy()
        
        # Perform PCA to get first principal component
        pca = PCA(n_components=1)
        pca.fit(candidate_matrix_np)
        
        # Get first principal component and convert back to original dtype
        d_pc0 = torch.from_numpy(pca.components_[0]).to(
            feature_direction.device,
            dtype=self.original_dtype
        )
        
        # Construct orthonormal basis using Gram-Schmidt
        self.b1 = feature_direction / (feature_direction.norm() + 1e-8)
        
        # Orthogonalize d_pc0 with respect to b1
        self.b2 = d_pc0 - torch.dot(d_pc0, self.b1) * self.b1
        self.b2 = self.b2 / (self.b2.norm() + 1e-8)
        
        # Construct projection matrix
        self.projection_matrix = (
            torch.outer(self.b1, self.b1) + 
            torch.outer(self.b2, self.b2)
        )
    
    def get_basis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get orthonormal basis vectors"""
        if self.b1 is None or self.b2 is None:
            raise RuntimeError("Plane not constructed. Call construct_plane() first.")
        
        return self.b1, self.b2
    
    def get_projection_matrix(self) -> torch.Tensor:
        """Get projection matrix"""
        if self.projection_matrix is None:
            raise RuntimeError("Plane not constructed. Call construct_plane() first.")
        
        return self.projection_matrix
    
    def project_onto_plane(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Project vector onto steering plane
        
        Args:
            vector: Vector to project
            
        Returns:
            Projected vector
        """
        P = self.get_projection_matrix()
        # Ensure dtype matches
        P = P.to(vector.device, dtype=vector.dtype)
        return torch.matmul(vector, P.T)
    
    def decompose_in_basis(
        self,
        vector: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Decompose vector in terms of basis {b1, b2}
        
        Args:
            vector: Vector to decompose
            
        Returns:
            Tuple of (coeff_b1, coeff_b2)
        """
        b1, b2 = self.get_basis()
        
        # Ensure dtype matches
        b1 = b1.to(vector.device, dtype=vector.dtype)
        b2 = b2.to(vector.device, dtype=vector.dtype)
        
        coeff_b1 = torch.dot(vector, b1).item()
        coeff_b2 = torch.dot(vector, b2).item()
        
        return coeff_b1, coeff_b2

    def project_candidates_onto_plane(
        self,
        candidates: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Project all candidate directions onto the steering plane
        
        Args:
            candidates: Dictionary of candidate directions
            
        Returns:
            Dictionary mapping layer names to (coeff_b1, coeff_b2)
        """
        projections = {}
        
        for layer_name, direction in candidates.items():
            coeff_b1, coeff_b2 = self.decompose_in_basis(direction)
            projections[layer_name] = (coeff_b1, coeff_b2)
        
        return projections