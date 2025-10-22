"""
Steering operator implementations
"""

import torch
import math
from typing import Optional, Dict

from .base import BaseSteeringOperator


class AngularSteeringOperator(BaseSteeringOperator):
    """
    Standard angular steering operator
    """
    
    def __init__(
        self,
        b1: torch.Tensor,
        b2: torch.Tensor,
        cache_rotations: bool = True
    ):
        """
        Initialize operator
        
        Args:
            b1: First basis vector (feature direction)
            b2: Second basis vector (orthogonal)
            cache_rotations: Whether to cache precomputed rotations
        """
        self.b1 = b1
        self.b2 = b2
        
        # Precompute projection matrix
        self.P = torch.outer(b1, b1) + torch.outer(b2, b2)
        
        # Rotation cache - store per dtype
        self.cache_rotations = cache_rotations
        self.rotation_cache: Dict[tuple, torch.Tensor] = {}
    
    def _precompute_rotation(self, theta_degrees: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Precompute rotation vector for angle with specific dtype
        
        Args:
            theta_degrees: Rotation angle in degrees
            dtype: Target dtype
            device: Target device
            
        Returns:
            Rotated unit vector in steering plane
        """
        cache_key = (theta_degrees, dtype, device)
        
        if self.cache_rotations and cache_key in self.rotation_cache:
            return self.rotation_cache[cache_key]
        
        # Convert to radians
        theta = math.radians(theta_degrees)
        
        # Get basis vectors with correct dtype and device
        b1 = self.b1.to(device=device, dtype=dtype)
        b2 = self.b2.to(device=device, dtype=dtype)
        
        # Compute [b1, b2] * R_theta * [1, 0]^T
        # This simplifies to: cos(theta)*b1 + sin(theta)*b2
        v_theta = math.cos(theta) * b1 + math.sin(theta) * b2
        
        if self.cache_rotations:
            self.rotation_cache[cache_key] = v_theta
        
        return v_theta
    
    def steer_activation(
        self,
        activation: torch.Tensor,
        theta: float
    ) -> torch.Tensor:
        """
        Apply angular steering
        
        Args:
            activation: Tensor of shape (..., hidden_dim)
            theta: Target angle in degrees
            
        Returns:
            Steered activation of same shape and dtype
        """
        # Store original dtype and device
        original_dtype = activation.dtype
        original_device = activation.device
        
        # Get basis vectors and projection matrix with correct dtype and device
        P = self.P.to(device=original_device, dtype=original_dtype)
        
        # Project onto steering plane
        proj_h = torch.matmul(activation, P.T)
        
        # Compute magnitude (keeping dtype)
        r = torch.linalg.norm(proj_h, dim=-1, keepdim=True)
        r = r.to(dtype=original_dtype)  # Ensure norm doesn't change dtype
        
        # Get precomputed rotation vector with correct dtype
        v_theta = self._precompute_rotation(theta, original_dtype, original_device)
        
        # Apply steering: h' = h - proj_P(h) + r * v_theta
        h_steered = activation - proj_h + r * v_theta
        
        # Ensure output has same dtype as input
        h_steered = h_steered.to(dtype=original_dtype)
        
        return h_steered
    
    def reset_cache(self) -> None:
        """Clear rotation cache"""
        self.rotation_cache.clear()


class AdaptiveSteeringOperator(AngularSteeringOperator):
    """
    Adaptive angular steering with conditional masking
    """
    
    def __init__(
        self,
        b1: torch.Tensor,
        b2: torch.Tensor,
        threshold: float = 0.0,
        cache_rotations: bool = True
    ):
        """
        Initialize adaptive operator
        
        Args:
            b1: First basis vector (feature direction)
            b2: Second basis vector (orthogonal)
            threshold: Alignment threshold for masking
            cache_rotations: Whether to cache rotations
        """
        super().__init__(b1, b2, cache_rotations)
        self.threshold = threshold
    
    def steer_activation(
        self,
        activation: torch.Tensor,
        theta: float
    ) -> torch.Tensor:
        """
        Apply adaptive steering with masking
        
        Args:
            activation: Tensor of shape (..., hidden_dim)
            theta: Target angle in degrees
            
        Returns:
            Steered activation (only aligned activations are rotated)
        """
        # Store original dtype and device
        original_dtype = activation.dtype
        original_device = activation.device
        
        # Get basis vectors with correct dtype and device
        b1 = self.b1.to(device=original_device, dtype=original_dtype)
        P = self.P.to(device=original_device, dtype=original_dtype)
        
        # Compute alignment with feature direction
        alignment = torch.matmul(activation, b1)
        
        # Create mask: only steer activations aligned with feature
        mask = (alignment > self.threshold).to(dtype=original_dtype)
        mask = mask.unsqueeze(-1)  # Add dimension for broadcasting
        
        # Project onto steering plane
        proj_h = torch.matmul(activation, P.T)
        
        # Compute magnitude (keeping dtype)
        r = torch.linalg.norm(proj_h, dim=-1, keepdim=True)
        r = r.to(dtype=original_dtype)  # Ensure norm doesn't change dtype
        
        # Get rotation vector with correct dtype
        v_theta = self._precompute_rotation(theta, original_dtype, original_device)
        
        # Apply conditional steering
        h_steered = activation + mask * (r * v_theta - proj_h)
        
        # Ensure output has same dtype as input
        h_steered = h_steered.to(dtype=original_dtype)
        
        return h_steered


class HouseholderSteeringOperator(BaseSteeringOperator):
    """
    Steering using Householder reflections (alternative method)
    """
    
    def __init__(self, feature_direction: torch.Tensor):
        """
        Initialize Householder operator
        
        Args:
            feature_direction: Direction to reflect across
        """
        self.direction = feature_direction / (feature_direction.norm() + 1e-8)
    
    def steer_activation(
        self,
        activation: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply Householder reflection with scaling
        
        Args:
            activation: Input activation
            alpha: Scaling factor for reflection
            
        Returns:
            Reflected and scaled activation
        """
        # Store original dtype
        original_dtype = activation.dtype
        
        # Move direction to same device and dtype
        v = self.direction.to(device=activation.device, dtype=original_dtype)
        
        # Compute Householder reflection
        # H = I - 2*v*v^T where v is unit direction
        
        # Compute reflection: h' = h - 2*(h.v)*v
        proj = torch.matmul(activation, v)
        reflection = activation - 2 * proj.unsqueeze(-1) * v
        
        # Scale reflection
        h_steered = activation + alpha * (reflection - activation)
        
        # Ensure output dtype matches input
        h_steered = h_steered.to(dtype=original_dtype)
        
        return h_steered
    
    def reset_cache(self) -> None:
        """No cache for Householder"""
        pass