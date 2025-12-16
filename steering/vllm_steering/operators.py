"""
vLLM Steering Operators

Provides steering operator classes for vLLM activation steering.
Each class implements a specific steering mode with optimized caching.
"""

import math
import torch
from typing import Dict, Optional, Tuple
from enum import Enum


class SteeringMode(str, Enum):
    """Steering mode enumeration."""
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    SELECTIVE = "selective"
    
    @classmethod
    def from_string(cls, mode: str) -> "SteeringMode":
        """Convert string to SteeringMode."""
        mode_lower = mode.lower()
        for m in cls:
            if m.value == mode_lower:
                return m
        raise ValueError(f"Unknown steering mode: {mode}. Valid modes: {[m.value for m in cls]}")


# Default values
DEFAULT_MODE = SteeringMode.STANDARD
DEFAULT_THRESHOLD = 0.0


class BaseSteeringOperator:
    """
    Base class for vLLM steering operators.
    
    Provides common functionality for caching device-specific tensors
    and rotation computations.
    """
    
    def __init__(self, b1: torch.Tensor, b2: torch.Tensor):
        """
        Initialize operator with basis vectors.
        
        Args:
            b1: First basis vector (feature direction)
            b2: Second basis vector (orthogonal)
        """
        self.b1 = b1
        self.b2 = b2
        
        # Precompute projection matrix: P = b1⊗b1^T + b2⊗b2^T
        self.P = torch.outer(b1, b1) + torch.outer(b2, b2)
        
        # Cache for device-specific tensors and rotations
        self._device_cache: Dict[Tuple, Dict] = {}
        self._rotation_cache: Dict[Tuple, torch.Tensor] = {}
    
    def _get_device_tensors(self, device: torch.device, dtype: torch.dtype) -> Dict:
        """Get or create cached device-specific tensors."""
        key = (device, dtype)
        if key not in self._device_cache:
            self._device_cache[key] = {
                'P': self.P.to(device=device, dtype=dtype),
                'b1': self.b1.to(device=device, dtype=dtype),
                'b2': self.b2.to(device=device, dtype=dtype),
            }
        return self._device_cache[key]
    
    def steer(
        self,
        hidden_states: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply steering to hidden states.
        
        Args:
            hidden_states: Tensor of shape (..., hidden_dim)
            theta: Steering angle in degrees
            layer_name: Optional layer name for selective mode
            
        Returns:
            Steered hidden states
        """
        raise NotImplementedError
    
    def clear_cache(self):
        """Clear all cached tensors."""
        self._device_cache.clear()
        self._rotation_cache.clear()


class StandardSteeringOperator(BaseSteeringOperator):
    """
    Standard angular steering operator.
    
    Formula: h' = h - P*h + ||P*h|| * v_theta
    where v_theta = cos(θ)*b1 + sin(θ)*b2
    """
    
    def _get_rotation_vector(
        self, 
        theta: float, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Get cached rotation vector v_theta = cos(θ)*b1 + sin(θ)*b2."""
        key = (device, dtype, theta, 'vector')
        if key not in self._rotation_cache:
            cached = self._get_device_tensors(device, dtype)
            theta_rad = math.radians(theta)
            self._rotation_cache[key] = (
                math.cos(theta_rad) * cached['b1'] + 
                math.sin(theta_rad) * cached['b2']
            )
        return self._rotation_cache[key]
    
    def steer(
        self,
        hidden_states: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply standard steering: h' = h - P*h + ||P*h|| * v_theta
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        cached = self._get_device_tensors(device, dtype)
        v_theta = self._get_rotation_vector(theta, device, dtype)
        
        # Project onto steering plane
        proj_h = torch.matmul(hidden_states, cached['P'].T)
        
        # Compute magnitude
        r = torch.linalg.norm(proj_h, dim=-1, keepdim=True)
        
        # Apply steering
        return hidden_states - proj_h + r * v_theta


class AdaptiveSteeringOperator(StandardSteeringOperator):
    """
    Adaptive steering with conditional masking.
    
    Formula: h' = h + mask * (||P*h|| * v_theta - P*h)
    where mask = (alignment > threshold)
    """
    
    def __init__(self, b1: torch.Tensor, b2: torch.Tensor, threshold: float = 0.0):
        super().__init__(b1, b2)
        self.threshold = threshold
    
    def steer(
        self,
        hidden_states: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply adaptive steering: h' = h + mask * (||P*h|| * v_theta - P*h)
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        cached = self._get_device_tensors(device, dtype)
        v_theta = self._get_rotation_vector(theta, device, dtype)
        
        # Compute alignment with feature direction
        alignment = torch.matmul(hidden_states, cached['b1'])
        mask = (alignment > self.threshold).to(dtype=dtype).unsqueeze(-1)
        
        # Project onto steering plane
        proj_h = torch.matmul(hidden_states, cached['P'].T)
        
        # Compute magnitude
        r = torch.linalg.norm(proj_h, dim=-1, keepdim=True)
        
        # Apply masked steering
        return hidden_states + mask * (r * v_theta - proj_h)


class SelectiveSteeringOperator(BaseSteeringOperator):
    """
    Selective steering with layer-specific rotation matrix.
    
    Formula: h' = h - P*h + V_θ*h
    where V_θ = cos(θ)*P + sin(θ)*(b2⊗b1^T - b1⊗b2^T)
    
    Only steers on layers where the mask is True.
    """
    
    def __init__(
        self, 
        b1: torch.Tensor, 
        b2: torch.Tensor, 
        layer_mask: Optional[Dict[str, bool]] = None
    ):
        super().__init__(b1, b2)
        self.layer_mask = layer_mask or {}
    
    def _get_rotation_matrix(
        self, 
        theta: float, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Get cached rotation matrix V_θ."""
        key = (device, dtype, theta, 'matrix')
        if key not in self._rotation_cache:
            cached = self._get_device_tensors(device, dtype)
            theta_rad = math.radians(theta)
            cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
            self._rotation_cache[key] = (
                cos_t * cached['P'] + 
                sin_t * (torch.outer(cached['b2'], cached['b1']) - 
                         torch.outer(cached['b1'], cached['b2']))
            )
        return self._rotation_cache[key]
    
    def should_steer(self, layer_name: str) -> bool:
        """Check if this layer should be steered."""
        if not self.layer_mask:
            return True
        return self.layer_mask.get(layer_name, False)
    
    def steer(
        self,
        hidden_states: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply selective steering: h' = h - P*h + V_θ*h
        Only steers if layer is in mask.
        """
        # Check layer mask
        if layer_name and not self.should_steer(layer_name):
            return hidden_states
        
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        cached = self._get_device_tensors(device, dtype)
        V_theta = self._get_rotation_matrix(theta, device, dtype)
        
        # Project onto steering plane
        proj_h = torch.matmul(hidden_states, cached['P'].T)
        
        # Apply rotation
        return hidden_states - proj_h + torch.matmul(hidden_states, V_theta.T)


def create_operator(
    mode: str,
    b1: torch.Tensor,
    b2: torch.Tensor,
    threshold: float = 0.0,
    layer_mask: Optional[Dict[str, bool]] = None,
) -> BaseSteeringOperator:
    """
    Factory function to create the appropriate operator.
    
    Args:
        mode: Steering mode ("standard", "adaptive", "selective")
        b1: First basis vector
        b2: Second basis vector
        threshold: Alignment threshold for adaptive mode
        layer_mask: Layer selection mask for selective mode
        
    Returns:
        Steering operator instance
    """
    mode_enum = SteeringMode.from_string(mode)
    
    if mode_enum == SteeringMode.ADAPTIVE:
        return AdaptiveSteeringOperator(b1, b2, threshold)
    elif mode_enum == SteeringMode.SELECTIVE:
        return SelectiveSteeringOperator(b1, b2, layer_mask)
    else:
        return StandardSteeringOperator(b1, b2)
