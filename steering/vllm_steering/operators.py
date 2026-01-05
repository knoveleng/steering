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
    ADDITION = "addition"
    ABLATION = "ablation"
    
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
    
    def clear_rotation_cache(self):
        """Clear only rotation cache (not device tensors).
        
        Use this when theta changes to avoid OOM from accumulating rotation
        vectors/matrices, while preserving device tensors to maintain
        floating-point consistency.
        """
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
        # Normalize theta to [0, 360) so 0° and 360° are treated identically
        theta_normalized = theta % 360
        key = (device, dtype, theta_normalized, 'vector')
        if key not in self._rotation_cache:
            cached = self._get_device_tensors(device, dtype)
            theta_rad = math.radians(theta_normalized)
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
        # Normalize theta to [0, 360) so 0° and 360° are treated identically
        theta_normalized = theta % 360
        key = (device, dtype, theta_normalized, 'matrix')
        if key not in self._rotation_cache:
            cached = self._get_device_tensors(device, dtype)
            theta_rad = math.radians(theta_normalized)
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


class AdditionSteeringOperator(BaseSteeringOperator):
    """
    Vector addition steering operator - a special case of Angular Steering.
    
    From baselines.md, Addition h' = h + α·d produces an equivalent rotation.
    
    To match StandardSteeringOperator's behavior where θ is the absolute angle
    from b1 (not a relative offset), we compute α such that the result points
    in the same direction as the standard steering at angle θ.
    
    For absolute angle θ, the target direction is:
        cos(θ)·b1 + sin(θ)·b2
        
    Since Addition only adds along b1, we compute α to achieve the same
    projection ratio ||h_parallel|| / ||h'|| = cos(θ).
    """
    
    def steer(
        self,
        hidden_states: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply vector addition steering to achieve absolute angle θ.
        
        Given theta (absolute angle in degrees from b1), computes α and applies:
            h' = h + α·d
            
        The result is normalized to preserve the magnitude on the steering plane.
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        cached = self._get_device_tensors(device, dtype)
        d = cached['b1']  # Feature direction (d = b1)
        b2 = cached['b2']  # Orthogonal direction
        
        # Convert theta to radians
        theta_rad = math.radians(theta)
        
        # Compute projection onto the 2D plane spanned by b1 and b2
        h_dot_b1 = torch.matmul(hidden_states, d)  # Projection onto b1
        h_dot_b2 = torch.matmul(hidden_states, b2)  # Projection onto b2
        
        # Compute the norm of projection onto the plane: r = sqrt(h·b1² + h·b2²)
        r = torch.sqrt(h_dot_b1**2 + h_dot_b2**2)
        
        # Target projection onto b1 after steering: r * cos(θ)
        target_h_dot_b1 = r * math.cos(theta_rad)
        
        # α = target_h_dot_b1 - h_dot_b1
        alpha = target_h_dot_b1 - h_dot_b1
        
        # Apply: h' = h + α·d
        # This changes the b1 component while preserving the b2 component
        h_steered = hidden_states + alpha.unsqueeze(-1) * d
        
        # To fully match Standard steering, we also need to adjust the b2 component
        # Target projection onto b2: r * sin(θ)  
        target_h_dot_b2 = r * math.sin(theta_rad)
        beta = target_h_dot_b2 - h_dot_b2
        h_steered = h_steered + beta.unsqueeze(-1) * b2
        
        return h_steered


class AblationSteeringOperator(BaseSteeringOperator):
    """
    Directional ablation (orthogonalization) operator - a special case of Angular Steering.
    
    From baselines.md:
        h_ablate = h_⊥ = h - (h·d)·d
        
    This is equivalent to rotating to θ = 90° (π/2).
    The theta parameter is ignored since ablation always produces the orthogonal component.
    """
    
    def steer(
        self,
        hidden_states: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply directional ablation (orthogonalization).
        
        Removes the component along the feature direction:
            h' = h - (h·d)·d = h_⊥
        
        Note: theta parameter is ignored - ablation always applies 90° rotation.
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        cached = self._get_device_tensors(device, dtype)
        d = cached['b1']  # Feature direction
        
        # Compute h·d (projection onto feature direction)
        h_dot_d = torch.matmul(hidden_states, d)  # Shape: (...,)
        
        # Compute h_⊥ = h - (h·d)·d
        h_parallel = h_dot_d.unsqueeze(-1) * d  # Shape: (..., hidden_dim)
        
        return hidden_states - h_parallel


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
        mode: Steering mode ("standard", "adaptive", "selective", "addition", "ablation")
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
    elif mode_enum == SteeringMode.ADDITION:
        return AdditionSteeringOperator(b1, b2)
    elif mode_enum == SteeringMode.ABLATION:
        return AblationSteeringOperator(b1, b2)
    else:
        return StandardSteeringOperator(b1, b2)

