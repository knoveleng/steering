"""
Steering operator implementations
"""

import torch
import math
from typing import Optional, Dict, List

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
        Precompute rotation vector for angle with specific dtype (OLD METHOD for backward compatibility)

        Args:
            theta_degrees: Rotation angle in degrees
            dtype: Target dtype
            device: Target device

        Returns:
            Rotated unit vector in steering plane (d-dimensional vector)
        """
        cache_key = (theta_degrees, dtype, device)

        if self.cache_rotations and cache_key in self.rotation_cache:
            return self.rotation_cache[cache_key]

        # Convert to radians
        theta = math.radians(theta_degrees)

        # Get basis vectors with correct dtype and device
        b1 = self.b1.to(device=device, dtype=dtype)
        b2 = self.b2.to(device=device, dtype=dtype)

        # OLD METHOD: Compute [b1, b2] * R_theta * [1, 0]^T
        # This simplifies to: cos(theta)*b1 + sin(theta)*b2
        v_theta = math.cos(theta) * b1 + math.sin(theta) * b2

        if self.cache_rotations:
            self.rotation_cache[cache_key] = v_theta

        return v_theta
    
    def steer_activation(
        self,
        activation: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Apply angular steering (OLD METHOD for backward compatibility)

        Implements: h' = h - proj_P(h) + r * v_theta
        where v_theta = cos(theta)*b1 + sin(theta)*b2

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
        Apply adaptive steering with masking (OLD METHOD for backward compatibility)

        Implements masked version: h' = h + mask * (r * v_theta - proj_P(h))
        where v_theta = cos(theta)*b1 + sin(theta)*b2

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


class SelectiveSteeringOperator(AngularSteeringOperator):
    """
    Selective angular steering that only steers on layers where
    positive and negative samples have opposite-signed mean projections
    along the chosen direction.

    This operator selects layers based on the criterion that the mean scalar
    projection of positive activations and the mean scalar projection of
    negative activations along the feature direction have opposite signs.
    """

    def __init__(
        self,
        b1: torch.Tensor,
        b2: torch.Tensor,
        layer_steering_mask: Dict[str, bool],
        cache_rotations: bool = True
    ):
        """
        Initialize selective operator

        Args:
            b1: First basis vector (feature direction)
            b2: Second basis vector (orthogonal)
            layer_steering_mask: Dict mapping layer names to bool (True = steer this layer)
            cache_rotations: Whether to cache rotations
        """
        super().__init__(b1, b2, cache_rotations)
        self.layer_steering_mask = layer_steering_mask
        self.current_layer = None

    @classmethod
    def from_activations(
        cls,
        positive_activations: Dict[str, torch.Tensor],
        negative_activations: Dict[str, torch.Tensor],
        feature_direction: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        cache_rotations: bool = True,
        method: str = 'opposite_signs',
        best_layer_idx: Optional[int] = None,
        **method_kwargs
    ) -> 'SelectiveSteeringOperator':
        """
        Create a SelectiveSteeringOperator by analyzing activations.

        Args:
            positive_activations: Dict[layer_name, Tensor(n_pos_samples, hidden_dim)]
            negative_activations: Dict[layer_name, Tensor(n_neg_samples, hidden_dim)]
            feature_direction: Feature direction vector to project onto
            b1: First basis vector (feature direction)
            b2: Second basis vector (orthogonal)
            cache_rotations: Whether to cache rotations
            method: Selection method - one of:
                - 'scatter': Scatter selection - all layers with opposite signs (non-contiguous)
                - 'weighted_quality': Range selection - weighted quality maximization (contiguous)
                - 'constrained_window': Range selection - constrained window optimization (contiguous)
                - 'change_point': Range selection - change-point detection (contiguous)
                - 'robust_plateau': Range selection - robust plateau detection (contiguous)
            best_layer_idx: Index of best layer (required for range methods)
            **method_kwargs: Additional arguments for specific methods

        Returns:
            SelectiveSteeringOperator instance with computed layer mask
        """
        # Scatter method: select all layers with opposite-signed projections
        # This may result in non-contiguous selection
        layer_steering_mask = cls.compute_layer_steering_mask(
            positive_activations,
            negative_activations,
            feature_direction
        )

        return cls(b1, b2, layer_steering_mask, cache_rotations)

    @staticmethod
    def compute_layer_steering_mask(
        positive_activations: Dict[str, torch.Tensor],
        negative_activations: Dict[str, torch.Tensor],
        feature_direction: torch.Tensor,
        require_all_at_index: bool = True
    ) -> Dict[str, bool]:
        """
        Compute which layers should be steered based on projection sign analysis.

        A layer is selected for steering if:
        - mean(positive_projections) and mean(negative_projections) have opposite signs
        - (optional) ALL layer components at the same index are True

        Args:
            positive_activations: Dict[layer_name, Tensor(n_pos_samples, hidden_dim)]
            negative_activations: Dict[layer_name, Tensor(n_neg_samples, hidden_dim)]
            feature_direction: Feature direction vector to project onto
            require_all_at_index: If True, only keep index if ALL its components are True

        Returns:
            Dict mapping layer names to bool (True = steer this layer)
        """
        layer_steering_mask = {}

        # Normalize feature direction
        feature_direction_norm = feature_direction / (feature_direction.norm() + 1e-8)
        feature_direction_norm = feature_direction_norm.float()

        for layer_name in positive_activations.keys():
            if layer_name not in negative_activations:
                layer_steering_mask[layer_name] = False
                continue

            # Get activations for this layer
            pos_acts = positive_activations[layer_name].float() / (positive_activations[layer_name].norm(dim=-1, keepdim=True) + 1e-8).float()
            neg_acts = negative_activations[layer_name].float() / (negative_activations[layer_name].norm(dim=-1, keepdim=True) + 1e-8).float()

            # Move feature direction to same device
            feat_dir = feature_direction_norm.to(pos_acts.device)

            # Compute projections onto feature direction for all samples
            pos_projections = pos_acts @ feat_dir  # Shape: (n_pos_samples,)
            neg_projections = neg_acts @ feat_dir  # Shape: (n_neg_samples,)

            # Compute mean projections
            pos_mean_proj = pos_projections.mean().item()
            neg_mean_proj = neg_projections.mean().item()

            # Select layer if mean projections have opposite signs
            opposite_signs = (pos_mean_proj * neg_mean_proj) < 0

            layer_steering_mask[layer_name] = opposite_signs

        # Apply index-based filtering if requested
        # if require_all_at_index:
        #     layer_steering_mask = SelectiveSteeringOperator._filter_by_index(layer_steering_mask)

        return layer_steering_mask

    @staticmethod
    def _filter_by_index(layer_steering_mask: Dict[str, bool]) -> Dict[str, bool]:
        """
        Keep layer index True only if ALL entries at that index are True.
        
        Extracts numeric index from layer names and groups by it.
        An index is valid only if all its entries are True.
        """
        from collections import defaultdict
        import re
        
        # Group by extracted index
        index_groups = defaultdict(list)
        
        for layer_name, is_true in layer_steering_mask.items():
            # Extract first number from layer name (the index)
            match = re.search(r'\.(\d+)\.', layer_name)
            if match:
                index = match.group(1)  # e.g., "11", "16", "18"
                index_groups[index].append((layer_name, is_true))
        
        # Find valid indices where ALL entries are True
        valid_indices = set()
        for index, entries in index_groups.items():
            if all(is_true for _, is_true in entries):
                valid_indices.add(index)
        
        # Build filtered mask
        filtered_mask = {}
        for layer_name in layer_steering_mask.keys():
            match = re.search(r'\.(\d+)\.', layer_name)
            if match:
                index = match.group(1)
                filtered_mask[layer_name] = index in valid_indices
            else:
                filtered_mask[layer_name] = False
        
        return filtered_mask
    

    @staticmethod
    def compute_layer_projection_stats(
        positive_activations: Dict[str, torch.Tensor],
        negative_activations: Dict[str, torch.Tensor],
        feature_direction: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute projection statistics for each layer.

        Returns detailed statistics about how positive and negative activations
        project onto the feature direction at each layer.

        Args:
            positive_activations: Dict[layer_name, Tensor(n_pos_samples, hidden_dim)]
            negative_activations: Dict[layer_name, Tensor(n_neg_samples, hidden_dim)]
            feature_direction: Feature direction vector to project onto

        Returns:
            Dict mapping layer names to statistics dict with keys:
                - 'pos_mean': Mean positive projection
                - 'neg_mean': Mean negative projection
                - 'pos_std': Std of positive projections
                - 'neg_std': Std of negative projections
                - 'opposite_signs': Whether means have opposite signs
                - 'separation': Absolute difference between means
        """
        stats = {}

        # Normalize feature direction
        feature_direction_norm = feature_direction / (feature_direction.norm() + 1e-8)
        feature_direction_norm = feature_direction_norm.float()

        for layer_name in positive_activations.keys():
            if layer_name not in negative_activations:
                continue

            # Get activations for this layer
            pos_acts = positive_activations[layer_name].float()
            neg_acts = negative_activations[layer_name].float()


            # Move feature direction to same device
            feat_dir = feature_direction_norm.to(pos_acts.device)

            # Compute projections
            pos_projections = torch.matmul(pos_acts, feat_dir)
            neg_projections = torch.matmul(neg_acts, feat_dir)

            # Compute statistics
            pos_mean = pos_projections.mean().item()
            neg_mean = neg_projections.mean().item()
            pos_std = pos_projections.std().item()
            neg_std = neg_projections.std().item()

            opposite_signs = (pos_mean * neg_mean) < 0
            separation = abs(pos_mean - neg_mean)

            stats[layer_name] = {
                'pos_mean': pos_mean,
                'neg_mean': neg_mean,
                'pos_std': pos_std,
                'neg_std': neg_std,
                'opposite_signs': opposite_signs,
                'separation': separation
            }

        return stats


    def set_current_layer(self, layer_name: str) -> None:
        """
        Set the current layer being processed

        Args:
            layer_name: Name of the current layer
        """
        self.current_layer = layer_name

    def should_steer_layer(self, layer_name: Optional[str] = None) -> bool:
        """
        Check if the given layer should be steered

        Args:
            layer_name: Layer name to check (uses current_layer if None)

        Returns:
            True if this layer should be steered
        """
        layer = layer_name if layer_name is not None else self.current_layer
        if layer is None:
            return True  # Default to steering if no layer specified
        return self.layer_steering_mask.get(layer, False)

    def get_selected_layers(self) -> List[str]:
        """
        Get list of layers that are selected for steering

        Returns:
            List of layer names where steering is enabled
        """
        return [layer for layer, should_steer in self.layer_steering_mask.items() if should_steer]

    def get_num_selected_layers(self) -> int:
        """
        Get number of layers selected for steering

        Returns:
            Number of layers where steering is enabled
        """
        return sum(self.layer_steering_mask.values())

    def _precompute_rotation_matrix(self, theta_degrees: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Precompute rotation matrix V_θ = [b1 b2] R_θ [b1 b2]^T (CORRECT METHOD for selective mode)

        Args:
            theta_degrees: Rotation angle in degrees
            dtype: Target dtype
            device: Target device

        Returns:
            Rotation matrix V_θ of shape (hidden_dim, hidden_dim)
        """
        cache_key = (theta_degrees, dtype, device, 'matrix')

        if self.cache_rotations and cache_key in self.rotation_cache:
            return self.rotation_cache[cache_key]

        # Convert to radians
        theta = math.radians(theta_degrees)

        # Get basis vectors with correct dtype and device
        b1 = self.b1.to(device=device, dtype=dtype)
        b2 = self.b2.to(device=device, dtype=dtype)

        # Compute R_θ = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Compute V_θ = [b1 b2] R_θ [b1 b2]^T
        # = cos(θ)*(b1⊗b1^T + b2⊗b2^T) + sin(θ)*(b2⊗b1^T - b1⊗b2^T)
        v_theta = (cos_theta * (torch.outer(b1, b1) + torch.outer(b2, b2)) +
                   sin_theta * (torch.outer(b2, b1) - torch.outer(b1, b2)))

        if self.cache_rotations:
            self.rotation_cache[cache_key] = v_theta

        return v_theta

    def steer_activation(
        self,
        activation: torch.Tensor,
        theta: float,
        layer_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Apply selective steering based on layer mask (CORRECT METHOD)

        Implements: h' = h - P*h + V_θ * h (only on selected layers)
        where V_θ = [b1 b2] R_θ [b1 b2]^T

        Args:
            activation: Tensor of shape (..., hidden_dim)
            theta: Target angle in degrees
            layer_name: Optional layer name (uses current_layer if None)

        Returns:
            Steered activation if layer should be steered, otherwise original activation
        """
        # Check if this layer should be steered
        should_steer = self.should_steer_layer(layer_name)

        if not should_steer:
            return activation

        # Store original dtype and device
        original_dtype = activation.dtype
        original_device = activation.device

        # Get projection matrix and V_θ with correct dtype and device
        P = self.P.to(device=original_device, dtype=original_dtype)
        V_theta = self._precompute_rotation_matrix(theta, original_dtype, original_device)

        # Apply CORRECT rotation: h' = h - P*h + V_θ * h
        h_steered = activation - torch.matmul(activation, P.T) + torch.matmul(activation, V_theta.T)

        # Ensure output has same dtype as input
        h_steered = h_steered.to(dtype=original_dtype)

        return h_steered

    def update_layer_mask(self, layer_steering_mask: Dict[str, bool]) -> None:
        """
        Update the layer steering mask

        Args:
            layer_steering_mask: New dict mapping layer names to bool
        """
        self.layer_steering_mask = layer_steering_mask