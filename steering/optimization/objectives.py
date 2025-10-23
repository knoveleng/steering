"""
Objective functions with theoretical bounds normalization

Uses exact theoretical bounds for separability:
- Min: 0 (no separation)
- Max: ||mean_harmful - mean_harmless||² (maximum possible separation)
- Adaptive scaling based on dimensionality for automatic headroom

This provides:
1. Exact, known bounds (no sampling needed)
2. Works for any model/dataset
3. Automatic headroom for optimization
4. Fast initialization
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class SeparabilityObjective:
    """
    Separability with theoretical min-max normalization to [0,1]
    
    Uses exact theoretical bounds:
    - Min = 0 (worst case: no separation)
    - Max = ||Δmean||² / expected_capture_rate (with adaptive headroom)
    
    Where expected_capture_rate accounts for the fact that a 2D subspace
    can only capture a fraction of the total variance/separation.
    """
    
    def __init__(
        self, 
        harmful_acts: Dict[str, torch.Tensor], 
        harmless_acts: Dict[str, torch.Tensor],
        normalize: bool = False,
        adaptive_scaling: bool = True
    ):
        """
        Args:
            harmful_acts: Dict[layer_name, Tensor(n_harmful, d_model)]
            harmless_acts: Dict[layer_name, Tensor(n_harmless, d_model)]
            normalize: Whether to normalize activations (default: False)
            adaptive_scaling: Whether to use adaptive headroom (default: True)
        """
        # Stack all activations
        self.harmful = torch.cat([v for v in harmful_acts.values()], dim=0)
        self.harmless = torch.cat([v for v in harmless_acts.values()], dim=0)
        self.original_dtype = self.harmful.dtype
        
        if normalize:
            self.harmful = F.normalize(self.harmful, dim=-1)
            self.harmless = F.normalize(self.harmless, dim=-1)
        
        # Compute theoretical bounds
        self._compute_theoretical_bounds(adaptive_scaling)
    
    def _compute_theoretical_bounds(self, adaptive_scaling: bool = True):
        """
        Compute theoretical min/max separability bounds.
        
        Min: Always 0 (worst case: harmful and harmless project to same point)
        Max: ||mean_harmful - mean_harmless||² (maximum possible separation)
        
        With adaptive_scaling, we add headroom based on how much separation
        a 2D subspace can realistically capture from high-dimensional space.
        """
        # Compute class means
        mean_harmful = self.harmful.mean(dim=0)  # (d_model,)
        mean_harmless = self.harmless.mean(dim=0)  # (d_model,)
        
        # Maximum possible separability = squared norm of difference
        max_possible = torch.sum((mean_harmful - mean_harmless) ** 2).item()
        
        # Minimum is always 0
        self.min_sep = 0.0
        
        if adaptive_scaling:
            # Add adaptive headroom based on dimensionality
            d_model = self.harmful.shape[1]
            
            # Expected capture rate: how much of the total separation
            # can a 2D subspace capture? This depends on:
            # 1. Intrinsic dimensionality of the data
            # 2. How concentrated the separation is in few dimensions
            
            # Conservative estimate: 2D captures sqrt(2/d) to 2/sqrt(d) of variance
            # For separation, use similar scaling
            min_capture = np.sqrt(2.0 / d_model)  # Lower bound
            max_capture = 0.95  # Upper bound (can't capture everything)
            
            # Use geometric mean for balanced estimate
            expected_capture = np.sqrt(min_capture * max_capture)
            expected_capture = np.clip(expected_capture, 0.3, 0.95)
            
            # Scale max_sep to provide headroom
            # If we expect to capture 60%, set max at 60% so initial value isn't at ceiling
            self.max_sep = max_possible / expected_capture
            
            print(f"Separability scaling (theoretical + adaptive):")
            print(f"  Max possible separation: {max_possible:.4f}")
            print(f"  Expected 2D capture rate: {100*expected_capture:.1f}%")
            print(f"  Scaled upper bound: {self.max_sep:.4f}")
            print(f"  Headroom: {100*(1-expected_capture):.1f}% above expected")
        else:
            # No adaptive scaling: use raw theoretical max
            self.max_sep = max_possible
            print(f"Separability scaling (theoretical):")
            print(f"  Range: [0, {self.max_sep:.4f}]")
        
        # Safety check
        if self.max_sep < 1e-8:
            print("Warning: Classes are nearly identical (max_sep ≈ 0)")
            print("  Setting max_sep = 1.0 to avoid division by zero")
            self.max_sep = 1.0
    
    def _compute_raw_separability(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute raw separability (squared distance between projected means)
        
        Returns:
            Scalar tensor with raw separability value
        """
        harmful = self.harmful.to(basis.dtype)
        harmless = self.harmless.to(basis.dtype)
        
        # Project onto 2D subspace
        proj_harmful = harmful @ basis  # (n_harmful, 2)
        proj_harmless = harmless @ basis  # (n_harmless, 2)
        
        # Compute mean projections
        mean_harmful = proj_harmful.mean(dim=0)  # (2,)
        mean_harmless = proj_harmless.mean(dim=0)  # (2,)
        
        # Squared Euclidean distance
        separability = torch.sum((mean_harmful - mean_harmless) ** 2)
        
        return separability
    
    def __call__(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized separability in [0,1].
        
        Args:
            basis: Orthonormal basis of shape (d_model, 2)
        
        Returns:
            Separability score in [0, 1]:
            - 0.0 = No separation (means project to same point)
            - 1.0 = Maximum expected separation (with headroom)
        """
        raw_sep = self._compute_raw_separability(basis)
        
        # Normalize to [0, 1]
        normalized = (raw_sep - self.min_sep) / (self.max_sep - self.min_sep + 1e-8)
        
        # Clamp to [0, 1] (should rarely hit 1.0 due to headroom)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        return normalized
    
    def get_raw_value(self, basis: torch.Tensor) -> float:
        """Get raw (unnormalized) separability for debugging"""
        return self._compute_raw_separability(basis).item()


class FocusObjective:
    """
    Focus objective: measures alignment with feature direction.
    
    Naturally in [0,1] range:
    - 0.0 = Orthogonal to feature direction
    - 1.0 = Perfectly aligned with feature direction
    """
    
    def __init__(self, feature_direction: torch.Tensor):
        """
        Args:
            feature_direction: The known steering direction vector
                             Will be normalized internally
        """
        self.feature_dir = feature_direction / (torch.norm(feature_direction) + 1e-8)
        self.original_dtype = self.feature_dir.dtype
    
    def __call__(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute focus score in [0,1].
        
        This is ||proj_S(d_feat)||² where S is the subspace spanned by basis.
        
        Args:
            basis: Orthonormal basis of shape (d_model, 2)
        
        Returns:
            Focus score in [0, 1]:
            - 0.0 = feature_direction orthogonal to subspace
            - 1.0 = feature_direction lies entirely in subspace
        """
        feature_dir = self.feature_dir.to(basis.dtype)
        
        # Project feature direction onto subspace
        proj_coeffs = feature_dir @ basis  # (2,)
        
        # Squared norm of projection
        focus = torch.sum(proj_coeffs ** 2)
        
        return focus


class CombinedObjective:
    """
    Combined objective with theoretical bounds normalization.
    
    Maximize: J(S) = α * Sep(S) + β * Focus(S)
    
    Where both Sep(S) and Focus(S) are in [0,1] with known, exact bounds:
    - Sep: [0, ||Δmean||²/expected_capture] (theoretical + adaptive)
    - Focus: [0, 1] (natural bounds)
    
    This makes α and β directly interpretable:
    - α=β → Equal importance
    - α=2β → Separability 2× more important
    - β=2α → Focus 2× more important
    
    Recommended starting point: α=1.0, β=2.0
    """
    
    def __init__(
        self,
        harmful_acts: Dict[str, torch.Tensor],
        harmless_acts: Dict[str, torch.Tensor],
        feature_direction: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 2.0,
        normalize_activations: bool = False,
        adaptive_scaling: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            harmful_acts: Harmful activations
            harmless_acts: Harmless activations
            feature_direction: Known steering direction
            alpha: Weight for separability (default: 1.0)
            beta: Weight for focus (default: 2.0)
            normalize_activations: Whether to normalize activations
            adaptive_scaling: Whether to use adaptive headroom
            verbose: Whether to print scaling information
        """
        self.separability = SeparabilityObjective(
            harmful_acts, 
            harmless_acts,
            normalize=normalize_activations,
            adaptive_scaling=adaptive_scaling
        )
        self.focus = FocusObjective(feature_direction)
        
        self.alpha = alpha
        self.beta = beta
        
        if verbose:
            print(f"\nObjective configuration:")
            print(f"  Weights: α={alpha:.2f} (separability), β={beta:.2f} (focus)")
            
            if alpha == beta:
                print(f"  → Equal weight to both objectives")
            elif alpha > beta:
                print(f"  → Separability is {alpha/beta:.1f}× more important")
            else:
                print(f"  → Focus is {beta/alpha:.1f}× more important")
            
            # Show expected contributions
            # Assuming initial sep≈0.7, focus≈1.0
            initial_sep_contrib = alpha * 0.7
            initial_focus_contrib = beta * 1.0
            total = initial_sep_contrib + initial_focus_contrib
            
            print(f"  Expected initial contributions:")
            print(f"    Separability: {initial_sep_contrib:.2f} ({100*initial_sep_contrib/total:.0f}%)")
            print(f"    Focus: {initial_focus_contrib:.2f} ({100*initial_focus_contrib/total:.0f}%)")
    
    def __call__(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute combined objective (for minimization).
        
        Both components are in [0,1] with exact bounds, so weights are 
        directly comparable across different models and datasets.
        
        Args:
            basis: Orthonormal basis of shape (d_model, 2)
        
        Returns:
            Objective value to MINIMIZE (negative of what we want to maximize)
        """
        sep = self.separability(basis)
        foc = self.focus(basis)
        
        # We want to MAXIMIZE: α*sep + β*focus
        # So we MINIMIZE: -(α*sep + β*focus)
        objective = self.alpha * sep + self.beta * foc
        
        return -objective
    
    def get_components(self, basis: torch.Tensor) -> Tuple[float, float]:
        """
        Get individual objective components (both in [0,1])
        
        Args:
            basis: Current basis
        
        Returns:
            (separability, focus) as Python floats, both in [0,1]
        """
        with torch.no_grad():
            sep = self.separability(basis).item()
            foc = self.focus(basis).item()
        return sep, foc
    
    def get_contributions(self, basis: torch.Tensor) -> Tuple[float, float]:
        """
        Get weighted contributions of each objective component.
        
        Useful for understanding which objective dominates.
        
        Returns:
            (α*separability, β*focus) as Python floats
        """
        sep, foc = self.get_components(basis)
        return self.alpha * sep, self.beta * foc
    
    def get_raw_separability(self, basis: torch.Tensor) -> float:
        """
        Get raw (unnormalized) separability for debugging.
        
        Returns:
            Raw separability value (not normalized to [0,1])
        """
        with torch.no_grad():
            return self.separability.get_raw_value(basis)