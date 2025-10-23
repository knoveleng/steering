"""
Objective functions for steering plane optimization

IMPROVEMENTS:
- Better handling of mixed dtypes (BFloat16 activations + Float32 basis)
- Explicit dtype conversion where needed
- More robust to numerical issues
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


class SeparabilityObjective:
    """
    Measure how well the subspace separates harmful vs harmless activations
    
    Higher is better - we want large distance between projected distributions
    """
    
    def __init__(self, harmful_acts: Dict[str, torch.Tensor], harmless_acts: Dict[str, torch.Tensor]):
        """
        Args:
            harmful_acts: Dict[layer_name, Tensor(n_harmful, d_model)]
            harmless_acts: Dict[layer_name, Tensor(n_harmless, d_model)]
        """
        # Stack all activations across layers for global objective
        self.harmful = torch.cat([v for v in harmful_acts.values()], dim=0)
        self.harmless = torch.cat([v for v in harmless_acts.values()], dim=0)
        
        # Store original dtype for reference
        self.original_dtype = self.harmful.dtype
        
        # Normalize to unit sphere (important for fair comparison)
        self.harmful = F.normalize(self.harmful, dim=-1)
        self.harmless = F.normalize(self.harmless, dim=-1)
    
    def __call__(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute separability score
        
        Args:
            basis: Tensor of shape (d_model, 2) representing orthonormal basis
                   [b1 | b2] where b1, b2 are column vectors
                   May be Float32 (from geoopt) even if activations are BFloat16
        
        Returns:
            Separability score (scalar tensor, higher is better)
        """
        # Convert activations to match basis dtype
        # This is necessary because PyTorch requires exact dtype match for matmul
        harmful = self.harmful.to(basis.dtype)
        harmless = self.harmless.to(basis.dtype)
        
        # Project activations onto the 2D subspace
        proj_harmful = harmful @ basis  # (n_harmful, 2)
        proj_harmless = harmless @ basis  # (n_harmless, 2)
        
        # Compute mean projections in 2D space
        mean_harmful = proj_harmful.mean(dim=0)  # (2,)
        mean_harmless = proj_harmless.mean(dim=0)  # (2,)
        
        # Separability = Euclidean distance between means in 2D projection
        separability = torch.norm(mean_harmful - mean_harmless)
        
        return separability


class PreservationObjective:
    """
    Measure how much the subspace preserves non-target information
    
    Lower is better - we want minimal disruption to original activations
    """
    
    def __init__(
        self, 
        all_acts: Dict[str, torch.Tensor],
        preservation_weight: str = 'magnitude'  # 'magnitude' or 'uniform'
    ):
        """
        Args:
            all_acts: All activations (harmful + harmless combined)
            preservation_weight: How to weight preservation
                'magnitude': Preserve high-magnitude activations more
                'uniform': Equal weight to all activations
        """
        # Combine all activations
        self.activations = torch.cat([v for v in all_acts.values()], dim=0)
        
        # Store original dtype
        self.original_dtype = self.activations.dtype
        
        # Normalize
        self.activations = F.normalize(self.activations, dim=-1)
        
        # Compute weights based on original magnitudes if needed
        if preservation_weight == 'magnitude':
            # Higher magnitude = more important to preserve
            all_acts_unnorm = torch.cat([v for v in all_acts.values()], dim=0)
            self.weights = torch.norm(all_acts_unnorm, dim=-1)
            self.weights = self.weights / (self.weights.sum() + 1e-8)  # Normalize weights
        else:
            self.weights = torch.ones(len(self.activations), 
                                     device=self.activations.device,
                                     dtype=self.activations.dtype) / len(self.activations)
        
        # Ensure weights are on the same device
        self.weights = self.weights.to(self.activations.device)
    
    def __call__(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute preservation cost (how much information is lost)
        
        Args:
            basis: Tensor of shape (d_model, 2)
                   May be Float32 even if activations are BFloat16
        
        Returns:
            Preservation cost (scalar, lower is better)
        """
        # Convert activations to match basis dtype (required by PyTorch)
        activations = self.activations.to(basis.dtype)
        
        # Project activations onto the 2D subspace
        proj = activations @ basis  # (n, 2)
        
        # Reconstruct from projection: h_reconstructed = proj @ basis^T
        reconstructed = proj @ basis.T  # (n, d_model)
        
        # Compute reconstruction error (weighted)
        # This measures how much information is lost by projecting to 2D
        errors = torch.norm(activations - reconstructed, dim=-1)  # (n,)
        
        # Convert weights to match error dtype if needed
        weights = self.weights.to(errors.dtype)
        
        weighted_error = (errors * weights).sum()
        
        return weighted_error


class CombinedObjective:
    """
    Combined objective for Grassmannian optimization
    
    Maximize: J(S) = α * Separability(S) - β * Preservation_Cost(S)
    
    IMPORTANT: This objective is designed to work with mixed dtypes:
    - Activations may be BFloat16 (from model)
    - Basis may be Float32 (from geoopt optimization)
    - We explicitly convert activations to match basis dtype before operations
    
    Note: PyTorch requires exact dtype match for matrix multiplication,
    so we must convert explicitly rather than relying on automatic promotion.
    """
    
    def __init__(
        self,
        harmful_acts: Dict[str, torch.Tensor],
        harmless_acts: Dict[str, torch.Tensor],
        alpha: float = 1.0,
        beta: float = 0.1,
        preservation_weight: str = 'uniform'
    ):
        """
        Args:
            harmful_acts: Harmful activations (may be BFloat16)
            harmless_acts: Harmless activations (may be BFloat16)
            alpha: Weight for separability (higher = prioritize separation)
            beta: Weight for preservation (higher = avoid disrupting other features)
            preservation_weight: How to weight preservation
        """
        self.separability = SeparabilityObjective(harmful_acts, harmless_acts)
        
        # Combine all activations for preservation
        all_acts = {**harmful_acts, **harmless_acts}
        self.preservation = PreservationObjective(all_acts, preservation_weight)
        
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Compute combined objective (we want to MAXIMIZE this)
        
        Args:
            basis: Tensor of shape (d_model, 2)
                   May be Float32 (from geoopt) even if activations are BFloat16
        
        Returns:
            Objective value to MINIMIZE (negative of what we want to maximize)
        """
        # Compute components
        # Note: Both will return Float32 tensors due to matrix mult with Float32 basis
        sep = self.separability(basis)
        pres_cost = self.preservation(basis)
        
        # Return negative because optimizers minimize by default
        # We want to maximize: α*separability - β*preservation_cost
        # So we minimize: -α*separability + β*preservation_cost
        objective = self.alpha * sep - self.beta * pres_cost
        
        return -objective  # Negative for minimization
    
    def get_components(self, basis: torch.Tensor) -> Tuple[float, float]:
        """
        Get individual objective components for logging
        
        Args:
            basis: Current basis (may be Float32)
        
        Returns:
            (separability, preservation_cost) as Python floats
        """
        with torch.no_grad():
            sep = self.separability(basis).item()
            pres = self.preservation(basis).item()
        return sep, pres