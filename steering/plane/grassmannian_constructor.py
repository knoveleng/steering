"""
Grassmannian Plane Constructor

Constructs optimal 2D steering plane via optimization on Grassmannian manifold.

FIXED VERSION:
- Correct objective sign (maximizes separability)
- Uses FocusObjective instead of PreservationObjective
- No normalization of activations
- Passes feature_direction to objective
"""

import torch
from typing import Dict, Tuple, Optional, List
import numpy as np
from sklearn.decomposition import PCA

from .base import BasePlaneConstructor
from ..optimization import SeparabilityObjective, FocusObjective, CombinedObjective
from ..optimization.manifold_utils import (
    project_to_grassmannian, 
    compute_grassmann_distance,
    riemannian_gradient,
    cayley_retraction
)
from ..utils.logger import setup_logger


class RiemannianAdam:
    """
    Manual implementation of Riemannian Adam optimizer
    
    Adapts Adam to Riemannian manifolds by:
    1. Computing Riemannian gradients
    2. Applying Adam momentum and adaptive learning rates
    3. Using retraction for manifold updates
    """
    
    def __init__(self, lr: float = 0.01, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        # State
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def step(self, basis: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Perform one optimization step
        
        Args:
            basis: Current basis point on manifold
            grad: Riemannian gradient at current point
            
        Returns:
            Updated basis after optimization step
        """
        self.t += 1
        
        # Initialize moments on first step
        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Compute Adam update direction
        update = m_hat / (torch.sqrt(v_hat) + self.eps)
        
        # Move in the direction (note: gradient descent, so negative)
        new_basis = basis - self.lr * update
        
        # Project back to manifold
        new_basis = project_to_grassmannian(new_basis)
        
        return new_basis
    
    def set_lr(self, lr: float):
        """Update learning rate"""
        self.lr = lr


class GrassmannianPlaneConstructor(BasePlaneConstructor):
    """
    Construct optimal 2D steering plane via Grassmannian optimization
    
    Optimizes: max_{S ∈ G(2,d)} α·Separability(S) + β·Focus(S)
    
    where:
    - Separability = distance between harmful/harmless cluster centers in 2D projection
    - Focus = alignment of subspace with known feature direction
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        lr: float = 0.1,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        use_geoopt: bool = True,
        use_lr_schedule: bool = True,
        gradient_clip: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        early_stopping_patience: int = 15,
        normalize_activations: bool = False,
        scaling_method: str = 'baseline',  # NEW: separability scaling method
        verbose: bool = True
    ):
        """
        Args:
            alpha: Separability weight (higher = prioritize separation)
            beta: Focus weight (higher = prioritize feature alignment)
            lr: Initial learning rate
            max_iterations: Maximum optimization iterations
            convergence_threshold: Stop if distance between iterates < this
            use_geoopt: Use geoopt library (if False, use manual implementation)
            use_lr_schedule: Use cosine annealing learning rate schedule
            gradient_clip: Clip gradients to this norm (0 = no clipping)
            betas: Adam optimizer betas
            early_stopping_patience: Stop if no improvement for this many iterations
            normalize_activations: Whether to normalize activations (default: False)
            scaling_method: How to scale separability ('baseline', 'magnitude', 'none')
            verbose: Print optimization progress
        """
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_geoopt = use_geoopt
        self.use_lr_schedule = use_lr_schedule
        self.gradient_clip = gradient_clip
        self.betas = betas
        self.early_stopping_patience = early_stopping_patience
        self.normalize_activations = normalize_activations
        self.scaling_method = scaling_method  # NEW: Store scaling method
        self.verbose = verbose
        
        self.logger = setup_logger(__name__)
        
        # Optimization history
        self.optimization_history = {
            'objectives': [],
            'separability': [],
            'focus': [],
            'distances': [],
            'learning_rates': []
        }
        
        # Results
        self.b1 = None
        self.b2 = None
        self.projection_matrix = None
        self.feature_direction = None
        self.original_dtype = None
    
    def construct_plane(
        self,
        feature_direction: torch.Tensor,
        candidates: Dict[str, torch.Tensor],
        harmful_activations: Dict[str, torch.Tensor],
        harmless_activations: Dict[str, torch.Tensor],
    ) -> None:
        """Construct optimal plane via Grassmannian optimization"""
        # Store original dtype
        self.original_dtype = feature_direction.dtype
        self.feature_direction = feature_direction
        
        if self.verbose:
            self.logger.info("="*60)
            self.logger.info("Grassmannian Plane Optimization")
            self.logger.info("="*60)
            self.logger.info(f"Optimizer: {'Geoopt RiemannianAdam' if self.use_geoopt else 'Manual Riemannian Adam'}")
            self.logger.info(f"Separability weight (α): {self.alpha}")
            self.logger.info(f"Focus weight (β): {self.beta}")
            self.logger.info(f"Initial learning rate: {self.lr}")
            self.logger.info(f"Adam betas: {self.betas}")
            self.logger.info(f"Normalize activations: {self.normalize_activations}")
            self.logger.info(f"Max iterations: {self.max_iterations}")
        
        # Step 1: Initialize from PCA
        initial_basis = self._initialize_from_pca(feature_direction, candidates)
        
        # Step 2: Create objective function (FIXED: now passes feature_direction)
        objective = CombinedObjective(
            harmful_activations,
            harmless_activations,
            feature_direction,  # CRITICAL: Pass feature direction for FocusObjective
            alpha=self.alpha,
            beta=self.beta,
            normalize_activations=self.normalize_activations,
            verbose=self.verbose
        )
        
        # Step 3: Optimize with Adam
        if self.use_geoopt:
            optimized_basis = self._optimize_with_geoopt(initial_basis, objective)
        else:
            optimized_basis = self._optimize_manual_adam(initial_basis, objective)
        
        # Step 4: Extract basis
        self.b1 = optimized_basis[:, 0]
        self.b2 = optimized_basis[:, 1]
        
        # Construct projection matrix
        self.projection_matrix = (
            torch.outer(self.b1, self.b1) + 
            torch.outer(self.b2, self.b2)
        )
        
        if self.verbose:
            sep, foc = objective.get_components(optimized_basis)
            self.logger.info("="*60)
            self.logger.info("✓ Optimization Complete")
            self.logger.info(f"Final Separability: {sep:.4f}")
            self.logger.info(f"Final Focus: {foc:.4f}")
            self.logger.info(f"Combined Objective: {self.alpha * sep + self.beta * foc:.4f}")
            self.logger.info("="*60)
    
    def _initialize_from_pca(
        self,
        feature_direction: torch.Tensor,
        candidates: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Initialize plane from (feature_direction, PCA_component)"""
        if self.verbose:
            self.logger.info("[1/3] Initializing from PCA...")
        
        candidate_matrix = torch.stack(list(candidates.values()))
        candidate_matrix_np = candidate_matrix.float().cpu().numpy()
        
        pca = PCA(n_components=1)
        pca.fit(candidate_matrix_np)
        
        d_pc0 = torch.from_numpy(pca.components_[0]).to(
            feature_direction.device,
            dtype=self.original_dtype
        )
        
        # Gram-Schmidt orthonormalization
        b1 = feature_direction / (feature_direction.norm() + 1e-8)
        b2 = d_pc0 - torch.dot(d_pc0, b1) * b1
        b2 = b2 / (b2.norm() + 1e-8)
        
        initial_basis = torch.stack([b1, b2], dim=1)
        
        if self.verbose:
            self.logger.info(f"  ✓ Initialized basis shape: {initial_basis.shape}")
        
        return initial_basis
    
    def _get_learning_rate(self, iteration: int) -> float:
        """Compute learning rate with cosine annealing"""
        if not self.use_lr_schedule:
            return self.lr
        
        lr_min = self.lr / 10
        lr_max = self.lr
        progress = iteration / self.max_iterations
        lr = lr_min + (lr_max - lr_min) * (1 + np.cos(np.pi * progress)) / 2
        
        return lr
    
    def _optimize_with_geoopt(
        self,
        initial_basis: torch.Tensor,
        objective: CombinedObjective
    ) -> torch.Tensor:
        """Optimize using geoopt library"""
        if self.verbose:
            self.logger.info("[2/3] Optimizing with Geoopt...")
        
        try:
            import geoopt
        except ImportError:
            self.logger.warning("Geoopt not installed, falling back to manual implementation")
            return self._optimize_manual_adam(initial_basis, objective)

        # Convert float dtype
        original_dtype = initial_basis.dtype
        initial_basis = initial_basis.float()
        
        # Create parameter on Stiefel manifold
        manifold = geoopt.manifolds.Stiefel()
        basis_param = geoopt.ManifoldParameter(initial_basis.clone(), manifold=manifold)
        
        # Create Riemannian Adam optimizer
        optimizer = geoopt.optim.RiemannianAdam([basis_param], lr=self.lr, betas=self.betas)
        
        best_loss = float('inf')
        patience_counter = 0
        prev_basis = basis_param.data.clone()
        
        for iteration in range(self.max_iterations):
            # Get current learning rate
            current_lr = self._get_learning_rate(iteration)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - geoopt handles dtype conversion automatically
            loss = objective(basis_param)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_([basis_param], self.gradient_clip)
            
            # Optimization step
            optimizer.step()
            
            # Measure convergence
            with torch.no_grad():
                distance = compute_grassmann_distance(prev_basis, basis_param.data)
                
                # Log metrics
                sep, foc = objective.get_components(basis_param)
                
                self.optimization_history['distances'].append(distance)
                self.optimization_history['objectives'].append(-loss.item())  # Negate to show actual objective
                self.optimization_history['separability'].append(sep)
                self.optimization_history['focus'].append(foc)
                self.optimization_history['learning_rates'].append(current_lr)
                
                if self.verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                    self.logger.info(
                        f"  Iter {iteration:3d}: Obj={-loss.item():.4f}, "
                        f"Sep={sep:.4f}, Foc={foc:.4f}, Dist={distance:.6f}, LR={current_lr:.5f}"
                    )
                
                # Early stopping
                if loss.item() < best_loss - 1e-6:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        self.logger.info(f"  ✓ Early stopping at iteration {iteration}")
                    break
                
                # Convergence check
                if distance < self.convergence_threshold:
                    if self.verbose:
                        self.logger.info(f"  ✓ Converged at iteration {iteration}")
                    break
                
                prev_basis = basis_param.data.clone()

        return basis_param.data.detach().to(dtype=original_dtype)
    
    def _optimize_manual_adam(
        self,
        initial_basis: torch.Tensor,
        objective: CombinedObjective
    ) -> torch.Tensor:
        """Optimize using manual Riemannian Adam"""
        if self.verbose:
            self.logger.info("[2/3] Optimizing with Manual Adam...")
        
        basis = initial_basis.clone().requires_grad_(True)
        prev_basis = basis.detach().clone()
        
        # Create Riemannian Adam optimizer
        optimizer = RiemannianAdam(
            lr=self.lr,
            betas=self.betas
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(self.max_iterations):
            # Get current learning rate
            current_lr = self._get_learning_rate(iteration)
            optimizer.set_lr(current_lr)
            
            # Forward pass
            loss = objective(basis)
            
            # Compute Euclidean gradient
            loss.backward()
            euclidean_grad = basis.grad.clone()
            
            # Optimization step
            with torch.no_grad():
                # Project to Riemannian gradient
                riemann_grad = riemannian_gradient(basis, euclidean_grad)
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    grad_norm = torch.norm(riemann_grad)
                    if grad_norm > self.gradient_clip:
                        riemann_grad = riemann_grad * (self.gradient_clip / grad_norm)
                
                # Adam step (handles negative gradient internally)
                basis_new = optimizer.step(basis, riemann_grad)
                
                # Measure convergence
                distance = compute_grassmann_distance(prev_basis, basis_new)
                
                # Log metrics
                sep, foc = objective.get_components(basis_new)
                
                self.optimization_history['distances'].append(distance)
                self.optimization_history['objectives'].append(-loss.item())
                self.optimization_history['separability'].append(sep)
                self.optimization_history['focus'].append(foc)
                self.optimization_history['learning_rates'].append(current_lr)
                
                if self.verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                    self.logger.info(
                        f"  Iter {iteration:3d}: Obj={-loss.item():.4f}, "
                        f"Sep={sep:.4f}, Foc={foc:.4f}, Dist={distance:.6f}"
                    )
                
                # Update basis
                basis.copy_(basis_new)
                basis.grad.zero_()
                prev_basis = basis_new.clone()
                
                # Early stopping
                if loss.item() < best_loss - 1e-6:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        self.logger.info(f"  ✓ Early stopping at iteration {iteration}")
                    break
                
                # Convergence check
                if distance < self.convergence_threshold:
                    if self.verbose:
                        self.logger.info(f"  ✓ Converged at iteration {iteration}")
                    break
        
        return basis.detach()
    
    def get_basis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the constructed basis vectors"""
        if self.b1 is None or self.b2 is None:
            raise ValueError("Plane not constructed yet. Call construct_plane() first.")
        return self.b1, self.b2
    
    def get_projection_matrix(self) -> torch.Tensor:
        """Get the projection matrix onto the plane"""
        if self.projection_matrix is None:
            raise ValueError("Plane not constructed yet. Call construct_plane() first.")
        return self.projection_matrix
    
    def get_optimization_history(self) -> Dict[str, List[float]]:
        """Get optimization trajectory for analysis"""
        return self.optimization_history
    
    def measure_contraction_constant(self) -> Optional[float]:
        """
        Estimate empirical contraction constant from optimization history.
        
        The contraction constant q measures how much distances shrink between
        consecutive iterations: d(t+1) / d(t).
        
        If q < 1, the optimization has the contraction property and convergence
        is guaranteed by the Banach fixed-point theorem.
        
        Returns:
            Maximum contraction ratio observed, or None if insufficient data
        """
        distances = self.optimization_history['distances']
        
        if len(distances) < 3:
            return None
        
        # Compute ratios of consecutive distances
        ratios = []
        for i in range(1, len(distances)):
            if distances[i-1] > 1e-8:  # Avoid division by zero
                ratio = distances[i] / distances[i-1]
                ratios.append(ratio)
        
        if not ratios:
            return None
        
        # Return maximum ratio (worst-case contraction constant)
        return max(ratios)
    
    def project_onto_plane(self, vector: torch.Tensor) -> torch.Tensor:
        """Project a vector onto the steering plane"""
        P = self.get_projection_matrix()
        P = P.to(vector.device, dtype=vector.dtype)
        return torch.matmul(vector, P.T)
    
    def decompose_in_basis(self, vector: torch.Tensor) -> Tuple[float, float]:
        """
        Decompose a vector in terms of basis {b1, b2}
        
        Returns coefficients (c1, c2) such that proj(vector) ≈ c1*b1 + c2*b2
        """
        b1, b2 = self.get_basis()
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
        
        Returns dictionary mapping layer names to (b1_coeff, b2_coeff) tuples
        """
        projections = {}
        for layer_name, direction in candidates.items():
            coeff_b1, coeff_b2 = self.decompose_in_basis(direction)
            projections[layer_name] = (coeff_b1, coeff_b2)
        return projections