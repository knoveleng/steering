"""
Grassmannian Plane Constructor with Adam Optimizer

IMPROVEMENTS:
- Uses Adam optimizer (both geoopt and manual)
- Fixed geoopt import (use Stiefel instead of Grassmann)
- Adaptive learning rate with cosine annealing
- Gradient clipping for stability
- Early stopping
"""

import torch
from typing import Dict, Tuple, Optional, List
import numpy as np
from sklearn.decomposition import PCA

from .base import BasePlaneConstructor
from ..optimization import SeparabilityObjective, PreservationObjective, CombinedObjective
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
    
    Adapts the Adam optimizer to Riemannian manifolds by:
    1. Computing Riemannian gradients
    2. Applying Adam momentum and adaptive learning rates
    3. Using Cayley retraction to stay on manifold
    
    Reference: "Riemannian Adaptive Optimization Methods" (ICLR 2019)
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ):
        """
        Initialize Riemannian Adam optimizer
        
        Args:
            lr: Learning rate
            betas: Coefficients for computing running averages (beta1, beta2)
            eps: Term for numerical stability
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        # State
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0     # Timestep
    
    def step(
        self,
        basis: torch.Tensor,
        grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one optimization step
        
        Args:
            basis: Current point on manifold (d_model, k)
            grad: Riemannian gradient (d_model, k)
        
        Returns:
            Updated basis after Adam step
        """
        # Initialize moments if first step
        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)
        
        # Increment timestep
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Compute Adam update direction
        # Note: We use negative gradient for minimization
        update_direction = -m_hat / (torch.sqrt(v_hat) + self.eps)
        
        # Apply Cayley retraction to move on manifold
        basis_new = cayley_retraction(basis, update_direction, self.lr)
        
        return basis_new
    
    def set_lr(self, lr: float):
        """Update learning rate"""
        self.lr = lr


class GrassmannianPlaneConstructor(BasePlaneConstructor):
    """
    Construct optimal 2D steering plane via Grassmannian optimization
    
    Uses Adam optimizer for better convergence compared to SGD.
    """
    
    def __init__(
        self,
        feature_direction: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 0.1,
        lr: float = 0.01,  # Lower for Adam
        betas: Tuple[float, float] = (0.9, 0.999),
        max_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        use_geoopt: bool = True,
        verbose: bool = True,
        use_lr_schedule: bool = True,
        gradient_clip: float = 1.0,
        patience: int = 10
    ):
        """
        Initialize Grassmannian optimizer with Adam
        
        Args:
            feature_direction: Optional pre-selected feature direction
            alpha: Weight for separability objective
            beta: Weight for preservation objective
            lr: Learning rate (0.01 is good for Adam)
            betas: Adam momentum parameters (beta1, beta2)
            max_iterations: Maximum optimization iterations
            convergence_threshold: Stop if distance < threshold
            use_geoopt: Use geoopt library (True) or manual Adam (False)
            verbose: Print optimization progress
            use_lr_schedule: Use cosine annealing learning rate schedule
            gradient_clip: Maximum gradient norm
            patience: Early stopping patience
        """
        self.feature_direction = feature_direction
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.betas = betas
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_geoopt = use_geoopt
        self.verbose = verbose
        self.use_lr_schedule = use_lr_schedule
        self.gradient_clip = gradient_clip
        self.patience = patience
        
        # State
        self.b1 = None
        self.b2 = None
        self.projection_matrix = None
        self.original_dtype = None
        
        # Optimization history
        self.optimization_history = {
            'distances': [],
            'objectives': [],
            'separability': [],
            'preservation': [],
            'learning_rates': []
        }
        
        self.logger = setup_logger(obj=self)
    
    def construct_plane(
        self,
        feature_direction: torch.Tensor,
        candidates: Dict[str, torch.Tensor],
        harmful_activations: Dict[str, torch.Tensor],
        harmless_activations: Dict[str, torch.Tensor],
    ) -> None:
        """Construct optimal plane via Grassmannian optimization with Adam"""
        # Store original dtype
        self.original_dtype = feature_direction.dtype
        self.feature_direction = feature_direction
        
        if self.verbose:
            self.logger.info("="*60)
            self.logger.info("Grassmannian Plane Optimization (Adam)")
            self.logger.info("="*60)
            self.logger.info(f"Optimizer: {'Geoopt RiemannianAdam' if self.use_geoopt else 'Manual Riemannian Adam'}")
            self.logger.info(f"Separability weight (α): {self.alpha}")
            self.logger.info(f"Preservation weight (β): {self.beta}")
            self.logger.info(f"Initial learning rate: {self.lr}")
            self.logger.info(f"Adam betas: {self.betas}")
            self.logger.info(f"LR scheduling: {'Enabled' if self.use_lr_schedule else 'Disabled'}")
            self.logger.info(f"Gradient clipping: {self.gradient_clip}")
            self.logger.info(f"Max iterations: {self.max_iterations}")
        
        # Step 1: Initialize from PCA
        initial_basis = self._initialize_from_pca(feature_direction, candidates)
        
        # Step 2: Create objective function
        objective = CombinedObjective(
            harmful_activations,
            harmless_activations,
            alpha=self.alpha,
            beta=self.beta,
            preservation_weight='uniform'
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
            sep, pres = objective.get_components(optimized_basis)
            self.logger.info("="*60)
            self.logger.info("✓ Optimization Complete")
            self.logger.info(f"Final Separability: {sep:.4f}")
            self.logger.info(f"Final Preservation Cost: {pres:.4f}")
            self.logger.info(f"Combined Objective: {self.alpha * sep - self.beta * pres:.4f}")
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
        """Optimize using geoopt library with RiemannianAdam"""
        if self.verbose:
            self.logger.info("[2/3] Optimizing with geoopt RiemannianAdam...")
        
        try:
            import geoopt
            
            # Store original dtype
            original_dtype = initial_basis.dtype
            
            # CRITICAL: Convert to float32 if BFloat16
            # Geoopt's internal operations (retr_transp) also use torch.linalg.solve
            # which doesn't support BFloat16
            if initial_basis.dtype == torch.bfloat16:
                if self.verbose:
                    self.logger.info("  Converting BFloat16 → Float32 for geoopt compatibility")
                initial_basis = initial_basis.float()
            
            # Use Stiefel manifold (not Grassmann - it doesn't exist in geoopt)
            # Stiefel: X^T X = I (orthonormal columns)
            manifold = geoopt.manifolds.Stiefel()
            
            # Create parameter on manifold
            basis_param = geoopt.ManifoldParameter(
                initial_basis.clone().detach(),
                manifold=manifold
            )
            
            # Use RiemannianAdam optimizer
            optimizer = geoopt.optim.RiemannianAdam(
                [basis_param],
                lr=self.lr,
                betas=self.betas
            )
            
            # Optimization loop
            prev_basis = initial_basis.clone()
            best_loss = float('inf')
            patience_counter = 0
            
            for iteration in range(self.max_iterations):
                # Update learning rate if using schedule
                if self.use_lr_schedule:
                    current_lr = self._get_learning_rate(iteration)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                else:
                    current_lr = self.lr
                
                optimizer.zero_grad()
                
                # Compute objective
                # Note: objective may have BFloat16 activations stored, but basis_param 
                # is Float32. PyTorch handles this automatically in matrix multiplications.
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
                    sep, pres = objective.get_components(basis_param)
                    
                    self.optimization_history['distances'].append(distance)
                    self.optimization_history['objectives'].append(loss.item())
                    self.optimization_history['separability'].append(sep)
                    self.optimization_history['preservation'].append(pres)
                    self.optimization_history['learning_rates'].append(current_lr)
                    
                    if self.verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                        self.logger.info(
                            f"  Iter {iteration:3d}: Loss={loss.item():.4f}, "
                            f"Sep={sep:.4f}, Pres={pres:.4f}, Dist={distance:.6f}, LR={current_lr:.5f}"
                        )
                    
                    # Early stopping
                    if loss.item() < best_loss - 1e-6:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.patience:
                        if self.verbose:
                            self.logger.info(f"  ✓ Early stopping at iteration {iteration}")
                        break
                    
                    # Convergence check
                    if distance < self.convergence_threshold:
                        if self.verbose:
                            self.logger.info(f"  ✓ Converged at iteration {iteration}")
                        break
                    
                    prev_basis = basis_param.data.clone()
            
            # Convert back to original dtype if needed
            result = basis_param.data.detach()
            if original_dtype == torch.bfloat16:
                if self.verbose:
                    self.logger.info("  Converting Float32 → BFloat16")
                result = result.bfloat16()
            
            return result
            
        except ImportError as e:
            self.logger.error(f"Failed to import geoopt: {e}")
            self.logger.info("Falling back to manual Adam optimization...")
            return self._optimize_manual_adam(initial_basis, objective)
    
    def _optimize_manual_adam(
        self,
        initial_basis: torch.Tensor,
        objective: CombinedObjective
    ) -> torch.Tensor:
        """Manual optimization using custom Riemannian Adam"""
        if self.verbose:
            self.logger.info("[2/3] Optimizing with manual Riemannian Adam...")
        
        basis = initial_basis.clone().detach().requires_grad_(True)
        prev_basis = basis.clone()
        
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
                sep, pres = objective.get_components(basis_new)
                
                self.optimization_history['distances'].append(distance)
                self.optimization_history['objectives'].append(loss.item())
                self.optimization_history['separability'].append(sep)
                self.optimization_history['preservation'].append(pres)
                self.optimization_history['learning_rates'].append(current_lr)
                
                if self.verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                    self.logger.info(
                        f"  Iter {iteration:3d}: Loss={loss.item():.4f}, "
                        f"Sep={sep:.4f}, Pres={pres:.4f}, Dist={distance:.6f}, LR={current_lr:.5f}"
                    )
                
                # Early stopping
                if loss.item() < best_loss - 1e-6:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        self.logger.info(f"  ✓ Early stopping at iteration {iteration}")
                    break
                
                # Convergence check
                if distance < self.convergence_threshold:
                    if self.verbose:
                        self.logger.info(f"  ✓ Converged at iteration {iteration}")
                    break
                
                # Update for next iteration
                prev_basis = basis.clone()
                basis = basis_new.detach().requires_grad_(True)
        
        return basis.detach()
    
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
    
    def get_optimization_history(self) -> Dict[str, List[float]]:
        """Get optimization trajectory for analysis"""
        return self.optimization_history
    
    def project_onto_plane(self, vector: torch.Tensor) -> torch.Tensor:
        """Project vector onto steering plane"""
        P = self.get_projection_matrix()
        P = P.to(vector.device, dtype=vector.dtype)
        return torch.matmul(vector, P.T)
    
    def decompose_in_basis(self, vector: torch.Tensor) -> Tuple[float, float]:
        """Decompose vector in terms of basis {b1, b2}"""
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
        """Project all candidate directions onto the steering plane"""
        projections = {}
        for layer_name, direction in candidates.items():
            coeff_b1, coeff_b2 = self.decompose_in_basis(direction)
            projections[layer_name] = (coeff_b1, coeff_b2)
        return projections
    
    def measure_contraction_constant(self) -> Optional[float]:
        """Estimate empirical contraction constant from optimization history"""
        distances = self.optimization_history['distances']
        
        if len(distances) < 3:
            return None
        
        ratios = []
        for i in range(1, len(distances)):
            if distances[i-1] > 1e-8:
                ratio = distances[i] / distances[i-1]
                ratios.append(ratio)
        
        if not ratios:
            return None
        
        return max(ratios)