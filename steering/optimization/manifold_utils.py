"""
Manifold optimization utilities for Grassmannian
"""

import torch
import torch.nn.functional as F


def project_to_grassmannian(matrix: torch.Tensor) -> torch.Tensor:
    """
    Project a matrix to the Grassmann manifold via QR decomposition
    
    This ensures the columns form an orthonormal basis
    
    Args:
        matrix: Tensor of shape (d_model, k) where k is subspace dimension
    
    Returns:
        Orthonormalized matrix of same shape
    """
    # Store original dtype
    original_dtype = matrix.dtype
    
    # Convert to float32 if BFloat16 (QR decomposition works better with float32)
    if matrix.dtype == torch.bfloat16:
        matrix = matrix.float()
    
    # QR decomposition ensures orthonormality
    Q, R = torch.linalg.qr(matrix)
    
    # Handle sign ambiguity: ensure diagonal of R is positive
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1  # Handle zero diagonal elements
    Q = Q * signs.unsqueeze(0)
    
    # Convert back to original dtype
    if original_dtype == torch.bfloat16:
        Q = Q.bfloat16()
    
    return Q


def cayley_retraction(X: torch.Tensor, G: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Cayley retraction for moving on Stiefel/Grassmann manifold
    
    This is a smooth retraction map that preserves orthonormality
    Complexity: O(d_model * k^2) which is efficient for k=2
    
    Args:
        X: Current point on manifold, shape (d_model, k)
        G: Riemannian gradient, shape (d_model, k)
        lr: Learning rate (step size)
    
    Returns:
        Updated point on manifold
    """
    # Store original dtype for conversion back
    original_dtype = X.dtype
    
    # Convert to float32 if BFloat16 (linalg operations don't support BFloat16)
    if X.dtype == torch.bfloat16:
        X = X.float()
        G = G.float()
    
    k = X.shape[1]
    
    # Construct skew-symmetric matrix: A = G @ X^T - X @ G^T
    A = G @ X.T - X @ G.T
    
    # Cayley transform: Y = (I - lr/2 * A)^{-1} @ (I + lr/2 * A) @ X
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    
    # For efficiency, use the formula: Y = X + (I - lr/2 * A)^{-1} @ (lr * A @ X)
    # This avoids computing full matrix inverse
    
    scaled_A = (lr / 2) * A
    
    # Compute: (I - lr/2 * A)^{-1} @ (lr * A @ X)
    # = (I - lr/2 * A)^{-1} @ (2 * scaled_A @ X)
    rhs = 2 * scaled_A @ X
    
    # Solve: (I - lr/2 * A) @ update = rhs
    # Using torch.linalg.solve for numerical stability
    try:
        update = torch.linalg.solve(I - scaled_A, rhs)
        Y = X + update
    except RuntimeError:
        # Fallback to pseudoinverse if singular
        update = torch.linalg.lstsq(I - scaled_A, rhs).solution
        Y = X + update
    
    # Project back to Grassmannian to ensure orthonormality
    # (Important due to numerical errors)
    Y = project_to_grassmannian(Y)
    
    # Convert back to original dtype
    if original_dtype == torch.bfloat16:
        Y = Y.bfloat16()
    
    return Y


def compute_grassmann_distance(X1: torch.Tensor, X2: torch.Tensor) -> float:
    """
    Compute geodesic distance between two points on Grassmann manifold
    
    This uses principal angles between subspaces
    
    Args:
        X1: First basis, shape (d_model, k)
        X2: Second basis, shape (d_model, k)
    
    Returns:
        Geodesic distance (scalar)
    """
    # Store original dtype
    original_dtype = X1.dtype
    
    # Convert to float32 if BFloat16 for numerical stability
    if X1.dtype == torch.bfloat16:
        X1 = X1.float()
        X2 = X2.float()
    
    # Compute overlap: X1^T @ X2
    overlap = X1.T @ X2
    
    # Singular values of overlap matrix give cosines of principal angles
    # We want: sqrt(sum(arccos(sigma_i)^2))
    singular_values = torch.linalg.svdvals(overlap)
    
    # Clamp to [-1, 1] for numerical stability
    singular_values = torch.clamp(singular_values, -1, 1)
    
    # Principal angles
    angles = torch.acos(singular_values)
    
    # Grassmann distance
    distance = torch.sqrt((angles ** 2).sum())
    
    return distance.item()


def riemannian_gradient(
    X: torch.Tensor,
    euclidean_grad: torch.Tensor
) -> torch.Tensor:
    """
    Project Euclidean gradient to Riemannian gradient on Grassmann manifold
    
    The tangent space at X is: T_X = {X*Omega + X_perp*M : Omega skew-symmetric}
    Projection: grad_R = grad_E - X @ (X^T @ grad_E)
    
    Args:
        X: Current point on manifold, shape (d_model, k)
        euclidean_grad: Standard gradient, shape (d_model, k)
    
    Returns:
        Riemannian gradient, shape (d_model, k)
    """
    # Project gradient to tangent space
    # Riemannian gradient = euclidean_grad - X @ (X^T @ euclidean_grad)
    
    XTG = X.T @ euclidean_grad  # (k, k)
    
    # For Grassmann manifold, we also need to skew-symmetrize
    # grad_R = euclidean_grad @ (I - X @ X^T) + X @ skew(X^T @ euclidean_grad)
    
    skew = (XTG - XTG.T) / 2  # Skew-symmetric part
    sym = (XTG + XTG.T) / 2   # Symmetric part (to be removed)
    
    grad_R = euclidean_grad - X @ sym
    
    return grad_R