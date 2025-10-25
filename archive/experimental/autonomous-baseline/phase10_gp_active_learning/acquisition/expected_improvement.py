"""
Expected Improvement (EI) acquisition function for Bayesian Optimization.

EI balances exploitation (high mean) and exploration (high uncertainty).
"""

import numpy as np
import torch
from scipy import stats
from botorch.acquisition import ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler


def expected_improvement(
    X_candidates: np.ndarray,
    gp_model,
    y_best: float,
    xi: float = 0.01
) -> np.ndarray:
    """
    Compute Expected Improvement for candidate points.
    
    EI(x) = E[max(f(x) - f_best, 0)]
          = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)
    
    where Z = (μ(x) - f_best - ξ) / σ(x)
    
    Args:
        X_candidates: Candidate feature matrix (N, D)
        gp_model: Fitted GP model with predict_with_uncertainty method
        y_best: Best observed target value so far
        xi: Exploration-exploitation trade-off (default: 0.01)
    
    Returns:
        Expected improvement values (N,)
    """
    # Get predictions and uncertainty
    mu, sigma = gp_model.predict_with_uncertainty(X_candidates, return_std=True)
    
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Z
    Z = (mu - y_best - xi) / sigma
    
    # Expected improvement
    ei = (mu - y_best - xi) * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
    
    return ei


def select_batch_ei(
    X_candidates: np.ndarray,
    gp_model,
    y_best: float,
    batch_size: int = 1,
    xi: float = 0.01
) -> np.ndarray:
    """
    Select batch of candidates with highest Expected Improvement.
    
    Args:
        X_candidates: Candidate feature matrix (N, D)
        gp_model: Fitted GP model
        y_best: Best observed target value
        batch_size: Number of candidates to select
        xi: Exploration-exploitation trade-off
    
    Returns:
        Indices of selected candidates (batch_size,)
    """
    # Compute EI for all candidates
    ei_values = expected_improvement(X_candidates, gp_model, y_best, xi=xi)
    
    # Select top batch_size
    selected_indices = np.argsort(ei_values)[-batch_size:][::-1]
    
    return selected_indices


def expected_improvement_acquisition(
    model,
    X_candidate: torch.Tensor,
    best_f: float,
    num_samples: int = 256
) -> torch.Tensor:
    """
    BoTorch-based Expected Improvement for GP/DKL models.
    
    Uses analytic EI (no MC sampling needed for GP).
    
    Args:
        model: Fitted GP or DKL model
        X_candidate: Candidate points (N, D) tensor
        best_f: Best observed function value
        num_samples: Unused (for API compatibility)
    
    Returns:
        EI values for each candidate (N,) tensor
    """
    # Use analytic EI (no sampler needed for GP)
    ei = ExpectedImprovement(model=model, best_f=best_f)
    
    # Ensure X_candidate has batch dimension for BoTorch
    if X_candidate.dim() == 2:
        X_candidate = X_candidate.unsqueeze(1)  # (N, 1, D)
    
    return ei(X_candidate).squeeze()


if __name__ == '__main__':
    # Test EI
    from ..models.gp_model import GPModel
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.rand(50, 10)
    y_train = X_train[:, 0] * 10 + X_train[:, 1] * 5 + np.random.randn(50) * 0.5
    
    X_candidates = np.random.rand(100, 10)
    
    # Fit GP
    gp = GPModel()
    gp.fit(X_train, y_train)
    
    # Compute EI
    y_best = y_train.max()
    ei_values = expected_improvement(X_candidates, gp, y_best)
    
    # Select batch
    selected = select_batch_ei(X_candidates, gp, y_best, batch_size=5)
    
    print(f"Best observed: {y_best:.2f}")
    print(f"EI range: [{ei_values.min():.4f}, {ei_values.max():.4f}]")
    print(f"Selected indices: {selected}")
    print(f"Selected EI values: {ei_values[selected]}")

