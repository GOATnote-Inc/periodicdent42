"""Acquisition functions for active learning.

Provides multiple strategies for selecting which samples to label next:
- Upper Confidence Bound (UCB): Optimistic exploration
- Expected Improvement (EI): Expected gain over current best
- Maximum Variance (MaxVar): Pure uncertainty sampling
- Expected Information Gain (EIG): Information-theoretic selection

All functions are designed to work with uncertainty-aware models from Phase 3.
"""

from typing import Literal, Optional

import numpy as np
from scipy.stats import norm


def upper_confidence_bound(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    kappa: float = 2.0,
    maximize: bool = True,
) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB balances exploitation (predicted value) and exploration (uncertainty).
    
    Formula:
        UCB(x) = μ(x) + κ * σ(x)  (for maximization)
        UCB(x) = μ(x) - κ * σ(x)  (for minimization)
    
    Args:
        y_pred: Predicted values (mean) (N,)
        y_std: Predicted standard deviations (N,)
        kappa: Exploration-exploitation trade-off (default: 2.0)
               Higher κ → more exploration
        maximize: Whether to maximize (True) or minimize (False) the objective
        
    Returns:
        UCB scores (N,): Higher values indicate better candidates
        
    Example:
        >>> y_pred = np.array([10.0, 20.0, 15.0])
        >>> y_std = np.array([1.0, 5.0, 2.0])
        >>> scores = upper_confidence_bound(y_pred, y_std, kappa=2.0)
        >>> best_idx = np.argmax(scores)
    """
    if len(y_pred) != len(y_std):
        raise ValueError("y_pred and y_std must have same length")
    
    if np.any(y_std < 0):
        raise ValueError("Standard deviations must be non-negative")
    
    if maximize:
        ucb = y_pred + kappa * y_std
    else:
        ucb = y_pred - kappa * y_std
    
    return ucb


def expected_improvement(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_best: float,
    xi: float = 0.01,
    maximize: bool = True,
) -> np.ndarray:
    """
    Expected Improvement (EI) acquisition function.
    
    EI measures the expected gain over the current best observation.
    
    Formula:
        EI(x) = E[max(0, f(x) - f_best)]
              = σ(x) * [Z * Φ(Z) + φ(Z)]
        where Z = (μ(x) - f_best - ξ) / σ(x)
    
    Args:
        y_pred: Predicted values (N,)
        y_std: Predicted standard deviations (N,)
        y_best: Best observed value so far
        xi: Exploration-exploitation trade-off (default: 0.01)
            Higher ξ → more exploration
        maximize: Whether to maximize (True) or minimize (False)
        
    Returns:
        EI scores (N,): Higher values indicate better candidates
        
    Reference:
        Močkus (1975) "On Bayesian Methods for Seeking the Extremum"
    """
    if len(y_pred) != len(y_std):
        raise ValueError("y_pred and y_std must have same length")
    
    if np.any(y_std < 0):
        raise ValueError("Standard deviations must be non-negative")
    
    # Avoid division by zero
    y_std_safe = np.maximum(y_std, 1e-8)
    
    if maximize:
        # Improvement = f(x) - f_best
        improvement = y_pred - y_best - xi
    else:
        # Improvement = f_best - f(x)
        improvement = y_best - y_pred - xi
    
    # Z-score
    Z = improvement / y_std_safe
    
    # Expected improvement
    ei = y_std_safe * (Z * norm.cdf(Z) + norm.pdf(Z))
    
    # Zero EI where std is zero
    ei[y_std == 0] = 0.0
    
    return ei


def maximum_variance(y_std: np.ndarray) -> np.ndarray:
    """
    Maximum Variance (MaxVar) acquisition function.
    
    Selects samples with highest predicted uncertainty.
    Pure exploration strategy (ignores predicted values).
    
    Formula:
        MaxVar(x) = σ²(x)
    
    Args:
        y_std: Predicted standard deviations (N,)
        
    Returns:
        Variance scores (N,): Higher values indicate more uncertain samples
    """
    if np.any(y_std < 0):
        raise ValueError("Standard deviations must be non-negative")
    
    return y_std ** 2


def expected_information_gain_proxy(
    y_std: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Expected Information Gain (EIG) proxy using uncertainty.
    
    Approximates EIG using predicted uncertainty. True EIG requires
    Monte Carlo estimation, so we use a fast proxy.
    
    Formula:
        EIG_proxy(x) = σ(x) * [1 + α * |μ(x) - μ_mean|]
    
    Args:
        y_std: Predicted standard deviations (N,)
        y_pred: Predicted values (N,) - optional, for bias weighting
        alpha: Bias term weight (default: 1.0)
               Higher α → favor samples far from mean prediction
        
    Returns:
        EIG proxy scores (N,): Higher values indicate more informative samples
        
    Note:
        This is a heuristic proxy. True EIG requires expensive computation.
    """
    if np.any(y_std < 0):
        raise ValueError("Standard deviations must be non-negative")
    
    eig_proxy = y_std.copy()
    
    if y_pred is not None:
        # Weight by distance from mean prediction
        if len(y_pred) != len(y_std):
            raise ValueError("y_pred and y_std must have same length")
        
        y_mean = y_pred.mean()
        distance = np.abs(y_pred - y_mean)
        
        # Normalize distance to [0, 1]
        distance_range = distance.max() - distance.min()
        if distance_range > 0:
            distance_norm = distance / distance_range
        else:
            distance_norm = np.zeros_like(distance)
        
        eig_proxy = y_std * (1 + alpha * distance_norm)
    
    return eig_proxy


def thompson_sampling(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_samples: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    Thompson Sampling acquisition scores.
    
    Samples from posterior distribution and counts how often each
    candidate would be selected as the best.
    
    Args:
        y_pred: Predicted values (N,)
        y_std: Predicted standard deviations (N,)
        n_samples: Number of Monte Carlo samples (default: 100)
        random_state: Random seed
        
    Returns:
        Selection probabilities (N,): Fraction of times each candidate is best
    """
    if len(y_pred) != len(y_std):
        raise ValueError("y_pred and y_std must have same length")
    
    if np.any(y_std < 0):
        raise ValueError("Standard deviations must be non-negative")
    
    rng = np.random.RandomState(random_state)
    
    # Sample from posterior N(μ, σ²)
    samples = rng.normal(
        loc=y_pred[:, np.newaxis],
        scale=y_std[:, np.newaxis],
        size=(len(y_pred), n_samples)
    )  # Shape: (N, n_samples)
    
    # Count how many times each candidate is best
    best_indices = samples.argmax(axis=0)  # Shape: (n_samples,)
    
    # Compute selection probability for each candidate
    selection_counts = np.bincount(best_indices, minlength=len(y_pred))
    selection_prob = selection_counts / n_samples
    
    return selection_prob


def create_acquisition_function(
    method: Literal["ucb", "ei", "maxvar", "eig_proxy", "thompson"],
    **kwargs,
):
    """
    Factory function to create acquisition functions.
    
    Args:
        method: Acquisition method
        **kwargs: Method-specific arguments
        
    Returns:
        Callable acquisition function
        
    Example:
        >>> acq_fn = create_acquisition_function("ucb", kappa=2.0, maximize=True)
        >>> scores = acq_fn(y_pred=predictions, y_std=uncertainties)
    """
    if method == "ucb":
        def acq_fn(y_pred, y_std, **extra_kwargs):
            merged_kwargs = {**kwargs, **extra_kwargs}
            return upper_confidence_bound(y_pred, y_std, **merged_kwargs)
        return acq_fn
    
    elif method == "ei":
        def acq_fn(y_pred, y_std, **extra_kwargs):
            merged_kwargs = {**kwargs, **extra_kwargs}
            return expected_improvement(y_pred, y_std, **merged_kwargs)
        return acq_fn
    
    elif method == "maxvar":
        def acq_fn(y_std, **extra_kwargs):
            return maximum_variance(y_std)
        return acq_fn
    
    elif method == "eig_proxy":
        def acq_fn(y_std, y_pred=None, **extra_kwargs):
            merged_kwargs = {**kwargs, **extra_kwargs}
            return expected_information_gain_proxy(y_std, y_pred, **merged_kwargs)
        return acq_fn
    
    elif method == "thompson":
        def acq_fn(y_pred, y_std, **extra_kwargs):
            merged_kwargs = {**kwargs, **extra_kwargs}
            return thompson_sampling(y_pred, y_std, **merged_kwargs)
        return acq_fn
    
    else:
        raise ValueError(
            f"Unknown acquisition method: {method}. "
            "Choose from: 'ucb', 'ei', 'maxvar', 'eig_proxy', 'thompson'"
        )

