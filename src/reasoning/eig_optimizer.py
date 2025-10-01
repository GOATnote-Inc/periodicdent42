"""
EIG-Driven Planning: Bayesian experimental design for maximum information gain per cost.

This module implements Expected Information Gain (EIG) calculations and active learning
strategies to prioritize experiments that reduce uncertainty most efficiently.

Moat: TIME - Maximize learning velocity through intelligent experiment selection.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from scipy.stats import entropy, norm
from scipy.optimize import minimize
import structlog

from configs.data_schema import Experiment, Protocol, Sample, Decision

logger = structlog.get_logger()


@dataclass
class EIGResult:
    """Result of EIG calculation."""
    candidate: np.ndarray
    eig: float
    eig_per_cost: float
    cost_hours: float
    cost_usd: float
    rationale: str


class GaussianProcessSurrogate:
    """Simple GP surrogate model for demonstration.
    
    In production, use GPyTorch or BoTorch for more sophisticated models.
    """
    
    def __init__(self, kernel_length_scale: float = 1.0, noise: float = 0.1):
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.length_scale = kernel_length_scale
        self.noise = noise
        self.logger = structlog.get_logger()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.logger.info("gp_fitted", n_train=len(X))
    
    def predict(self, X_test: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict mean and std at test points."""
        if self.X_train is None:
            raise ValueError("Model not fitted")
        
        # RBF kernel: k(x, x') = exp(-||x - x'||^2 / (2 * l^2))
        def rbf_kernel(X1, X2):
            dists = np.sum(X1**2, axis=1, keepdims=True) + \
                    np.sum(X2**2, axis=1, keepdims=True).T - \
                    2 * np.dot(X1, X2.T)
            return np.exp(-dists / (2 * self.length_scale**2))
        
        K = rbf_kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        K_star = rbf_kernel(self.X_train, X_test)
        K_star_star = rbf_kernel(X_test, X_test)
        
        # Posterior mean
        K_inv = np.linalg.inv(K)
        mean = K_star.T @ K_inv @ self.y_train
        
        if return_std:
            # Posterior variance
            var = np.diag(K_star_star - K_star.T @ K_inv @ K_star)
            std = np.sqrt(np.maximum(var, 0))  # Ensure non-negative
            return mean, std
        
        return mean, None
    
    def posterior_variance(self, X_test: np.ndarray) -> np.ndarray:
        """Get posterior variance at test points."""
        _, std = self.predict(X_test, return_std=True)
        return std**2 if std is not None else np.zeros(len(X_test))


class EIGOptimizer:
    """EIG-driven experiment planner.
    
    Moat: TIME - Bayesian optimization for maximum information/cost.
    """
    
    def __init__(self, gp_model: GaussianProcessSurrogate, cost_model: Optional[Dict[str, float]] = None):
        self.gp = gp_model
        self.cost_model = cost_model or {}
        self.logger = structlog.get_logger()
    
    def calculate_eig(self, X_candidate: np.ndarray, n_samples: int = 100) -> float:
        """Calculate Expected Information Gain for candidate.
        
        EIG = H(θ) - E[H(θ|y)]
        where θ = model parameters, y = new observation
        
        Args:
            X_candidate: Candidate experiment parameters (shape: (n_dim,) or (1, n_dim))
            n_samples: Number of samples for Monte Carlo approximation
        
        Returns:
            EIG value (higher = more informative)
        """
        if X_candidate.ndim == 1:
            X_candidate = X_candidate.reshape(1, -1)
        
        # Prior entropy: current uncertainty
        prior_samples = self.gp.predict(X_candidate, return_std=True)[0]
        prior_std = self.gp.predict(X_candidate, return_std=True)[1]
        
        if prior_std is None or prior_std[0] < 1e-6:
            return 0.0  # Already certain, no information gain
        
        H_prior = 0.5 * np.log(2 * np.pi * np.e * prior_std[0]**2)
        
        # Expected posterior entropy (Monte Carlo approximation)
        H_posterior_expected = 0.0
        
        for _ in range(n_samples):
            # Sample hypothetical observation
            y_hypothetical = np.random.normal(prior_samples[0], prior_std[0])
            
            # Simulate GP update (simplified: just reduces variance)
            # In production, use GPyTorch's fantasize() method
            posterior_std = prior_std[0] * np.sqrt(1 - 0.5)  # Assume 50% reduction
            
            H_posterior = 0.5 * np.log(2 * np.pi * np.e * posterior_std**2)
            H_posterior_expected += H_posterior
        
        H_posterior_expected /= n_samples
        
        eig = H_prior - H_posterior_expected
        
        return max(eig, 0.0)  # EIG should be non-negative
    
    def estimate_cost(self, X_candidate: np.ndarray, instrument_id: str) -> Tuple[float, float]:
        """Estimate experiment cost in hours and USD.
        
        Args:
            X_candidate: Experiment parameters
            instrument_id: Target instrument
        
        Returns:
            (cost_hours, cost_usd)
        """
        # Simple cost model (replace with actual estimates)
        base_time = self.cost_model.get(f"{instrument_id}_time", 2.0)
        base_cost = self.cost_model.get(f"{instrument_id}_cost", 100.0)
        
        # Add complexity factor based on parameters
        complexity = 1.0 + 0.1 * np.sum(np.abs(X_candidate))
        
        return base_time * complexity, base_cost * complexity
    
    def calculate_eig_per_cost(self, X_candidate: np.ndarray, instrument_id: str) -> EIGResult:
        """Calculate EIG per unit cost.
        
        This is the key metric: information gained per dollar/hour spent.
        
        Moat: TIME - Optimize learning velocity, not just information.
        """
        eig = self.calculate_eig(X_candidate)
        cost_hours, cost_usd = self.estimate_cost(X_candidate, instrument_id)
        
        # Combined cost metric (normalize to similar scale)
        total_cost = cost_hours + cost_usd / 100.0
        
        eig_per_cost = eig / total_cost if total_cost > 0 else 0.0
        
        rationale = f"EIG={eig:.3f}, cost={cost_hours:.1f}h/${cost_usd:.0f}, ratio={eig_per_cost:.3f}"
        
        return EIGResult(
            candidate=X_candidate,
            eig=eig,
            eig_per_cost=eig_per_cost,
            cost_hours=cost_hours,
            cost_usd=cost_usd,
            rationale=rationale
        )
    
    def select_batch_greedy(
        self,
        X_pool: np.ndarray,
        batch_size: int,
        instrument_id: str
    ) -> List[EIGResult]:
        """Greedily select batch of high-EIG experiments.
        
        Args:
            X_pool: Pool of candidate experiments (n_candidates, n_dim)
            batch_size: Number of experiments to select
            instrument_id: Target instrument
        
        Returns:
            List of selected experiments with EIG scores
        
        Moat: TIME - Batch selection for parallel execution.
        """
        selected = []
        remaining_indices = list(range(len(X_pool)))
        
        for i in range(min(batch_size, len(X_pool))):
            # Calculate EIG for all remaining candidates
            eig_results = []
            
            for idx in remaining_indices:
                result = self.calculate_eig_per_cost(X_pool[idx], instrument_id)
                eig_results.append((idx, result))
            
            # Select highest EIG/cost
            best_idx, best_result = max(eig_results, key=lambda x: x[1].eig_per_cost)
            
            selected.append(best_result)
            remaining_indices.remove(best_idx)
            
            self.logger.info(
                "experiment_selected",
                iteration=i,
                eig=best_result.eig,
                eig_per_cost=best_result.eig_per_cost
            )
            
            # Simulate GP update for next iteration (fantasize)
            # In production, use GPyTorch's condition_on_observations
            # For now, just continue with current GP
        
        return selected
    
    def uncertainty_sampling(self, X_pool: np.ndarray, top_k: int = 5) -> np.ndarray:
        """Select points with highest predictive variance.
        
        Simple baseline strategy for comparison.
        """
        _, std = self.gp.predict(X_pool, return_std=True)
        
        if std is None:
            return X_pool[:top_k]
        
        top_indices = np.argsort(std)[-top_k:]
        return X_pool[top_indices]
    
    def upper_confidence_bound(
        self,
        X_pool: np.ndarray,
        beta: float = 2.0,
        top_k: int = 5
    ) -> np.ndarray:
        """UCB acquisition function.
        
        UCB = mean + beta * std
        """
        mean, std = self.gp.predict(X_pool, return_std=True)
        
        if std is None:
            return X_pool[:top_k]
        
        ucb = mean + beta * std
        top_indices = np.argsort(ucb)[-top_k:]
        
        return X_pool[top_indices]


def generate_decision_log(
    selected: List[EIGResult],
    alternatives: List[EIGResult],
    agent_id: str = "eig_optimizer"
) -> Decision:
    """Generate glass-box decision log.
    
    Moat: INTERPRETABILITY - Every planning decision is explainable.
    """
    # Format rationale
    rationale_parts = [
        f"Selected {len(selected)} experiments to maximize EIG/cost.",
        f"Total expected EIG: {sum(r.eig for r in selected):.2f}",
        f"Total cost: {sum(r.cost_hours for r in selected):.1f} hours, ${sum(r.cost_usd for r in selected):.0f}",
        "Selection prioritizes high-uncertainty regions while minimizing cost."
    ]
    
    rationale = " ".join(rationale_parts)
    
    # Format alternatives
    alternatives_list = [
        {
            "eig": alt.eig,
            "eig_per_cost": alt.eig_per_cost,
            "reason": f"Lower EIG/cost ratio ({alt.eig_per_cost:.3f} vs {selected[0].eig_per_cost:.3f})"
        }
        for alt in alternatives[:5]  # Top 5 alternatives
    ]
    
    return Decision(
        agent_id=agent_id,
        action=f"select_batch_{len(selected)}",
        rationale=rationale,
        confidence=0.85,  # Could be based on GP uncertainty
        alternatives_considered=alternatives_list,
        input_state={"n_pool": len(alternatives) + len(selected)},
        expected_outcome={"total_eig": sum(r.eig for r in selected)}
    )


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    
    X_train = np.random.rand(10, 2)  # 10 experiments, 2 parameters
    y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] + np.random.randn(10) * 0.1
    
    # Fit GP
    gp = GaussianProcessSurrogate()
    gp.fit(X_train, y_train)
    
    # Create optimizer
    cost_model = {
        "xrd-001_time": 2.0,
        "xrd-001_cost": 100.0
    }
    optimizer = EIGOptimizer(gp, cost_model)
    
    # Generate candidate pool
    X_pool = np.random.rand(50, 2)
    
    # Select batch
    selected = optimizer.select_batch_greedy(X_pool, batch_size=5, instrument_id="xrd-001")
    
    print("Selected experiments:")
    for i, result in enumerate(selected):
        print(f"{i+1}. EIG={result.eig:.3f}, cost={result.cost_hours:.1f}h, ratio={result.eig_per_cost:.3f}")
    
    # Generate decision log
    decision = generate_decision_log(selected, [], agent_id="demo")
    print(f"\nDecision rationale: {decision.rationale}")

