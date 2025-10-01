"""
Hybrid Bayesian Optimization + RL Meta-Strategy.

BO handles local search (data-efficient).
RL decides when to explore new regions (meta-strategy).
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HybridOptimizer:
    """
    Hybrid: BO (local) + RL (global strategy).
    
    BO: Sample-efficient local search
    RL: Learns when to explore new regions
    """
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        exploration_threshold: float = 0.5,
        exploitation_budget: int = 10,
    ):
        self.bounds = bounds
        self.n_params = len(bounds)
        self.exploration_threshold = exploration_threshold
        self.exploitation_budget = exploitation_budget
        
        # Bayesian optimization components
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.1**2,
            n_restarts_optimizer=10,
            normalize_y=True,
        )
        
        # History
        self.X_observed = []
        self.y_observed = []
        self.best_value = -np.inf
        self.best_params = None
        
        # Meta-strategy state
        self.steps_in_region = 0
        self.current_region_center = None
        
        logger.info(f"Initialized HybridOptimizer: {self.n_params}D")
    
    def propose_next(self) -> np.ndarray:
        """
        Propose next experiment.
        
        Decision:
        - If little data or stalled: EXPLORE (RL jumps to new region)
        - If improving: EXPLOIT (BO refines locally)
        """
        if len(self.X_observed) < 5:
            # Bootstrap: random exploration
            return self._random_sample()
        
        # Check if we should explore new region
        if self._should_explore():
            logger.info("Meta-strategy: EXPLORE new region")
            self.steps_in_region = 0
            return self._explore_new_region()
        else:
            logger.info(f"Meta-strategy: EXPLOIT locally ({self.steps_in_region}/{self.exploitation_budget})")
            self.steps_in_region += 1
            return self._exploit_local()
    
    def update(self, params: np.ndarray, value: float):
        """Update with new observation."""
        self.X_observed.append(params)
        self.y_observed.append(value)
        
        if value > self.best_value:
            self.best_value = value
            self.best_params = params
        
        # Update GP
        if len(self.X_observed) >= 2:
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
    
    def _should_explore(self) -> bool:
        """
        Decide: explore new region or exploit current?
        
        Explore if:
        - Spent exploitation budget
        - No improvement in last N steps
        - High uncertainty everywhere
        """
        # Budget exhausted
        if self.steps_in_region >= self.exploitation_budget:
            return True
        
        # Check improvement rate
        if len(self.y_observed) >= 5:
            recent = self.y_observed[-5:]
            improvement = max(recent) - min(recent)
            if improvement < 0.01:  # Stalled
                return True
        
        return False
    
    def _explore_new_region(self) -> np.ndarray:
        """
        RL meta-strategy: Jump to high-uncertainty region.
        
        Uses GP uncertainty to find unexplored areas.
        """
        # Sample candidates
        n_candidates = 100
        candidates = np.random.rand(n_candidates, self.n_params)
        
        # Scale to bounds
        scaled = self._scale_to_bounds(candidates)
        
        # Predict uncertainty
        if len(self.X_observed) >= 2:
            _, std = self.gp.predict(scaled, return_std=True)
            # Pick highest uncertainty (most unexplored)
            idx = np.argmax(std)
        else:
            idx = np.random.randint(n_candidates)
        
        self.current_region_center = scaled[idx]
        return scaled[idx]
    
    def _exploit_local(self) -> np.ndarray:
        """
        Bayesian optimization: UCB acquisition.
        
        Refines around current best region.
        """
        if len(self.X_observed) < 2:
            return self._random_sample()
        
        # UCB acquisition function
        def acquisition(x, kappa=2.0):
            x = x.reshape(1, -1)
            mean, std = self.gp.predict(x, return_std=True)
            return -(mean[0] + kappa * std[0])  # Negative for minimization
        
        # Multi-start optimization
        best_acq = np.inf
        best_x = None
        
        for _ in range(10):
            x0 = np.random.rand(self.n_params)
            res = minimize(
                acquisition,
                x0,
                bounds=[(0, 1)] * self.n_params,
                method='L-BFGS-B'
            )
            
            if res.fun < best_acq:
                best_acq = res.fun
                best_x = res.x
        
        return self._scale_to_bounds(best_x.reshape(1, -1))[0]
    
    def _random_sample(self) -> np.ndarray:
        """Random sample from bounds."""
        x = np.random.rand(self.n_params)
        return self._scale_to_bounds(x.reshape(1, -1))[0]
    
    def _scale_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Scale [0,1] to actual bounds."""
        scaled = []
        for i, (name, (low, high)) in enumerate(self.bounds.items()):
            if x.ndim == 1:
                scaled.append(low + x[i] * (high - low))
            else:
                scaled.append(low + x[:, i] * (high - low))
        
        if x.ndim == 1:
            return np.array(scaled)
        else:
            return np.column_stack(scaled)

