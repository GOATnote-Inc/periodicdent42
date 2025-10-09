"""Active learning loop with budget management and GO/NO-GO gates.

Orchestrates the active learning process:
1. Train model on labeled data
2. Compute acquisition scores on unlabeled pool
3. Select diverse batch
4. Apply budget constraints and GO/NO-GO gates
5. Query labels and update dataset
"""

from typing import Callable, Optional

import numpy as np


class ActiveLearningLoop:
    """
    Active learning loop with budget management.
    
    Manages the iterative process of:
    - Training models
    - Selecting samples to label
    - Updating datasets
    - Tracking budget
    """
    
    def __init__(
        self,
        base_model,
        acquisition_fn: Callable,
        diversity_selector: Optional[Callable] = None,
        budget: int = 100,
        batch_size: int = 10,
        stopping_criterion: Optional[Callable] = None,
        random_state: int = 42,
    ):
        """
        Initialize active learning loop.
        
        Args:
            base_model: Uncertainty-aware model (from Phase 3)
            acquisition_fn: Acquisition function (from acquisition.py)
            diversity_selector: Diversity selector (from diversity.py) - optional
            budget: Maximum number of labels to acquire
            batch_size: Number of samples per iteration
            stopping_criterion: Optional function(metrics) -> bool to stop early
            random_state: Random seed
        """
        self.base_model = base_model
        self.acquisition_fn = acquisition_fn
        self.diversity_selector = diversity_selector
        self.budget = budget
        self.batch_size = batch_size
        self.stopping_criterion = stopping_criterion
        self.random_state = random_state
        
        self.history_ = []
        self.budget_used_ = 0
    
    def run(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        y_unlabeled: Optional[np.ndarray] = None,  # For simulation only
        n_iterations: Optional[int] = None,
    ) -> dict:
        """
        Run active learning loop.
        
        Args:
            X_labeled: Initial labeled features (N_labeled, D)
            y_labeled: Initial labeled targets (N_labeled,)
            X_unlabeled: Unlabeled pool features (N_unlabeled, D)
            y_unlabeled: True labels for unlabeled pool (for simulation only)
            n_iterations: Maximum iterations (default: budget // batch_size)
            
        Returns:
            Dictionary with final state and history
        """
        if n_iterations is None:
            n_iterations = self.budget // self.batch_size
        
        # Initialize
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_pool = X_unlabeled.copy()
        
        if y_unlabeled is not None:
            y_pool = y_unlabeled.copy()
        else:
            y_pool = None
        
        # Active learning iterations
        for iteration in range(n_iterations):
            # Check budget
            if self.budget_used_ >= self.budget:
                print(f"Budget exhausted: {self.budget_used_}/{self.budget}")
                break
            
            if len(X_pool) == 0:
                print("Unlabeled pool exhausted")
                break
            
            # 1. Train model
            self.base_model.fit(X_train, y_train)
            
            # 2. Compute acquisition scores
            if hasattr(self.base_model, "predict_with_uncertainty"):
                y_pred, _, _ = self.base_model.predict_with_uncertainty(X_pool)
                y_std = self.base_model.get_epistemic_uncertainty(X_pool)
            else:
                y_pred = self.base_model.predict(X_pool)
                y_std = np.ones(len(X_pool))  # Fallback
            
            # Compute acquisition scores
            # Handle different acquisition function signatures
            try:
                # Try with y_pred and y_std
                acq_scores = self.acquisition_fn(y_pred=y_pred, y_std=y_std)
            except TypeError:
                # Try with just y_std (e.g., maxvar)
                acq_scores = self.acquisition_fn(y_std=y_std)
            
            # 3. Select batch
            actual_batch_size = min(self.batch_size, len(X_pool), self.budget - self.budget_used_)
            
            if actual_batch_size <= 0:
                break
            
            if self.diversity_selector is not None:
                # Diversity-aware selection
                selected_indices = self.diversity_selector(
                    X_candidates=X_pool,
                    acquisition_scores=acq_scores,
                    batch_size=actual_batch_size,
                )
            else:
                # Pure acquisition: select top-k
                selected_indices = np.argsort(acq_scores)[-actual_batch_size:][::-1]
            
            # 4. Query labels (simulate if y_pool available)
            X_query = X_pool[selected_indices]
            
            if y_pool is not None:
                y_query = y_pool[selected_indices]
            else:
                # In real scenario, would query oracle/lab
                raise ValueError("y_unlabeled must be provided for simulation")
            
            # 5. Update datasets
            X_train = np.vstack([X_train, X_query])
            y_train = np.concatenate([y_train, y_query])
            
            # Remove queried samples from pool
            mask = np.ones(len(X_pool), dtype=bool)
            mask[selected_indices] = False
            X_pool = X_pool[mask]
            if y_pool is not None:
                y_pool = y_pool[mask]
            
            self.budget_used_ += actual_batch_size
            
            # 6. Record history
            self.history_.append({
                "iteration": iteration,
                "n_labeled": len(X_train),
                "n_pool": len(X_pool),
                "budget_used": self.budget_used_,
                "selected_indices": selected_indices,
                "acquisition_scores": acq_scores[selected_indices],
            })
            
            # 7. Check stopping criterion
            if self.stopping_criterion is not None:
                metrics = {
                    "iteration": iteration,
                    "n_labeled": len(X_train),
                    "budget_used": self.budget_used_,
                }
                if self.stopping_criterion(metrics):
                    print(f"Stopping criterion met at iteration {iteration}")
                    break
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_pool": X_pool,
            "y_pool": y_pool,
            "history": self.history_,
            "budget_used": self.budget_used_,
        }
    
    def get_history(self) -> list[dict]:
        """Get history of active learning iterations."""
        return self.history_
    
    def reset(self):
        """Reset history and budget."""
        self.history_ = []
        self.budget_used_ = 0


def go_no_go_gate(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """
    GO/NO-GO gate for autonomous lab deployment.
    
    Decision rules:
        - GO: Prediction interval entirely above threshold_min
        - NO-GO: Prediction interval entirely below threshold_max
        - MAYBE: Interval overlaps thresholds → query for more information
    
    Args:
        y_pred: Predicted values (N,)
        y_std: Predicted uncertainties (N,)
        y_lower: Lower bounds of prediction intervals (N,)
        y_upper: Upper bounds of prediction intervals (N,)
        threshold_min: Minimum acceptable value (e.g., T_c > 77K for LN2)
        threshold_max: Maximum acceptable value (e.g., T_c < 200K for cryogenics)
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        Decisions (N,): 1=GO, 0=MAYBE (query), -1=NO-GO
        
    Example:
        >>> # Superconductor screening: looking for T_c > 77K
        >>> decisions = go_no_go_gate(
        ...     y_pred, y_std, y_lower, y_upper,
        ...     threshold_min=77.0, threshold_max=np.inf
        ... )
        >>> go_samples = (decisions == 1)
        >>> query_samples = (decisions == 0)
        >>> no_go_samples = (decisions == -1)
    """
    decisions = np.zeros(len(y_pred), dtype=int)
    
    # GO: Lower bound > threshold_min AND upper bound < threshold_max
    go_mask = (y_lower > threshold_min) & (y_upper < threshold_max)
    decisions[go_mask] = 1
    
    # NO-GO: Upper bound < threshold_min OR lower bound > threshold_max
    no_go_mask = (y_upper < threshold_min) | (y_lower > threshold_max)
    decisions[no_go_mask] = -1
    
    # MAYBE: Everything else (overlaps thresholds)
    # Already initialized to 0
    
    return decisions

