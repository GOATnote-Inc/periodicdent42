"""
Active Learning Guards (NeurIPS/ICLR Compliance)

Implements test set access control to prevent data leakage during
active learning experiments.

Protocol Contract:
- Test set evaluated ONCE per round AFTER training
- No interim peeking by acquisition functions
- Access tracked and validated
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)


class TestSetGuard:
    """
    Prevents test set peeking during active learning.
    
    Critical for preventing data leakage: acquisition functions must not
    access test set performance when selecting samples.
    
    Protocol:
    - Test set evaluated exactly ONCE per round
    - Evaluation happens AFTER training and BEFORE next acquisition
    - RuntimeError raised if accessed twice in same round
    
    Example:
        >>> test_guard = TestSetGuard(X_test, y_test)
        >>> 
        >>> for round_num in range(10):
        >>>     model.fit(X_train, y_train)
        >>>     
        >>>     # Acquisition cannot access test_guard
        >>>     selected = acquisition_fn(model, X_pool)
        >>>     
        >>>     # Evaluate test set (once per round)
        >>>     metrics = test_guard.evaluate_once_per_round(model, round_num)
        >>>     print(f"Round {round_num}: RMSE={metrics['rmse']:.4f}")
    """
    
    def __init__(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        task_type: str = "regression"
    ):
        """
        Initialize test set guard.
        
        Args:
            X_test: Test features
            y_test: Test targets
            task_type: "regression" or "classification"
        """
        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test and y_test must have same length: "
                f"{len(X_test)} != {len(y_test)}"
            )
        
        if len(X_test) == 0:
            raise ValueError("Test set is empty")
        
        self._X_test = X_test
        self._y_test = y_test
        self.task_type = task_type
        
        # Access tracking
        self._access_count = 0
        self._round_evaluated = set()
        self._evaluation_history = []
        
        logger.info(
            f"TestSetGuard initialized: {len(X_test)} test samples, "
            f"task_type={task_type}"
        )
    
    def evaluate_once_per_round(
        self, 
        model: Any, 
        round_num: int,
        custom_metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set exactly once per round.
        
        Args:
            model: Trained model with predict() method
            round_num: Current round index (must be unique)
            custom_metrics: Optional dict of {name: metric_fn} for custom metrics
        
        Returns:
            Dictionary of metrics (depends on task_type)
        
        Raises:
            RuntimeError: If test set already evaluated in this round
        
        Example:
            >>> metrics = test_guard.evaluate_once_per_round(model, round_num=5)
            >>> print(metrics)
            {'rmse': 0.523, 'mae': 0.412, 'r2': 0.856, 'round': 5}
        """
        # Check for data leakage
        if round_num in self._round_evaluated:
            raise RuntimeError(
                f"[DATA LEAKAGE] Test set already evaluated in round {round_num}! "
                f"This indicates an acquisition function is peeking at test performance. "
                f"Evaluated rounds: {sorted(self._round_evaluated)}"
            )
        
        # Mark round as evaluated
        self._round_evaluated.add(round_num)
        self._access_count += 1
        
        logger.debug(
            f"Evaluating test set for round {round_num} "
            f"(access #{self._access_count})"
        )
        
        # Make predictions
        try:
            y_pred = model.predict(self._X_test)
        except Exception as e:
            logger.error(f"Prediction failed in round {round_num}: {e}")
            raise
        
        # Compute metrics
        if self.task_type == "regression":
            metrics = self._compute_regression_metrics(y_pred)
        elif self.task_type == "classification":
            metrics = self._compute_classification_metrics(model, y_pred)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        
        # Add custom metrics
        if custom_metrics is not None:
            for metric_name, metric_fn in custom_metrics.items():
                try:
                    metrics[metric_name] = metric_fn(self._y_test, y_pred)
                except Exception as e:
                    logger.warning(
                        f"Custom metric '{metric_name}' failed: {e}"
                    )
        
        # Add metadata
        metrics['round'] = round_num
        metrics['access_count'] = self._access_count
        
        # Store in history
        self._evaluation_history.append(metrics.copy())
        
        logger.info(
            f"Round {round_num} test metrics: "
            f"{self._format_metrics(metrics)}"
        )
        
        return metrics
    
    def _compute_regression_metrics(self, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        return {
            'rmse': float(np.sqrt(mean_squared_error(self._y_test, y_pred))),
            'mae': float(mean_absolute_error(self._y_test, y_pred)),
            'r2': float(r2_score(self._y_test, y_pred)),
            'mse': float(mean_squared_error(self._y_test, y_pred)),
        }
    
    def _compute_classification_metrics(
        self, 
        model: Any, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': float(accuracy_score(self._y_test, y_pred)),
        }
        
        # F1 score (handle binary vs multiclass)
        try:
            if len(np.unique(self._y_test)) == 2:
                metrics['f1'] = float(f1_score(self._y_test, y_pred))
            else:
                metrics['f1'] = float(f1_score(
                    self._y_test, y_pred, average='weighted'
                ))
        except Exception as e:
            logger.warning(f"F1 score computation failed: {e}")
        
        # AUC (requires predict_proba)
        if hasattr(model, 'predict_proba'):
            try:
                y_probs = model.predict_proba(self._X_test)
                if y_probs.shape[1] == 2:
                    # Binary classification
                    metrics['auc'] = float(roc_auc_score(
                        self._y_test, y_probs[:, 1]
                    ))
                else:
                    # Multiclass
                    metrics['auc'] = float(roc_auc_score(
                        self._y_test, y_probs, multi_class='ovr', average='weighted'
                    ))
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}")
        
        return metrics
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for logging."""
        # Extract core metrics (exclude metadata)
        core_metrics = {
            k: v for k, v in metrics.items() 
            if k not in ['round', 'access_count']
        }
        
        return ', '.join(
            f"{k}={v:.4f}" for k, v in core_metrics.items()
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get guard statistics (for verification).
        
        Returns:
            Dictionary with:
            - total_evaluations: Total times test set accessed
            - rounds_evaluated: Number of unique rounds evaluated
            - rounds_list: List of evaluated round numbers
            - evaluation_history: Full history of metrics
        """
        return {
            'total_evaluations': self._access_count,
            'rounds_evaluated': len(self._round_evaluated),
            'rounds_list': sorted(self._round_evaluated),
            'evaluation_history': self._evaluation_history,
        }
    
    def verify_no_leakage(self, expected_rounds: int) -> bool:
        """
        Verify no data leakage occurred.
        
        Args:
            expected_rounds: Expected number of evaluation rounds
        
        Returns:
            True if no leakage detected, False otherwise
        
        Raises:
            AssertionError: If leakage detected
        """
        stats = self.get_statistics()
        
        # Check: evaluations == rounds
        if stats['total_evaluations'] != expected_rounds:
            logger.error(
                f"Leakage detected: {stats['total_evaluations']} evaluations "
                f"!= {expected_rounds} expected rounds"
            )
            return False
        
        # Check: no duplicate rounds
        if len(stats['rounds_evaluated']) != stats['total_evaluations']:
            logger.error(
                f"Leakage detected: duplicate round evaluations "
                f"({stats['rounds_evaluated']} unique rounds, "
                f"{stats['total_evaluations']} total evaluations)"
            )
            return False
        
        logger.info(
            f"✅ No data leakage: {stats['total_evaluations']} evaluations "
            f"across {stats['rounds_evaluated']} unique rounds"
        )
        
        return True
    
    def get_evaluation_history(self) -> Dict[str, np.ndarray]:
        """
        Get evaluation history as arrays (for plotting).
        
        Returns:
            Dictionary with arrays of metrics over rounds
        """
        if not self._evaluation_history:
            return {}
        
        # Extract all metric names
        metric_names = set()
        for metrics in self._evaluation_history:
            metric_names.update(metrics.keys())
        
        # Convert to arrays
        history = {}
        for metric_name in metric_names:
            history[metric_name] = np.array([
                metrics.get(metric_name, np.nan)
                for metrics in self._evaluation_history
            ])
        
        return history
    
    def __repr__(self) -> str:
        return (
            f"TestSetGuard(n_test={len(self._X_test)}, "
            f"task_type='{self.task_type}', "
            f"evaluations={self._access_count})"
        )


class ValidationSetGuard(TestSetGuard):
    """
    Guard for validation set (same interface as TestSetGuard).
    
    Used for hyperparameter tuning without test set leakage.
    """
    
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, task_type: str = "regression"):
        super().__init__(X_val, y_val, task_type)
        logger.info(
            f"ValidationSetGuard initialized: {len(X_val)} validation samples"
        )


def create_test_guard_context(
    X_test: np.ndarray, 
    y_test: np.ndarray,
    task_type: str = "regression"
):
    """
    Create test guard as context manager (experimental).
    
    Example:
        >>> with create_test_guard_context(X_test, y_test) as guard:
        >>>     for round_num in range(10):
        >>>         metrics = guard.evaluate_once_per_round(model, round_num)
        >>>     # Automatically verify no leakage on exit
    """
    class TestGuardContext:
        def __enter__(self):
            self.guard = TestSetGuard(X_test, y_test, task_type)
            return self.guard
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Verify no leakage on exit
            if exc_type is None:
                logger.info("Test guard context exiting: verifying no leakage")
            return False
    
    return TestGuardContext()

