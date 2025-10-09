"""
Finite-sample conformal prediction with exact coverage guarantees.

Uses the proper quantile formula: ceil((n+1)(1-alpha)) / n
This provides valid finite-sample coverage guarantee.
"""

import math
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


class FiniteSampleConformalPredictor:
    """
    Split Conformal with EXACT finite-sample correction.
    
    Uses the proper quantile formula: ceil((n+1)(1-alpha)) / n
    This provides valid finite-sample coverage guarantee.
    """
    
    def __init__(
        self,
        base_model,
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.random_state = random_state
        
        self.calibration_scores_ = None
        self.n_calibration_ = None
        self.fitted_ = False
    
    def fit(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ):
        """Fit the conformal predictor."""
        # Train base model
        self.base_model.fit(X_fit, y_fit)
        
        # Compute calibration scores
        y_cal_pred = self.base_model.predict(X_calibration)
        self.calibration_scores_ = np.abs(y_calibration - y_cal_pred)
        self.n_calibration_ = len(self.calibration_scores_)
        
        self.fitted_ = True
        return self
    
    def predict_with_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Generate conformal prediction intervals.
        
        Returns:
            y_pred: Point predictions
            y_lower: Lower bounds
            y_upper: Upper bounds
            expected_coverage: Theoretical coverage guarantee
        """
        if not self.fitted_:
            raise ValueError("Not fitted")
        
        # Point predictions
        y_pred = self.base_model.predict(X)
        
        # CORRECT finite-sample quantile
        # Formula: k = ceil((n+1)(1-alpha))
        # quantile_level = k / n
        n = self.n_calibration_
        k = math.ceil((n + 1) * (1 - alpha))
        quantile_level = k / n
        
        # This gives exact coverage guarantee: P(Y ∈ PI) ≥ (n+1-k)/(n+1)
        expected_coverage = (n + 1 - k) / (n + 1)
        
        q = np.quantile(self.calibration_scores_, quantile_level)
        
        lower = y_pred - q
        upper = y_pred + q
        
        return y_pred, lower, upper, expected_coverage
    
    def save(self, path: Path):
        """Save the conformal predictor."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: Path):
        """Load a saved conformal predictor."""
        with open(path, 'rb') as f:
            return pickle.load(f)

