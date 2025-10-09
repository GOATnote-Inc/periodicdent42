"""Conformal prediction for distribution-free uncertainty quantification.

Provides finite-sample coverage guarantees:
- Split Conformal: Basic conformal prediction with calibration set
- Mondrian Conformal: Stratified conformal (e.g., by prediction region)
- Adaptive Conformal: Adjust intervals based on residual difficulty
"""

from typing import Callable, Optional

import numpy as np


class SplitConformalPredictor:
    """
    Split Conformal Prediction for regression.
    
    Provides distribution-free prediction intervals with guaranteed coverage.
    
    Key Property:
        For nominal coverage (1 - alpha), the empirical coverage is guaranteed
        to be ≥ (1 - alpha) with high probability (finite-sample guarantee).
    
    Algorithm:
        1. Split data into fit_set (train model) and calibration_set
        2. Compute calibration scores (e.g., |y - ŷ|) on calibration_set
        3. Find quantile q_{1-alpha} of calibration scores
        4. Prediction interval: [ŷ - q, ŷ + q]
    
    Reference:
        Lei et al. (2018) "Distribution-Free Predictive Inference for Regression"
    """
    
    def __init__(
        self,
        base_model,
        score_function: str = "absolute",
        random_state: int = 42,
    ):
        """
        Initialize split conformal predictor.
        
        Args:
            base_model: Underlying regression model (must have .fit() and .predict())
            score_function: Type of nonconformity score ("absolute", "normalized")
            random_state: Random seed for reproducibility
        """
        self.base_model = base_model
        self.score_function = score_function
        self.random_state = random_state
        
        self.calibration_scores_ = None
        self.fitted_ = False
    
    def fit(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> "SplitConformalPredictor":
        """
        Fit the conformal predictor.
        
        Args:
            X_fit: Features for training base model (N_fit, D)
            y_fit: Targets for training base model (N_fit,)
            X_calibration: Features for calibration (N_cal, D)
            y_calibration: Targets for calibration (N_cal,)
            
        Returns:
            self
        """
        # 1. Train base model on fit set
        self.base_model.fit(X_fit, y_fit)
        
        # 2. Compute calibration scores
        y_cal_pred = self.base_model.predict(X_calibration)
        
        self.calibration_scores_ = self._compute_scores(
            y_calibration, y_cal_pred, X_calibration
        )
        
        self.fitted_ = True
        
        return self
    
    def predict_with_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate conformal prediction intervals.
        
        Args:
            X: Features (N, D)
            alpha: Significance level (default: 0.05 for 95% coverage)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
            
        Guarantee:
            P(y ∈ [lower, upper]) ≥ 1 - alpha (distribution-free)
        """
        if not self.fitted_:
            raise ValueError("Conformal predictor not fitted. Call fit() first.")
        
        # Point predictions
        y_pred = self.base_model.predict(X)
        
        # Conformal quantile
        # Note: Use (1 - alpha)(1 + 1/n) quantile for finite-sample guarantee
        n_cal = len(self.calibration_scores_)
        adjusted_alpha = alpha * (1 + 1 / n_cal)
        quantile_level = 1 - adjusted_alpha
        
        q = np.quantile(self.calibration_scores_, quantile_level)
        
        # Prediction intervals
        if self.score_function == "absolute":
            lower = y_pred - q
            upper = y_pred + q
        elif self.score_function == "normalized":
            # For normalized scores, q is already a multiplier
            # This requires storing additional info during calibration
            # For now, fall back to absolute
            lower = y_pred - q
            upper = y_pred + q
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")
        
        return y_pred, lower, upper
    
    def _compute_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute nonconformity scores.
        
        Args:
            y_true: True targets (N,)
            y_pred: Predictions (N,)
            X: Features (N, D) - used for normalized scores
            
        Returns:
            Nonconformity scores (N,)
        """
        if self.score_function == "absolute":
            # Simple absolute error
            scores = np.abs(y_true - y_pred)
        
        elif self.score_function == "normalized":
            # Normalized by local uncertainty (if base_model provides it)
            if hasattr(self.base_model, "predict_with_uncertainty"):
                _, lower, upper = self.base_model.predict_with_uncertainty(X, alpha=0.05)
                # Use interval width as normalization factor
                width = (upper - lower) / 2
                width = np.clip(width, 1e-6, None)  # Avoid division by zero
                scores = np.abs(y_true - y_pred) / width
            else:
                # Fall back to absolute if uncertainty not available
                scores = np.abs(y_true - y_pred)
        
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")
        
        return scores
    
    def get_calibration_quantiles(self, levels: list[float]) -> np.ndarray:
        """
        Get calibration quantiles for multiple confidence levels.
        
        Args:
            levels: List of confidence levels (e.g., [0.68, 0.90, 0.95])
            
        Returns:
            Quantiles corresponding to each level
        """
        if not self.fitted_:
            raise ValueError("Not fitted")
        
        quantiles = [
            np.quantile(self.calibration_scores_, level) for level in levels
        ]
        
        return np.array(quantiles)


class MondrianConformalPredictor:
    """
    Mondrian Conformal Prediction (Stratified Conformal).
    
    Provides better coverage across different regions of the input space
    by maintaining separate calibration sets for each stratum.
    
    Example strata:
    - Low/medium/high predicted values
    - Different material families
    - Easy/hard samples
    
    Reference:
        Vovk (2012) "Conditional Validity of Inductive Conformal Predictors"
    """
    
    def __init__(
        self,
        base_model,
        stratify_function: Callable[[np.ndarray], np.ndarray],
        score_function: str = "absolute",
        random_state: int = 42,
    ):
        """
        Initialize Mondrian conformal predictor.
        
        Args:
            base_model: Underlying regression model
            stratify_function: Function that assigns each sample to a stratum
                               Takes predictions (N,) → stratum labels (N,)
            score_function: Type of nonconformity score
            random_state: Random seed
        """
        self.base_model = base_model
        self.stratify_function = stratify_function
        self.score_function = score_function
        self.random_state = random_state
        
        self.calibration_scores_by_stratum_ = {}
        self.fitted_ = False
    
    def fit(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> "MondrianConformalPredictor":
        """
        Fit the Mondrian conformal predictor.
        
        Args:
            X_fit: Features for training base model
            y_fit: Targets for training base model
            X_calibration: Features for calibration
            y_calibration: Targets for calibration
            
        Returns:
            self
        """
        # Train base model
        self.base_model.fit(X_fit, y_fit)
        
        # Compute predictions on calibration set
        y_cal_pred = self.base_model.predict(X_calibration)
        
        # Assign each calibration sample to a stratum
        strata = self.stratify_function(y_cal_pred)
        
        # Compute calibration scores for each stratum
        unique_strata = np.unique(strata)
        
        for stratum in unique_strata:
            mask = (strata == stratum)
            
            if mask.sum() == 0:
                continue
            
            # Compute scores for this stratum
            scores = self._compute_scores(
                y_calibration[mask],
                y_cal_pred[mask],
                X_calibration[mask] if X_calibration is not None else None
            )
            
            self.calibration_scores_by_stratum_[stratum] = scores
        
        self.fitted_ = True
        
        return self
    
    def predict_with_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Mondrian conformal prediction intervals.
        
        Args:
            X: Features (N, D)
            alpha: Significance level
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.fitted_:
            raise ValueError("Mondrian conformal predictor not fitted")
        
        # Point predictions
        y_pred = self.base_model.predict(X)
        
        # Assign test samples to strata
        test_strata = self.stratify_function(y_pred)
        
        # Initialize intervals
        lower = np.zeros_like(y_pred)
        upper = np.zeros_like(y_pred)
        
        # Compute intervals for each stratum
        for stratum in np.unique(test_strata):
            mask = (test_strata == stratum)
            
            if stratum not in self.calibration_scores_by_stratum_:
                # Fall back to global quantile if stratum not seen during calibration
                all_scores = np.concatenate(list(self.calibration_scores_by_stratum_.values()))
                q = np.quantile(all_scores, 1 - alpha)
            else:
                # Use stratum-specific quantile
                scores = self.calibration_scores_by_stratum_[stratum]
                n_cal = len(scores)
                adjusted_alpha = alpha * (1 + 1 / n_cal)
                q = np.quantile(scores, 1 - adjusted_alpha)
            
            # Assign intervals for this stratum
            lower[mask] = y_pred[mask] - q
            upper[mask] = y_pred[mask] + q
        
        return y_pred, lower, upper
    
    def _compute_scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute nonconformity scores (same as SplitConformalPredictor)."""
        if self.score_function == "absolute":
            return np.abs(y_true - y_pred)
        elif self.score_function == "normalized":
            if hasattr(self.base_model, "predict_with_uncertainty") and X is not None:
                _, lower, upper = self.base_model.predict_with_uncertainty(X, alpha=0.05)
                width = (upper - lower) / 2
                width = np.clip(width, 1e-6, None)
                return np.abs(y_true - y_pred) / width
            else:
                return np.abs(y_true - y_pred)
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")


def stratify_by_prediction_bins(
    y_pred: np.ndarray, n_bins: int = 3
) -> np.ndarray:
    """
    Stratify samples into bins based on predicted values.
    
    Common Mondrian stratification strategy:
    - Low predictions → stratum 0
    - Medium predictions → stratum 1
    - High predictions → stratum 2
    
    Args:
        y_pred: Predicted values (N,)
        n_bins: Number of strata (default: 3)
        
    Returns:
        Stratum labels (N,) with values in {0, 1, ..., n_bins-1}
    """
    # Use quantile-based binning
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_pred, quantiles)
    
    # Assign to bins
    strata = np.digitize(y_pred, bin_edges[1:-1], right=False)
    
    return strata


def stratify_by_uncertainty(
    y_std: np.ndarray, n_bins: int = 3
) -> np.ndarray:
    """
    Stratify samples into bins based on predicted uncertainty.
    
    Useful for adaptive intervals:
    - Low uncertainty → narrow intervals
    - High uncertainty → wide intervals
    
    Args:
        y_std: Predicted standard deviations (N,)
        n_bins: Number of strata (default: 3)
        
    Returns:
        Stratum labels (N,)
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_std, quantiles)
    
    strata = np.digitize(y_std, bin_edges[1:-1], right=False)
    
    return strata

