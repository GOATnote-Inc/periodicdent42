"""
Variance calibration for regression models.

Calibrates predicted standard deviations to match empirical errors.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Tuple


class VarianceCalibrator:
    """
    Calibrate predicted uncertainties using isotonic regression.
    
    Maps predicted standard deviations to calibrated standard deviations
    such that the empirical coverage matches the nominal coverage.
    
    Reference:
        Kuleshov, V., Fenner, N., & Ermon, S. (2018).
        "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
    """
    
    def __init__(self):
        self.isotonic_regressor_ = None
        self.fitted_ = False
    
    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray
    ) -> "VarianceCalibrator":
        """
        Fit the variance calibrator.
        
        Args:
            y_true: True values (N,)
            y_pred: Predicted values (N,)
            y_std: Predicted standard deviations (N,)
        
        Returns:
            self
        """
        # Compute absolute residuals
        residuals = np.abs(y_true - y_pred)
        
        # Fit isotonic regression: residuals ~ predicted_std
        self.isotonic_regressor_ = IsotonicRegression(
            y_min=residuals.min(),
            y_max=residuals.max(),
            out_of_bounds='clip'
        )
        self.isotonic_regressor_.fit(y_std, residuals)
        
        self.fitted_ = True
        return self
    
    def transform(self, y_std: np.ndarray) -> np.ndarray:
        """
        Transform predicted standard deviations to calibrated ones.
        
        Args:
            y_std: Predicted standard deviations (N,)
        
        Returns:
            Calibrated standard deviations (N,)
        """
        if not self.fitted_:
            raise ValueError("VarianceCalibrator not fitted")
        
        # Predict calibrated standard deviations
        y_std_calibrated = self.isotonic_regressor_.predict(y_std)
        
        return y_std_calibrated
    
    def fit_transform(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray
    ) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            y_true: True values (N,)
            y_pred: Predicted values (N,)
            y_std: Predicted standard deviations (N,)
        
        Returns:
            Calibrated standard deviations (N,)
        """
        self.fit(y_true, y_pred, y_std)
        return self.transform(y_std)


class CalibratedUncertaintyModel:
    """
    Wrapper that calibrates a base model's uncertainty estimates.
    
    Uses variance calibration + conformal prediction for both:
    1. Calibrated probabilistic uncertainties (via isotonic regression)
    2. Valid prediction intervals (via conformal prediction)
    """
    
    def __init__(
        self,
        base_model,
        conformal_predictor,
        random_state: int = 42
    ):
        self.base_model = base_model
        self.conformal_predictor = conformal_predictor
        self.variance_calibrator = VarianceCalibrator()
        self.random_state = random_state
        self.fitted_ = False
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """
        Fit base model, calibrate variances, and fit conformal predictor.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for calibration)
            y_val: Validation targets (for calibration)
        """
        # 1. Train base model
        self.base_model.fit(X_train, y_train)
        
        # 2. Calibrate variances on validation set
        y_val_pred = self.base_model.predict(X_val)
        y_val_std = self.base_model.get_epistemic_uncertainty(X_val)
        self.variance_calibrator.fit(y_val, y_val_pred, y_val_std)
        
        # 3. Fit conformal predictor (already trained base model)
        self.conformal_predictor.base_model = self.base_model
        self.conformal_predictor.fit(X_train, y_train, X_val, y_val)
        
        self.fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point predictions."""
        return self.base_model.predict(X)
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with CALIBRATED uncertainty estimates.
        
        Returns:
            y_pred: Point predictions
            y_std_calibrated: Calibrated standard deviations
        """
        y_pred = self.base_model.predict(X)
        y_std = self.base_model.get_epistemic_uncertainty(X)
        y_std_calibrated = self.variance_calibrator.transform(y_std)
        
        return y_pred, y_std_calibrated
    
    def predict_with_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Predict with calibrated intervals from conformal prediction.
        
        Returns:
            y_pred: Point predictions
            y_lower: Lower bounds
            y_upper: Upper bounds
            expected_coverage: Theoretical coverage
        """
        return self.conformal_predictor.predict_with_interval(X, alpha=alpha)

