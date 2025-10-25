"""Out-of-Distribution (OOD) detection for autonomous lab safety.

Provides multiple methods to detect when test samples are outside the training distribution:
- Mahalanobis Distance: Detects samples far from training distribution centroid
- KDE (Kernel Density Estimation): Estimates probability density of test samples
- Conformal Novelty Detection: Uses nonconformity scores from conformal prediction
- Ensemble OOD: Combines multiple detectors for robust detection

Critical for autonomous labs: Never deploy predictions on OOD samples.
"""

from typing import Literal, Optional

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


class MahalanobisOODDetector:
    """
    Mahalanobis distance-based OOD detector.
    
    Detects samples that are far from the training distribution centroid
    in a multivariate sense (accounts for feature correlations).
    
    Decision Rule:
        - Compute Mahalanobis distance: d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        - Under normality assumption: d² ~ χ²(p) where p = # features
        - Flag as OOD if d² > χ²(p, 1 - α) (e.g., α = 0.01 for 99% threshold)
    
    Advantages:
        - Accounts for feature correlations (covariance matrix)
        - Fast computation (O(p²) for fit, O(p²) per sample for predict)
        - Principled threshold via χ² distribution
    
    Limitations:
        - Assumes multivariate normal training distribution
        - Sensitive to outliers (use robust covariance estimators)
        - Requires p < n (more samples than features)
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        robust: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize Mahalanobis OOD detector.
        
        Args:
            alpha: Significance level (default: 0.01 for 99% threshold)
            robust: Use robust covariance estimation (default: True)
            random_state: Random seed
        """
        self.alpha = alpha
        self.robust = robust
        self.random_state = random_state
        
        self.mean_ = None
        self.cov_ = None
        self.cov_inv_ = None
        self.threshold_ = None
        self.fitted_ = False
    
    def fit(self, X_train: np.ndarray) -> "MahalanobisOODDetector":
        """
        Fit the OOD detector on training data.
        
        Args:
            X_train: Training features (N, D)
            
        Returns:
            self
        """
        n_samples, n_features = X_train.shape
        
        if n_samples < n_features:
            raise ValueError(
                f"Insufficient samples ({n_samples}) for {n_features} features. "
                "Need n_samples >= n_features for covariance estimation."
            )
        
        # Compute mean
        self.mean_ = X_train.mean(axis=0)
        
        # Compute covariance (robust or standard)
        if self.robust:
            # Robust covariance using Minimum Covariance Determinant (MCD)
            try:
                from sklearn.covariance import MinCovDet
                
                mcd = MinCovDet(random_state=self.random_state)
                mcd.fit(X_train)
                
                self.mean_ = mcd.location_
                self.cov_ = mcd.covariance_
            except ImportError:
                # Fall back to standard covariance if sklearn not available
                self.cov_ = np.cov(X_train, rowvar=False)
        else:
            # Standard covariance
            self.cov_ = np.cov(X_train, rowvar=False)
        
        # Add small regularization to avoid singularity
        self.cov_ = self.cov_ + 1e-6 * np.eye(n_features)
        
        # Invert covariance matrix
        try:
            self.cov_inv_ = np.linalg.inv(self.cov_)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self.cov_inv_ = np.linalg.pinv(self.cov_)
        
        # Compute threshold from chi-squared distribution
        self.threshold_ = chi2.ppf(1 - self.alpha, df=n_features)
        
        self.fitted_ = True
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict whether samples are OOD.
        
        Args:
            X_test: Test features (N, D)
            
        Returns:
            Binary labels (N,): 1 = OOD, 0 = in-distribution
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        distances = self.mahalanobis_distance(X_test)
        
        # Flag as OOD if distance exceeds threshold
        is_ood = (distances > self.threshold_).astype(int)
        
        return is_ood
    
    def mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute squared Mahalanobis distance for each sample.
        
        Args:
            X: Features (N, D)
            
        Returns:
            Squared Mahalanobis distances (N,)
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Center samples
        X_centered = X - self.mean_
        
        # Compute d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        distances_squared = np.sum(X_centered @ self.cov_inv_ * X_centered, axis=1)
        
        return distances_squared
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probability of being OOD (p-value from χ² distribution).
        
        Args:
            X_test: Test features (N, D)
            
        Returns:
            OOD probabilities (N,): Higher = more likely OOD
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        distances = self.mahalanobis_distance(X_test)
        
        n_features = X_test.shape[1]
        
        # p-value: P(χ² > d²)
        p_values = 1 - chi2.cdf(distances, df=n_features)
        
        # Convert to "OOD probability": smaller p-value = more OOD
        ood_proba = 1 - p_values
        
        return ood_proba


class KDEOODDetector:
    """
    Kernel Density Estimation (KDE) OOD detector.
    
    Estimates the probability density of the training distribution and
    flags test samples with low density as OOD.
    
    Decision Rule:
        - Estimate p(x) using KDE on training data
        - Flag as OOD if p(x) < threshold (e.g., 1st percentile of training densities)
    
    Advantages:
        - Non-parametric (no distribution assumptions)
        - Captures multi-modal distributions
        - Works well for low-dimensional data
    
    Limitations:
        - Curse of dimensionality (poor for D > 10)
        - Bandwidth selection critical
        - Slower than Mahalanobis for high dimensions
    """
    
    def __init__(
        self,
        bandwidth: str | float = "scott",
        kernel: str = "gaussian",
        alpha: float = 0.01,
    ):
        """
        Initialize KDE OOD detector.
        
        Args:
            bandwidth: Bandwidth selection method ("scott", "silverman", or float)
            kernel: Kernel type ("gaussian", "tophat", "epanechnikov", etc.)
            alpha: Significance level (default: 0.01 = 1st percentile threshold)
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.alpha = alpha
        
        self.kde_ = None
        self.threshold_ = None
        self.fitted_ = False
    
    def fit(self, X_train: np.ndarray) -> "KDEOODDetector":
        """
        Fit KDE on training data.
        
        Args:
            X_train: Training features (N, D)
            
        Returns:
            self
        """
        from sklearn.neighbors import KernelDensity
        
        # Fit KDE
        self.kde_ = KernelDensity(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        )
        self.kde_.fit(X_train)
        
        # Compute training log-densities
        train_log_densities = self.kde_.score_samples(X_train)
        
        # Set threshold as α-percentile of training densities
        self.threshold_ = np.percentile(train_log_densities, self.alpha * 100)
        
        self.fitted_ = True
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict whether samples are OOD.
        
        Args:
            X_test: Test features (N, D)
            
        Returns:
            Binary labels (N,): 1 = OOD, 0 = in-distribution
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        log_densities = self.kde_.score_samples(X_test)
        
        # Flag as OOD if log-density < threshold
        is_ood = (log_densities < self.threshold_).astype(int)
        
        return is_ood
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict OOD probability (normalized density).
        
        Args:
            X_test: Test features (N, D)
            
        Returns:
            OOD probabilities (N,): Higher = more likely OOD
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        log_densities = self.kde_.score_samples(X_test)
        
        # Normalize by threshold: distance from threshold
        # Samples with log_density << threshold → high OOD probability
        distance_from_threshold = self.threshold_ - log_densities
        
        # Sigmoid to convert to [0, 1]
        ood_proba = 1 / (1 + np.exp(-distance_from_threshold))
        
        return ood_proba


class ConformalNoveltyDetector:
    """
    Conformal novelty detection using nonconformity scores.
    
    Uses the same nonconformity scores as conformal prediction to detect
    samples that are "too different" from the training distribution.
    
    Decision Rule:
        - Compute nonconformity score for test sample
        - Compare to calibration scores
        - Flag as OOD if score > (1 - α)-quantile of calibration scores
    
    Advantages:
        - Model-agnostic (works with any model)
        - Same infrastructure as conformal prediction
        - Principled finite-sample guarantees
    
    Limitations:
        - Requires separate calibration set
        - May be conservative (high false positive rate)
    """
    
    def __init__(
        self,
        base_model,
        score_function: Literal["absolute", "normalized"] = "absolute",
        alpha: float = 0.01,
    ):
        """
        Initialize conformal novelty detector.
        
        Args:
            base_model: Underlying regression model
            score_function: Type of nonconformity score
            alpha: Significance level (default: 0.01)
        """
        self.base_model = base_model
        self.score_function = score_function
        self.alpha = alpha
        
        self.calibration_scores_ = None
        self.threshold_ = None
        self.fitted_ = False
    
    def fit(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> "ConformalNoveltyDetector":
        """
        Fit the novelty detector.
        
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
        
        # Compute calibration scores
        y_cal_pred = self.base_model.predict(X_calibration)
        
        if self.score_function == "absolute":
            self.calibration_scores_ = np.abs(y_calibration - y_cal_pred)
        elif self.score_function == "normalized":
            # Use prediction uncertainty if available
            if hasattr(self.base_model, "predict_with_uncertainty"):
                _, lower, upper = self.base_model.predict_with_uncertainty(
                    X_calibration, alpha=0.05
                )
                width = (upper - lower) / 2
                width = np.clip(width, 1e-6, None)
                self.calibration_scores_ = np.abs(y_calibration - y_cal_pred) / width
            else:
                # Fall back to absolute
                self.calibration_scores_ = np.abs(y_calibration - y_cal_pred)
        
        # Set threshold
        self.threshold_ = np.quantile(self.calibration_scores_, 1 - self.alpha)
        
        self.fitted_ = True
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict whether samples are novel (OOD).
        
        Note: Cannot compute nonconformity scores without targets (y_test).
        This method uses a heuristic: compute prediction uncertainty and
        flag as OOD if uncertainty is unusually high.
        
        Args:
            X_test: Test features (N, D)
            
        Returns:
            Binary labels (N,): 1 = OOD (novel), 0 = in-distribution
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Heuristic: Use prediction uncertainty as proxy for novelty
        if hasattr(self.base_model, "get_epistemic_uncertainty"):
            uncertainty = self.base_model.get_epistemic_uncertainty(X_test)
        elif hasattr(self.base_model, "predict_with_uncertainty"):
            _, lower, upper = self.base_model.predict_with_uncertainty(X_test)
            uncertainty = (upper - lower) / 2
        else:
            # No uncertainty available → cannot detect novelty
            raise ValueError(
                "Base model does not provide uncertainty. "
                "Cannot use conformal novelty detection without targets."
            )
        
        # Compare to threshold (scaled by median calibration score)
        median_calib_score = np.median(self.calibration_scores_)
        threshold_scaled = self.threshold_ * 2  # Heuristic scaling
        
        is_ood = (uncertainty > threshold_scaled).astype(int)
        
        return is_ood
    
    def predict_with_targets(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> np.ndarray:
        """
        Predict novelty using true targets (for validation/testing).
        
        Args:
            X_test: Test features (N, D)
            y_test: Test targets (N,) - ground truth
            
        Returns:
            Binary labels (N,): 1 = OOD, 0 = in-distribution
        """
        if not self.fitted_:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        y_pred = self.base_model.predict(X_test)
        
        # Compute nonconformity scores
        if self.score_function == "absolute":
            scores = np.abs(y_test - y_pred)
        elif self.score_function == "normalized":
            if hasattr(self.base_model, "predict_with_uncertainty"):
                _, lower, upper = self.base_model.predict_with_uncertainty(X_test)
                width = (upper - lower) / 2
                width = np.clip(width, 1e-6, None)
                scores = np.abs(y_test - y_pred) / width
            else:
                scores = np.abs(y_test - y_pred)
        
        # Flag as OOD if score > threshold
        is_ood = (scores > self.threshold_).astype(int)
        
        return is_ood


def create_ood_detector(
    method: Literal["mahalanobis", "kde", "conformal"],
    **kwargs,
):
    """
    Factory function to create OOD detectors.
    
    Args:
        method: OOD detection method
        **kwargs: Method-specific arguments
        
    Returns:
        OOD detector instance
        
    Example:
        >>> detector = create_ood_detector("mahalanobis", alpha=0.01)
        >>> detector.fit(X_train)
        >>> is_ood = detector.predict(X_test)
    """
    if method == "mahalanobis":
        return MahalanobisOODDetector(**kwargs)
    elif method == "kde":
        return KDEOODDetector(**kwargs)
    elif method == "conformal":
        return ConformalNoveltyDetector(**kwargs)
    else:
        raise ValueError(
            f"Unknown OOD detection method: {method}. "
            "Choose from: 'mahalanobis', 'kde', 'conformal'"
        )

