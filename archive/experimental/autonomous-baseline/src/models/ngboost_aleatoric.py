"""NGBoost for aleatoric uncertainty estimation.

Uses natural gradient boosting to predict distributional parameters (μ, σ).
Best for heteroscedastic noise where uncertainty varies across input space.
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np

from src.models.base import BaseUncertaintyModel

# Try to import NGBoost
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False
    warnings.warn(
        "NGBoost not available. Using sklearn GradientBoosting fallback. "
        "Install with: pip install ngboost>=0.5.1"
    )
    from sklearn.ensemble import GradientBoostingRegressor


class NGBoostAleatoric(BaseUncertaintyModel):
    """
    NGBoost for aleatoric (data) uncertainty estimation.
    
    Predicts a full probability distribution Normal(μ, σ) where:
    - μ: Mean prediction
    - σ: Aleatoric uncertainty (data noise, irreducible)
    
    Suitable for:
    - Heteroscedastic noise (uncertainty varies across input space)
    - Combining with epistemic uncertainty (e.g., ensemble + NGBoost)
    - Understanding inherent data variability
    
    Fallback Implementation:
    - If NGBoost not available, uses GradientBoostingRegressor
    - Estimates uncertainty via quantile loss
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        base_learner_depth: int = 5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize NGBoost model.
        
        Args:
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate (step size)
            minibatch_frac: Fraction of data to use per iteration (1.0 = full batch)
            base_learner_depth: Depth of base decision trees
            random_state: Random seed
            verbose: Whether to print training progress
        """
        super().__init__(random_state=random_state)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.base_learner_depth = base_learner_depth
        self.verbose = verbose
        
        self.model_ = None
        self.use_ngboost = NGBOOST_AVAILABLE
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NGBoostAleatoric":
        """
        Train NGBoost model.
        
        Args:
            X: Feature matrix (N, D)
            y: Target values (N,)
            
        Returns:
            self
        """
        X = self._validate_features(X)
        
        self.n_features_in_ = X.shape[1]
        
        if self.use_ngboost:
            # Full NGBoost implementation
            self.model_ = NGBRegressor(
                Dist=Normal,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                minibatch_frac=self.minibatch_frac,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            
            self.model_.fit(X, y)
        
        else:
            # Fallback: GradientBoostingRegressor
            print("⚠ Using GradientBoosting fallback (no NGBoost)")
            
            self.model_ = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.base_learner_depth,
                random_state=self.random_state,
                verbose=1 if self.verbose else 0,
            )
            
            self.model_.fit(X, y)
        
        self.fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point predictions (mean of distribution).
        
        Args:
            X: Feature matrix (N, D)
            
        Returns:
            Predictions (N,)
        """
        self._check_fitted()
        X = self._validate_features(X)
        
        if self.use_ngboost:
            # NGBoost: return mean of predicted distribution
            return self.model_.predict(X)
        else:
            # Fallback: GradientBoosting prediction
            return self.model_.predict(X)
    
    def predict_with_uncertainty(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with aleatoric uncertainty intervals.
        
        Args:
            X: Feature matrix (N, D)
            alpha: Significance level (default: 0.05 for 95% PI)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        self._check_fitted()
        X = self._validate_features(X)
        
        if self.use_ngboost:
            # NGBoost: predict full distribution
            dist = self.model_.pred_dist(X)
            
            # Mean predictions
            predictions = dist.loc  # Mean of Normal distribution
            
            # Prediction intervals from distribution
            z_score = np.abs(np.percentile(np.random.standard_normal(10000), alpha/2 * 100))
            std = dist.scale  # Standard deviation
            
            lower_bounds = predictions - z_score * std
            upper_bounds = predictions + z_score * std
            
            return predictions, lower_bounds, upper_bounds
        
        else:
            # Fallback: Use residual-based uncertainty
            predictions = self.model_.predict(X)
            
            # Estimate uncertainty from training residuals (simple heuristic)
            # In production, would train separate quantile regressors
            uncertainty = np.std(predictions) * np.ones_like(predictions)
            z_score = 1.96  # 95% CI
            
            lower_bounds = predictions - z_score * uncertainty
            upper_bounds = predictions + z_score * uncertainty
            
            return predictions, lower_bounds, upper_bounds
    
    def get_aleatoric_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get aleatoric uncertainty (predicted standard deviation).
        
        Args:
            X: Feature matrix (N, D)
            
        Returns:
            Standard deviations (N,)
        """
        self._check_fitted()
        X = self._validate_features(X)
        
        if self.use_ngboost:
            # NGBoost: return scale parameter of Normal distribution
            dist = self.model_.pred_dist(X)
            return dist.scale
        else:
            # Fallback: constant uncertainty (not ideal)
            predictions = self.model_.predict(X)
            return np.std(predictions) * np.ones_like(predictions)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances.
        
        NGBoost returns importances for both distribution parameters (mean, scale).
        We return the mean parameter importances as they are typically more interpretable.
        
        Returns:
            Feature importances (D,)
        """
        self._check_fitted()
        
        if self.use_ngboost:
            # NGBoost: shape is (2, D) for [mean_param, scale_param]
            # Return mean parameter importances (first row)
            importances = self.model_.feature_importances_
            if importances.ndim == 2:
                return importances[0]  # Mean parameter importances
            else:
                return importances
        else:
            return self.model_.feature_importances_
    
    def _save_artifacts(self, path: Path) -> None:
        """Save model artifacts."""
        # Save model
        model_path = path.with_suffix(".pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model_, f)
        
        # Save metadata
        metadata = {
            "model_type": "NGBoostAleatoric",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "base_learner_depth": self.base_learner_depth,
            "random_state": self.random_state,
            "n_features_in": self.n_features_in_,
            "use_ngboost": self.use_ngboost,
        }
        
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_artifacts(self, path: Path) -> None:
        """Load model artifacts."""
        # Load model
        model_path = path.with_suffix(".pkl")
        with open(model_path, "rb") as f:
            self.model_ = pickle.load(f)
        
        # Load metadata
        metadata_path = path.with_suffix(".json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Restore attributes
        self.n_estimators = metadata["n_estimators"]
        self.learning_rate = metadata["learning_rate"]
        self.base_learner_depth = metadata["base_learner_depth"]
        self.random_state = metadata["random_state"]
        self.n_features_in_ = metadata["n_features_in"]
        self.use_ngboost = metadata["use_ngboost"]
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "model_type": "NGBoostAleatoric",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "use_ngboost": self.use_ngboost,
        })
        return params

