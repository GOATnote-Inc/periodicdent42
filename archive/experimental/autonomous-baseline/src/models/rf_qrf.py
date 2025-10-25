"""Random Forest with Quantile Regression Forest for uncertainty quantification.

Uses ensemble variance and quantile predictions to estimate epistemic uncertainty.
Fast, robust baseline suitable for production deployment.
"""

import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseUncertaintyModel


class RandomForestQRF(BaseUncertaintyModel):
    """
    Random Forest with Quantile Regression Forest (QRF) uncertainty.
    
    Provides two types of uncertainty:
    1. Epistemic (model uncertainty): Variance across trees
    2. Prediction intervals: Quantile predictions from individual trees
    
    Suitable for:
    - Fast baseline (no hyperparameter tuning needed)
    - Production deployment (low latency)
    - Robust to outliers
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 30,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize Random Forest QRF.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            max_features: Number of features for best split ("sqrt", "log2", int, float)
            bootstrap: Whether to use bootstrap samples
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        super().__init__(random_state=random_state)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        
        self.model_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestQRF":
        """
        Train Random Forest model.
        
        Args:
            X: Feature matrix (N, D)
            y: Target values (N,)
            
        Returns:
            self
        """
        X = self._validate_features(X)
        
        self.n_features_in_ = X.shape[1]
        
        # Create and train Random Forest
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        
        self.model_.fit(X, y)
        self.fitted_ = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point predictions (mean across trees).
        
        Args:
            X: Feature matrix (N, D)
            
        Returns:
            Predictions (N,)
        """
        self._check_fitted()
        X = self._validate_features(X)
        
        return self.model_.predict(X)
    
    def predict_with_uncertainty(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty intervals.
        
        Uses quantile predictions from individual trees to construct
        prediction intervals. Also returns variance across trees.
        
        Args:
            X: Feature matrix (N, D)
            alpha: Significance level (default: 0.05 for 95% PI)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        self._check_fitted()
        X = self._validate_features(X)
        
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model_.estimators_
        ])  # Shape: (n_estimators, N)
        
        # Mean prediction
        predictions = tree_predictions.mean(axis=0)
        
        # Quantile-based prediction intervals
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_bounds = np.quantile(tree_predictions, lower_quantile, axis=0)
        upper_bounds = np.quantile(tree_predictions, upper_quantile, axis=0)
        
        return predictions, lower_bounds, upper_bounds
    
    def get_epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get epistemic uncertainty (variance across trees).
        
        Args:
            X: Feature matrix (N, D)
            
        Returns:
            Standard deviations (N,)
        """
        self._check_fitted()
        X = self._validate_features(X)
        
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model_.estimators_
        ])
        
        # Return standard deviation across trees
        return tree_predictions.std(axis=0)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the Random Forest.
        
        Returns:
            Feature importances (D,)
        """
        self._check_fitted()
        return self.model_.feature_importances_
    
    def _save_artifacts(self, path: Path) -> None:
        """Save model artifacts."""
        # Save sklearn model
        model_path = path.with_suffix(".pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model_, f)
        
        # Save metadata
        metadata = {
            "model_type": "RandomForestQRF",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_features_in": self.n_features_in_,
            "feature_importances": self.model_.feature_importances_.tolist() if self.model_ else None,
        }
        
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_artifacts(self, path: Path) -> None:
        """Load model artifacts."""
        # Load sklearn model
        model_path = path.with_suffix(".pkl")
        with open(model_path, "rb") as f:
            self.model_ = pickle.load(f)
        
        # Load metadata
        metadata_path = path.with_suffix(".json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Restore attributes
        self.n_estimators = metadata["n_estimators"]
        self.max_depth = metadata["max_depth"]
        self.min_samples_split = metadata["min_samples_split"]
        self.min_samples_leaf = metadata["min_samples_leaf"]
        self.max_features = metadata["max_features"]
        self.bootstrap = metadata["bootstrap"]
        self.random_state = metadata["random_state"]
        self.n_features_in_ = metadata["n_features_in"]
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "model_type": "RandomForestQRF",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "n_features": self.n_features_in_,
        })
        return params

