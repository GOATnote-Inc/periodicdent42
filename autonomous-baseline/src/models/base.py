"""Base model interface for uncertainty-aware Tc prediction models.

All uncertainty models must implement the UncertaintyModel protocol to ensure
consistent API for training, prediction, and uncertainty quantification.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, Tuple

import numpy as np


class UncertaintyModel(Protocol):
    """
    Protocol for uncertainty-aware regression models.
    
    All models must provide:
    - fit(X, y): Train the model
    - predict(X): Point predictions (mean)
    - predict_with_uncertainty(X): Predictions + uncertainty intervals
    - save(path): Save model to disk
    - load(path): Load model from disk (classmethod)
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "UncertaintyModel":
        """
        Train the model on data.
        
        Args:
            X: Feature matrix (N, D)
            y: Target values (N,)
            
        Returns:
            self (for chaining)
        """
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point predictions (mean/median).
        
        Args:
            X: Feature matrix (N, D)
            
        Returns:
            Predictions (N,)
        """
        ...
    
    def predict_with_uncertainty(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty intervals.
        
        Args:
            X: Feature matrix (N, D)
            alpha: Significance level for (1-alpha) confidence (default: 0.05 for 95%)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
            All arrays have shape (N,)
        """
        ...
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        ...
    
    @classmethod
    def load(cls, path: Path) -> "UncertaintyModel":
        """Load model from disk."""
        ...


class BaseUncertaintyModel(ABC):
    """
    Abstract base class for uncertainty models with common functionality.
    
    Subclasses must implement:
    - fit(X, y)
    - predict(X)
    - predict_with_uncertainty(X, alpha)
    - _save_artifacts(path)
    - _load_artifacts(path)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize base model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.fitted_ = False
        self.feature_names_in_ = None
        self.n_features_in_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseUncertaintyModel":
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions."""
        pass
    
    @abstractmethod
    def predict_with_uncertainty(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty."""
        pass
    
    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.fitted_:
            raise ValueError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )
    
    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        """Validate input features."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.n_features_in_ is not None:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {X.shape[1]}"
                )
        
        return X
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model (will create .pkl and metadata)
        """
        if not self.fitted_:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model-specific artifacts (implemented by subclass)
        self._save_artifacts(path)
        
        print(f"✓ Saved {self.__class__.__name__} to {path}")
    
    @abstractmethod
    def _save_artifacts(self, path: Path) -> None:
        """Save model-specific artifacts (implemented by subclass)."""
        pass
    
    @classmethod
    def load(cls, path: Path) -> "BaseUncertaintyModel":
        """
        Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        # Create instance and load artifacts
        instance = cls()
        instance._load_artifacts(path)
        instance.fitted_ = True
        
        print(f"✓ Loaded {cls.__name__} from {path}")
        
        return instance
    
    @abstractmethod
    def _load_artifacts(self, path: Path) -> None:
        """Load model-specific artifacts (implemented by subclass)."""
        pass
    
    def get_params(self) -> dict:
        """Get model parameters as dictionary."""
        return {
            "random_state": self.random_state,
            "fitted": self.fitted_,
            "n_features": self.n_features_in_,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.fitted_ else "unfitted"
        return f"{self.__class__.__name__}(random_state={self.random_state}, status={status})"

