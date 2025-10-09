"""Feature scaling with persistence for deployment."""

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class ScalerMetadata(BaseModel):
    """Metadata for fitted scaler."""
    
    scaler_type: Literal["standard", "robust", "minmax"]
    feature_names: list[str]
    n_features: int
    n_samples_fit: int
    means: list[float] | None = None
    stds: list[float] | None = None
    mins: list[float] | None = None
    maxs: list[float] | None = None


class FeatureScaler:
    """
    Wrapper for sklearn scalers with persistence and metadata.
    
    Supports StandardScaler (default), RobustScaler, and MinMaxScaler.
    """
    
    def __init__(
        self,
        method: Literal["standard", "robust", "minmax"] = "standard",
        feature_names: list[str] | None = None,
    ):
        """
        Initialize scaler.
        
        Args:
            method: Scaling method ("standard", "robust", or "minmax")
            feature_names: Names of features (for validation)
        """
        self.method = method
        self.feature_names = feature_names
        self.fitted_ = False
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit(self, X: np.ndarray | pd.DataFrame, feature_names: list[str] | None = None) -> "FeatureScaler":
        """
        Fit scaler to data.
        
        Args:
            X: Feature matrix (N, D)
            feature_names: Feature names (optional, for DataFrame)
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        self.scaler.fit(X)
        self.fitted_ = True
        self.n_samples_fit_ = X.shape[0]
        self.n_features_ = X.shape[1]
        
        return self
    
    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Args:
            X: Feature matrix (N, D)
            
        Returns:
            Scaled features (same type as input)
        """
        if not self.fitted_:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        is_dataframe = isinstance(X, pd.DataFrame)
        
        if is_dataframe:
            X_values = X.values
            X_scaled = self.scaler.transform(X_values)
            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray | pd.DataFrame, feature_names: list[str] | None = None) -> np.ndarray | pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            X: Scaled feature matrix (N, D)
            
        Returns:
            Original-scale features
        """
        if not self.fitted_:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        is_dataframe = isinstance(X, pd.DataFrame)
        
        if is_dataframe:
            X_values = X.values
            X_original = self.scaler.inverse_transform(X_values)
            return pd.DataFrame(X_original, columns=X.columns, index=X.index)
        else:
            return self.scaler.inverse_transform(X)
    
    def save(self, path: Path) -> None:
        """
        Save scaler to disk.
        
        Args:
            path: Path to save scaler (will create .pkl and .json)
        """
        if not self.fitted_:
            raise ValueError("Cannot save unfitted scaler")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save scaler object
        scaler_path = path.with_suffix(".pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = self.get_metadata()
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        
        print(f"✓ Saved scaler to {scaler_path}")
        print(f"✓ Saved metadata to {metadata_path}")
    
    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        """
        Load scaler from disk.
        
        Args:
            path: Path to scaler file
            
        Returns:
            Loaded FeatureScaler instance
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path.with_suffix(".json")
        with open(metadata_path) as f:
            metadata_dict = json.load(f)
        
        metadata = ScalerMetadata(**metadata_dict)
        
        # Load scaler object
        scaler_path = path.with_suffix(".pkl")
        with open(scaler_path, "rb") as f:
            scaler_obj = pickle.load(f)
        
        # Create FeatureScaler instance
        instance = cls(method=metadata.scaler_type, feature_names=metadata.feature_names)
        instance.scaler = scaler_obj
        instance.fitted_ = True
        instance.n_samples_fit_ = metadata.n_samples_fit
        instance.n_features_ = metadata.n_features
        
        print(f"✓ Loaded scaler from {scaler_path}")
        
        return instance
    
    def get_metadata(self) -> ScalerMetadata:
        """Generate metadata for fitted scaler."""
        if not self.fitted_:
            raise ValueError("Scaler not fitted")
        
        metadata = ScalerMetadata(
            scaler_type=self.method,
            feature_names=self.feature_names or [],
            n_features=self.n_features_,
            n_samples_fit=self.n_samples_fit_,
        )
        
        # Add scaler-specific statistics
        if self.method == "standard":
            metadata.means = self.scaler.mean_.tolist()
            metadata.stds = self.scaler.scale_.tolist()
        elif self.method == "robust":
            metadata.means = self.scaler.center_.tolist()
            metadata.stds = self.scaler.scale_.tolist()
        elif self.method == "minmax":
            metadata.mins = self.scaler.data_min_.tolist()
            metadata.maxs = self.scaler.data_max_.tolist()
        
        return metadata
    
    def get_feature_stats(self) -> pd.DataFrame:
        """Get feature statistics as DataFrame."""
        if not self.fitted_:
            raise ValueError("Scaler not fitted")
        
        metadata = self.get_metadata()
        
        stats = {
            "feature": metadata.feature_names or [f"feature_{i}" for i in range(metadata.n_features)],
        }
        
        if metadata.means is not None:
            stats["mean"] = metadata.means
        if metadata.stds is not None:
            stats["std"] = metadata.stds
        if metadata.mins is not None:
            stats["min"] = metadata.mins
        if metadata.maxs is not None:
            stats["max"] = metadata.maxs
        
        return pd.DataFrame(stats)


def scale_features(
    X_train: np.ndarray | pd.DataFrame,
    X_val: np.ndarray | pd.DataFrame | None = None,
    X_test: np.ndarray | pd.DataFrame | None = None,
    method: Literal["standard", "robust", "minmax"] = "standard",
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray | pd.DataFrame, ...]:
    """
    Convenience function to scale train/val/test sets together.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)
        method: Scaling method
        feature_names: Feature names
        
    Returns:
        Tuple of (scaler, X_train_scaled, X_val_scaled, X_test_scaled)
        Returns None for val/test if not provided
    """
    scaler = FeatureScaler(method=method, feature_names=feature_names)
    
    X_train_scaled = scaler.fit_transform(X_train, feature_names)
    
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled

