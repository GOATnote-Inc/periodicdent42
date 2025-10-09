"""
Gaussian Process Model using BoTorch for Active Learning.

Key improvements over Random Forest:
1. Calibrated uncertainty (GP posterior)
2. Informative for active learning
3. Smooth predictions with uncertainty
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GPModel:
    """
    Gaussian Process model for Tc prediction with BoTorch.
    
    Features:
    - SingleTaskGP with Matérn 5/2 kernel (default)
    - Automatic relevance determination (ARD)
    - Calibrated uncertainty quantification
    - Active learning-ready
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64,
        random_state: int = 42
    ):
        """
        Initialize GP model.
        
        Args:
            device: 'cpu' or 'cuda'
            dtype: torch data type (float32 or float64)
            random_state: Random seed
        """
        self.device = device
        self.dtype = dtype
        self.random_state = random_state
        
        # Model components
        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None
        
        # Scalers
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # Training data (for refitting)
        self.X_train_np: Optional[np.ndarray] = None
        self.y_train_np: Optional[np.ndarray] = None
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.fitted_ = False
        
        logger.info(f"✅ GPModel initialized (device={device}, dtype={dtype})")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP model to training data.
        
        Args:
            X: Feature matrix (N, D)
            y: Target values (N,)
        """
        logger.info(f"🔧 Fitting GP on {len(X)} samples...")
        
        # Store raw data for refitting
        self.X_train_np = X.copy()
        self.y_train_np = y.copy()
        
        # Fit scalers
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Convert to torch tensors
        X_train = torch.tensor(X_scaled, dtype=self.dtype, device=self.device)
        y_train = torch.tensor(y_scaled, dtype=self.dtype, device=self.device)
        
        # Create SingleTaskGP
        self.model = SingleTaskGP(X_train, y_train.unsqueeze(-1))
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        # Fit hyperparameters
        fit_gpytorch_mll(self.mll)
        
        self.fitted_ = True
        logger.info("✅ GP fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions.
        
        Args:
            X: Feature matrix (N, D)
        
        Returns:
            Predictions (N,)
        """
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale and convert to torch
        X_scaled = self.X_scaler.transform(X)
        X_torch = torch.tensor(X_scaled, dtype=self.dtype, device=self.device)
        
        # Predict
        self.model.eval()  # type: ignore
        with torch.no_grad():
            posterior = self.model.posterior(X_torch)  # type: ignore
            y_pred_scaled = posterior.mean.cpu().numpy().ravel()
        
        # Inverse transform
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty (Gaussian posterior).
        
        Args:
            X: Feature matrix (N, D)
            return_std: If True, return standard deviation; if False, return variance
        
        Returns:
            Tuple of (predictions, uncertainty)
        """
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale and convert to torch
        X_scaled = self.X_scaler.transform(X)
        X_torch = torch.tensor(X_scaled, dtype=self.dtype, device=self.device)
        
        # Predict
        self.model.eval()  # type: ignore
        with torch.no_grad():
            posterior = self.model.posterior(X_torch)  # type: ignore
            y_pred_scaled = posterior.mean.cpu().numpy().ravel()
            
            if return_std:
                # Standard deviation
                y_std_scaled = posterior.variance.sqrt().cpu().numpy().ravel()
            else:
                # Variance
                y_std_scaled = posterior.variance.cpu().numpy().ravel()
        
        # Inverse transform
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Scale uncertainty (approximate)
        # Note: This is an approximation because we're scaling variance/std
        y_scale = self.y_scaler.scale_[0]
        y_unc = y_std_scaled * y_scale
        
        return y_pred, y_unc
    
    def get_acquisition_values(
        self,
        X: np.ndarray,
        acquisition: str = 'ucb',
        beta: float = 2.0
    ) -> np.ndarray:
        """
        Compute acquisition function values.
        
        Args:
            X: Feature matrix (N, D)
            acquisition: 'ucb' (Upper Confidence Bound) or 'std' (Max Std Dev)
            beta: UCB trade-off parameter (higher = more exploration)
        
        Returns:
            Acquisition values (N,)
        """
        y_pred, y_std = self.predict_with_uncertainty(X, return_std=True)
        
        if acquisition == 'ucb':
            # Upper Confidence Bound: mean + beta * std
            acq_values = y_pred + beta * y_std
        elif acquisition == 'std':
            # Maximum uncertainty: just std
            acq_values = y_std
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")
        
        return acq_values
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Update GP with new data points (incremental learning).
        
        Args:
            X_new: New feature matrix (N_new, D)
            y_new: New target values (N_new,)
        """
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Append to training data
        X_combined = np.vstack([self.X_train_np, X_new])
        y_combined = np.hstack([self.y_train_np, y_new])
        
        # Refit
        self.fit(X_combined, y_combined)
    
    def save(self, path: Path):
        """Save GP model and scalers."""
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save state dict
        state = {
            'model_state': self.model.state_dict() if self.model else None,
            'X_scaler': self.X_scaler,
            'y_scaler': self.y_scaler,
            'X_train_np': self.X_train_np,
            'y_train_np': self.y_train_np,
            'fitted': self.fitted_,
            'device': self.device,
            'dtype': str(self.dtype),
            'random_state': self.random_state
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"💾 GP model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "GPModel":
        """Load GP model from file."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Reconstruct model
        gp = cls(
            device=state['device'],
            dtype=getattr(torch, state['dtype'].split('.')[-1]),
            random_state=state['random_state']
        )
        
        gp.X_scaler = state['X_scaler']
        gp.y_scaler = state['y_scaler']
        gp.X_train_np = state['X_train_np']
        gp.y_train_np = state['y_train_np']
        gp.fitted_ = state['fitted']
        
        # Refit model if fitted
        if gp.fitted_ and gp.X_train_np is not None:
            gp.fit(gp.X_train_np, gp.y_train_np)
        
        logger.info(f"📂 GP model loaded from {path}")
        return gp


if __name__ == '__main__':
    # Test GP model
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.rand(50, 10)
    y_train = X_train[:, 0] * 10 + X_train[:, 1] * 5 + np.random.randn(50) * 0.5
    
    X_test = np.random.rand(10, 10)
    
    # Fit GP
    gp = GPModel()
    gp.fit(X_train, y_train)
    
    # Predict
    y_pred = gp.predict(X_test)
    y_pred_unc, y_std = gp.predict_with_uncertainty(X_test)
    
    print(f"\nPredictions: {y_pred[:5]}")
    print(f"Uncertainty: {y_std[:5]}")
    print(f"UCB values:  {gp.get_acquisition_values(X_test, 'ucb')[:5]}")

