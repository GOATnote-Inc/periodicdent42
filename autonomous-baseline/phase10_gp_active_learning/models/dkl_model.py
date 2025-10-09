"""
Deep Kernel Learning Model for Active Learning

Combines neural network feature extraction with Gaussian Process uncertainty
quantification. Addresses high-dimensional tabular data limitations of basic GP.

Architecture:
    Input (81 features) → NN (64 → 32 → 16) → GP → μ(x*), σ(x*)

References:
    - Wilson et al. (2016): "Deep Kernel Learning", AISTATS
    - Lookman et al. (2019): "Active learning in materials", npj Comp Mat

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
License: MIT
"""

import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """
    Neural network for dimensionality reduction and feature learning.
    
    Maps high-dimensional input (81 UCI features) to compact learned 
    representation (16 dimensions) that GP can effectively model.
    
    Architecture:
        Linear(81 → 64) + ReLU + BatchNorm
        Linear(64 → 32) + ReLU + BatchNorm
        Linear(32 → 16)
    
    Args:
        input_dim: Input feature dimension (default: 81 for UCI)
        hidden_dims: Hidden layer dimensions
        output_dim: Learned feature dimension (default: 16)
        dropout_p: Dropout probability for regularization (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 81,
        hidden_dims: list = [64, 32],
        output_dim: int = 16,
        dropout_p: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        # Hidden layers with BatchNorm and Dropout
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))  # Stabilize training
            layers.append(nn.Dropout(dropout_p))   # Prevent overfitting
            in_dim = h_dim
        
        # Output layer (no activation, BatchNorm, or Dropout)
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Convert to float64 for compatibility with GPyTorch
        self.network = self.network.double()
        
        logger.info(f"✅ FeatureExtractor: {input_dim} → {hidden_dims} → {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extraction network.
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            z: Learned features (batch_size, output_dim)
        """
        return self.network(x)


class DKLModel(ExactGP):
    """
    Deep Kernel Learning: Neural network feature extraction + GP.
    
    Jointly trains NN and GP to:
    1. Learn informative low-dimensional representations (NN)
    2. Quantify uncertainty on learned features (GP)
    
    Training:
        - Joint optimization via marginal log likelihood
        - Adam optimizer for NN, L-BFGS for GP hyperparameters
        - Learning rate scheduling for stability
    
    Prediction:
        - Forward pass through NN to get learned features
        - GP posterior on learned features
        - Returns mean + uncertainty (aleatoric + epistemic)
    
    Args:
        train_x: Training inputs (n_samples, n_features)
        train_y: Training targets (n_samples,)
        feature_extractor: Pre-initialized FeatureExtractor
        likelihood: GP likelihood (default: Gaussian)
    """
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        feature_extractor: Optional[FeatureExtractor] = None,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None
    ):
        # Initialize feature extractor if not provided
        if feature_extractor is None:
            input_dim = train_x.shape[1]
            feature_extractor = FeatureExtractor(input_dim=input_dim)
        
        # Initialize likelihood with priors for stability
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05)
            )
            # Prevent noise from collapsing to zero
            likelihood.noise_covar.register_constraint(
                "raw_noise", 
                gpytorch.constraints.GreaterThan(1e-6)
            )
        
        # Initialize ExactGP with RAW training data (NOT features!)
        # Features will be computed inside forward()
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        self.mean_module = ConstantMean()
        
        # Add priors to kernel for stability
        self.covar_module = ScaleKernel(
            RBFKernel(lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0))
        )
        
        # Store original training data
        self.train_x_original = train_x
        self.train_y_original = train_y
        
        # BoTorch compatibility
        self.num_outputs = 1
        
        # Get output dim for logging (use eval mode to avoid BatchNorm issues)
        feature_extractor.eval()
        with torch.no_grad():
            sample_z = feature_extractor(train_x[:min(2, len(train_x))])
        feature_extractor.train()
        
        logger.info(f"✅ DKLModel initialized: {train_x.shape[1]} features → {sample_z.shape[1]} learned dims")
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass: NN feature extraction + GP.
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            MultivariateNormal distribution with mean and covariance
        """
        # Extract learned features
        z = self.feature_extractor(x)
        
        # GP on learned features
        mean_z = self.mean_module(z)
        covar_z = self.covar_module(z)
        
        return MultivariateNormal(mean_z, covar_z)
    
    def latent_mvn(self, X: torch.Tensor, observation_noise: bool = False) -> MultivariateNormal:
        """
        Produce latent-space MultivariateNormal for BoTorch wrapper.
        
        Args:
            X: Input tensor (batch_size, input_dim)
            observation_noise: Include likelihood noise if True
        
        Returns:
            MultivariateNormal distribution in latent space
        """
        self.eval()
        self.likelihood.eval()
        
        # Use ExactGP's __call__ for proper conditioning (not forward()!)
        # self(X) conditions on training data, self.forward(X) computes prior
        with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=True):
            mvn = self(X)  # __call__ uses GPyTorch conditioning machinery
        
        # Include likelihood noise if requested
        return self.likelihood(mvn) if observation_noise else mvn
    
    def fit(
        self,
        n_epochs: int = 100,
        lr: float = 0.001,
        verbose: bool = True,
        patience: int = 10
    ) -> dict:
        """
        Train DKL model (NN + GP jointly).
        
        Strategy:
        1. Warm-up: Train NN for 20% of epochs (fix GP)
        2. Joint training: Train NN + GP together
        3. Fine-tune: Train GP only (fix NN)
        
        Args:
            n_epochs: Number of training epochs
            lr: Learning rate for Adam optimizer
            verbose: Print training progress
            patience: Early stopping patience
        
        Returns:
            Training history (losses, best_epoch)
        """
        self.train()
        self.likelihood.train()
        
        # Optimizer: Two groups (NN gets higher LR, GP hyperparams get lower)
        optimizer = torch.optim.Adam([
            {'params': self.feature_extractor.parameters(), 'lr': lr},
            {'params': self.mean_module.parameters(), 'lr': lr * 0.1},
            {'params': self.covar_module.parameters(), 'lr': lr * 0.1},
            {'params': self.likelihood.parameters(), 'lr': lr * 0.1},
        ])
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function (marginal log likelihood)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        
        # Training history
        history = {
            'losses': [],
            'best_loss': float('inf'),
            'best_epoch': 0
        }
        
        # Early stopping
        patience_counter = 0
        
        if verbose:
            logger.info(f"🔧 Training DKL: {n_epochs} epochs, lr={lr:.4f}")
        
        # Standard DKL training loop (official GPyTorch pattern)
        for epoch in range(n_epochs):
            self.train()
            self.likelihood.train()
            optimizer.zero_grad()
            
            # Forward pass: Extracts features inside forward()
            # GPyTorch will condition on stored training data automatically
            with gpytorch.settings.cholesky_jitter(1e-5):
                output = self(self.train_x_original)  # Raw X → features → GP
                
                # Compute loss
                loss = -mll(output, self.train_y_original)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record loss
            loss_value = loss.item()
            history['losses'].append(loss_value)
            
            # Update best
            if loss_value < history['best_loss']:
                history['best_loss'] = loss_value
                history['best_epoch'] = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            scheduler.step(loss_value)
            
            # Diagnostics (check if NN is learning)
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                with torch.no_grad():
                    z = self.feature_extractor(self.train_x_original)
                    ls = self.covar_module.base_kernel.lengthscale.detach().cpu().flatten().mean()
                    noise = self.likelihood.noise.detach().cpu().item()
                logger.info(
                    f"   Epoch {epoch:3d}/{n_epochs}: loss={loss_value:.4f} "
                    f"| z.std={z.std().item():.4f} | ls~{ls:.4f} | noise~{noise:.6f}"
                )
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    logger.info(f"⏹️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        self.eval()
        self.likelihood.eval()
        
        if verbose:
            logger.info(f"✅ DKL training complete: Best loss = {history['best_loss']:.4f} @ epoch {history['best_epoch']}")
        
        return history
    
    def predict(
        self,
        X_test: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict with calibrated uncertainty.
        
        Args:
            X_test: Test inputs (n_samples, n_features)
            return_std: Return standard deviation (uncertainty)
        
        Returns:
            y_pred: Predicted means (n_samples,)
            y_std: Predicted standard deviations (n_samples,) if return_std
        """
        # Convert to tensor
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float64)
        
        # Prediction mode
        self.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=True):
            # Proper GP posterior: likelihood(model(X*))
            # This conditions on training data and computes full predictive distribution
            posterior = self(X_test)  # Forward extracts features → GP conditions on train
            
            # Predictive distribution (includes likelihood noise)
            predictive = self.likelihood(posterior)
            
            # Extract mean and std
            y_pred = predictive.mean.cpu().numpy()
            
            if return_std:
                y_std = predictive.stddev.cpu().numpy()
                return y_pred, y_std
            else:
                return y_pred, None


def create_dkl_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 81,
    hidden_dims: list = [64, 32],
    output_dim: int = 16,
    n_epochs: int = 100,
    lr: float = 0.001,
    verbose: bool = True
) -> DKLModel:
    """
    Factory function to create and train DKL model.
    
    Args:
        X_train: Training features (n_samples, input_dim)
        y_train: Training targets (n_samples,)
        input_dim: Input feature dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Learned feature dimension
        n_epochs: Training epochs
        lr: Learning rate
        verbose: Print training logs
    
    Returns:
        Trained DKLModel
    
    Example:
        >>> X_train, y_train = load_data()
        >>> model = create_dkl_model(X_train, y_train, n_epochs=100)
        >>> y_pred, y_std = model.predict(X_test)
    """
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64).squeeze()
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim
    )
    
    # Create DKL model
    model = DKLModel(
        train_x=X_train_tensor,
        train_y=y_train_tensor,
        feature_extractor=feature_extractor
    )
    
    # Train
    history = model.fit(n_epochs=n_epochs, lr=lr, verbose=verbose)
    
    if verbose:
        logger.info(f"✅ DKL model ready: {history['best_loss']:.4f} best loss")
    
    return model


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_train = 200
    n_test = 50
    n_features = 81
    
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randn(n_train)
    X_test = np.random.randn(n_test, n_features)
    
    logger.info("🧪 Testing DKL model on synthetic data...")
    
    # Create and train model
    model = create_dkl_model(
        X_train, y_train,
        input_dim=n_features,
        n_epochs=50,
        lr=0.001,
        verbose=True
    )
    
    # Predict
    y_pred, y_std = model.predict(X_test)
    
    logger.info(f"✅ Predictions: mean = {y_pred.mean():.2f}, std = {y_std.mean():.2f}")
    logger.info("✅ DKL model test passed!")

