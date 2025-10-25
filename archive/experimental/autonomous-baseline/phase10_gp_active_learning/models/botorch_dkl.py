"""
BoTorch-Compatible Deep Kernel Learning Wrapper

Wraps our DKL ExactGP model to conform to BoTorch's Model API,
enabling use with analytic acquisition functions like ExpectedImprovement.

References:
- https://botorch.org/docs/models
- https://botorch.readthedocs.io/en/latest/models.html#gpytorchmodel
- https://docs.gpytorch.ai/en/latest/examples/08_Deep_Kernel_Learning/

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
License: MIT
"""

import torch
from typing import Optional, List, Union
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
import logging

logger = logging.getLogger(__name__)


class BoTorchDKL(GPyTorchModel):
    """
    BoTorch wrapper for Deep Kernel Learning.
    
    Conforms to BoTorch Model API:
    - Implements posterior() returning GPyTorchPosterior
    - Supports observation_noise parameter
    - Supports posterior_transform (for affine transforms)
    - Single-outcome (num_outputs=1) for analytic EI
    
    Usage:
        # Train DKL model
        dkl = DKLModel(train_x, train_y, feature_extractor)
        dkl.fit(n_epochs=100)
        
        # Wrap for BoTorch
        model = BoTorchDKL(dkl)
        
        # Use with analytic EI
        from botorch.acquisition import ExpectedImprovement
        acq = ExpectedImprovement(model=model, best_f=best_f)
        ei_values = acq(X_candidates)
    """
    
    _num_outputs = 1  # Single-outcome for analytic EI
    
    def __init__(
        self,
        dkl_exactgp,
        outcome_transform=None,
        input_transform=None
    ):
        """
        Initialize BoTorch DKL wrapper.
        
        Args:
            dkl_exactgp: Trained DKLModel (ExactGP with feature_extractor)
            outcome_transform: Optional BoTorch outcome transform
            input_transform: Optional BoTorch input transform
        """
        super().__init__()
        self.model = dkl_exactgp
        self.likelihood = dkl_exactgp.likelihood
        self.outcome_transform = outcome_transform
        self.input_transform = input_transform
        
        logger.info(f"✅ BoTorchDKL wrapper initialized (num_outputs={self._num_outputs})")
    
    @property
    def num_outputs(self) -> int:
        """Number of outputs (always 1 for single-task)."""
        return self._num_outputs
    
    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, torch.Tensor] = False,
        posterior_transform=None,
        **kwargs,
    ) -> GPyTorchPosterior:
        """
        Compute posterior distribution at test points X.
        
        This is the key method for BoTorch compatibility. It must:
        1. Handle X in (batch, q, d) format (BoTorch convention)
        2. Return GPyTorchPosterior wrapping MultivariateNormal
        3. Support observation_noise parameter
        4. Support posterior_transform (affine transforms)
        
        Args:
            X: Test points, shape (batch, q, d) or (batch, d)
               For analytic EI, q must be 1
            output_indices: Not used (single-outcome model)
            observation_noise: Include likelihood noise if True
            posterior_transform: Optional posterior transform
            **kwargs: Additional arguments (ignored)
        
        Returns:
            GPyTorchPosterior with .mean and .variance
        """
        # BoTorch expects X as (batch, q, d). For analytic EI, q=1.
        if X.dim() == 2:  # (n, d) -> (n, 1, d)
            X = X.unsqueeze(1)
        
        # Verify q=1 for analytic EI
        assert X.size(-2) == 1, (
            f"Analytic EI requires q=1, got q={X.size(-2)}. "
            "Use qExpectedImprovement for batch acquisition."
        )
        
        # Remove q dimension for processing: (batch, 1, d) -> (batch, d)
        X_ = X.squeeze(-2)
        
        # Apply input transform if provided
        if self.input_transform is not None:
            X_ = self.input_transform(X_)
        
        # Get latent-space MultivariateNormal from DKL
        mvn: MultivariateNormal = self.model.latent_mvn(
            X_, 
            observation_noise=bool(observation_noise)
        )
        
        # Apply outcome transform if provided (must preserve Gaussianity)
        if self.outcome_transform is not None:
            mvn = self.outcome_transform.untransform_posterior(
                GPyTorchPosterior(mvn)
            ).mvn
        
        # Wrap in GPyTorchPosterior
        posterior = GPyTorchPosterior(mvn)
        
        # Apply posterior_transform if provided (e.g., scalarization)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        
        return posterior
    
    def condition_on_observations(self, X, Y, **kwargs):
        """
        Condition model on new observations (for fantasy updates).
        
        For simplicity, we retrain the model. In production, you could
        use GPyTorch's fantasy model updates for efficiency.
        
        Args:
            X: New observations (features)
            Y: New observations (targets)
            **kwargs: Additional arguments
        
        Returns:
            New BoTorchDKL with updated training data
        """
        # Concatenate new data with existing
        new_train_x = torch.cat([self.model.train_x_original, X])
        new_train_y = torch.cat([self.model.train_y_original, Y])
        
        # Create new DKL model with updated data
        from .dkl_model import DKLModel
        
        conditioned_dkl = DKLModel(
            train_x=new_train_x,
            train_y=new_train_y,
            feature_extractor=self.model.feature_extractor,
            likelihood=self.model.likelihood
        )
        
        # Quick retraining (fewer epochs for fantasies)
        conditioned_dkl.fit(n_epochs=10, verbose=False)
        
        # Wrap and return
        return BoTorchDKL(
            conditioned_dkl,
            self.outcome_transform,
            self.input_transform
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_train = 100
    n_features = 81
    
    train_x = torch.randn(n_train, n_features, dtype=torch.float64)
    train_y = torch.randn(n_train, dtype=torch.float64)
    
    logger.info("🧪 Testing BoTorchDKL wrapper...")
    
    # Create and train DKL
    from dkl_model import create_dkl_model
    dkl = create_dkl_model(
        train_x.numpy(), train_y.numpy(),
        input_dim=n_features,
        n_epochs=20,
        verbose=False
    )
    
    # Wrap for BoTorch
    model = BoTorchDKL(dkl)
    logger.info(f"   num_outputs: {model.num_outputs}")
    
    # Test posterior
    test_x = torch.randn(10, n_features, dtype=torch.float64)
    posterior = model.posterior(test_x)
    
    logger.info(f"   Posterior mean shape: {posterior.mean.shape}")
    logger.info(f"   Posterior variance shape: {posterior.variance.shape}")
    
    # Test with BoTorch EI
    from botorch.acquisition import ExpectedImprovement
    
    best_f = train_y.max().item()
    acq = ExpectedImprovement(model=model, best_f=best_f)
    
    # EI values (q=1 format)
    test_x_q1 = test_x.unsqueeze(1)  # (10, 1, 81)
    ei_values = acq(test_x_q1)
    
    logger.info(f"   EI values shape: {ei_values.shape}")
    logger.info(f"   EI range: [{ei_values.min():.4f}, {ei_values.max():.4f}]")
    logger.info("✅ BoTorchDKL test passed!")

