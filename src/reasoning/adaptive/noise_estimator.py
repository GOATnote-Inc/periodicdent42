"""
Noise estimation from pilot experiments - EXPERIMENTAL.

Attempts to estimate measurement noise standard deviation from a small
number of pilot experiments. This is used to decide between optimization
algorithms.

## Scientific Caution:
- Noise estimation from small samples is inherently uncertain
- Requires sufficient data for reliable estimation (typically n≥5 replicates)
- Assumes noise is stationary (constant across parameter space)
- May fail with heteroscedastic noise (variance changes with input)

## Known Limitations:
- Small sample bias (underestimates variance)
- Assumes Gaussian noise (may not hold for all systems)
- No confidence intervals yet (TODO)
- Not validated on real hardware
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NoiseEstimate:
    """
    Container for noise estimation results with uncertainty.
    
    Attributes:
        std: Estimated standard deviation
        confidence_interval: 95% CI (lower, upper) if available
        method: How the estimate was computed
        sample_size: Number of observations used
        reliable: Whether we have enough data for reliable estimation
    """
    std: float
    confidence_interval: Optional[Tuple[float, float]] = None
    method: str = "unknown"
    sample_size: int = 0
    reliable: bool = False
    
    def __str__(self) -> str:
        if self.confidence_interval:
            ci_low, ci_high = self.confidence_interval
            return f"σ={self.std:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}]), n={self.sample_size}, method={self.method}"
        return f"σ={self.std:.3f} (n={self.sample_size}, method={self.method})"


class NoiseEstimator:
    """
    Estimate measurement noise from experimental data.
    
    ## Approach:
    We try multiple estimation methods and report which was used:
    
    1. **Replicate-based** (most reliable):
       - If we have repeated measurements at same parameters
       - Direct std() calculation
       - Requires: n≥3 replicates per condition
    
    2. **Residual-based** (model-dependent):
       - Fit a model (GP, polynomial) to data
       - Estimate noise from residuals
       - Requires: n≥10 total points, well-specified model
    
    3. **Sequential difference** (least reliable):
       - Estimate from differences between adjacent experiments
       - Assumes smooth objective function
       - Requires: n≥5 sequential points
    
    ## Confidence Intervals:
    - Uses bootstrap or analytical CI when possible
    - Reports "not reliable" if sample size too small
    """
    
    def __init__(self, min_reliable_samples: int = 5):
        """
        Initialize noise estimator.
        
        Args:
            min_reliable_samples: Minimum samples for "reliable" flag
        """
        self.min_reliable_samples = min_reliable_samples
    
    def estimate_from_replicates(
        self,
        replicate_groups: List[List[float]]
    ) -> NoiseEstimate:
        """
        Estimate noise from replicated measurements.
        
        Most reliable method when available.
        
        Args:
            replicate_groups: List of replicate measurements
                Example: [[1.1, 1.2, 1.0], [2.1, 2.3, 2.0]]
                         Two conditions, 3 replicates each
        
        Returns:
            NoiseEstimate with pooled standard deviation
        """
        if not replicate_groups or not any(replicate_groups):
            logger.warning("No replicate data provided")
            return NoiseEstimate(std=0.0, method="none", reliable=False)
        
        # Filter groups with enough replicates
        valid_groups = [g for g in replicate_groups if len(g) >= 2]
        
        if not valid_groups:
            logger.warning("No groups with ≥2 replicates")
            return NoiseEstimate(std=0.0, method="insufficient_replicates", reliable=False)
        
        # Compute standard deviation for each group
        group_stds = [np.std(group, ddof=1) for group in valid_groups]
        
        # Pooled standard deviation (assumes equal variance)
        # This is more stable than taking mean of stds
        all_values = []
        group_means = []
        for group in valid_groups:
            group_means.append(np.mean(group))
            all_values.extend(group)
        
        # Calculate pooled variance
        total_ss = 0
        total_df = 0
        for group in valid_groups:
            group_mean = np.mean(group)
            ss = np.sum((np.array(group) - group_mean) ** 2)
            df = len(group) - 1
            total_ss += ss
            total_df += df
        
        if total_df > 0:
            pooled_variance = total_ss / total_df
            pooled_std = np.sqrt(pooled_variance)
        else:
            pooled_std = np.mean(group_stds)
        
        # Confidence interval using chi-squared distribution
        # (assumes normality - may not hold!)
        from scipy import stats
        ci_low = np.sqrt(total_ss / stats.chi2.ppf(0.975, total_df))
        ci_high = np.sqrt(total_ss / stats.chi2.ppf(0.025, total_df))
        
        sample_size = sum(len(g) for g in valid_groups)
        reliable = sample_size >= self.min_reliable_samples
        
        logger.info(
            f"Replicate-based noise estimate: {pooled_std:.3f} "
            f"from {len(valid_groups)} groups, {sample_size} total measurements"
        )
        
        return NoiseEstimate(
            std=pooled_std,
            confidence_interval=(ci_low, ci_high),
            method="replicate_pooled",
            sample_size=sample_size,
            reliable=reliable
        )
    
    def estimate_from_sequential(
        self,
        values: List[float],
        assume_smooth: bool = True
    ) -> NoiseEstimate:
        """
        Estimate noise from sequential experiments.
        
        CAUTION: This is the LEAST reliable method.
        - Assumes objective function is smooth
        - Differences include both noise AND function gradient
        - Will overestimate if function changes rapidly
        
        Args:
            values: Sequential observations
            assume_smooth: If False, warns about potential bias
        
        Returns:
            NoiseEstimate (marked as less reliable)
        """
        if len(values) < 3:
            logger.warning("Need ≥3 points for sequential estimation")
            return NoiseEstimate(
                std=0.0,
                method="insufficient_data",
                sample_size=len(values),
                reliable=False
            )
        
        # First differences
        diffs = np.diff(values)
        
        # Noise std ≈ std(differences) / sqrt(2)
        # (if function is flat, differences are pure noise)
        noise_std = np.std(diffs, ddof=1) / np.sqrt(2)
        
        # WARNING: This will be biased if function has gradient
        if not assume_smooth:
            logger.warning(
                "Sequential estimation may be biased by function gradient. "
                "Consider using replicates or residual-based estimation."
            )
        
        # Very rough CI using bootstrap
        from scipy import stats
        n = len(diffs)
        se = noise_std / np.sqrt(n)  # Standard error (rough approximation)
        ci_low = max(0, noise_std - 1.96 * se)
        ci_high = noise_std + 1.96 * se
        
        reliable = len(values) >= self.min_reliable_samples and assume_smooth
        
        logger.info(
            f"Sequential noise estimate: {noise_std:.3f} from {len(values)} points "
            f"{'(WARNING: assumes smooth function)' if not reliable else ''}"
        )
        
        return NoiseEstimate(
            std=noise_std,
            confidence_interval=(ci_low, ci_high),
            method="sequential_differences",
            sample_size=len(values),
            reliable=reliable
        )
    
    def estimate_from_residuals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "gp"
    ) -> NoiseEstimate:
        """
        Estimate noise from model residuals.
        
        Fits a model (GP or polynomial) and estimates noise from
        residuals. More reliable than sequential differences.
        
        Args:
            X: Input parameters (n_samples, n_features)
            y: Observed values (n_samples,)
            model_type: "gp" (Gaussian Process) or "polynomial"
        
        Returns:
            NoiseEstimate from residuals
        """
        if len(y) < 10:
            logger.warning("Need ≥10 points for reliable residual estimation")
            return NoiseEstimate(
                std=0.0,
                method="insufficient_data_for_model",
                sample_size=len(y),
                reliable=False
            )
        
        try:
            if model_type == "gp":
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, WhiteKernel
                
                # Fit GP with explicit noise kernel
                kernel = RBF() + WhiteKernel(noise_level=0.1)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=5,
                    normalize_y=True
                )
                gp.fit(X, y)
                
                # Extract noise from WhiteKernel
                noise_std = np.sqrt(gp.kernel_.k2.noise_level)
                
                logger.info(f"GP-based noise estimate: {noise_std:.3f}")
                
                return NoiseEstimate(
                    std=noise_std,
                    method="gp_white_kernel",
                    sample_size=len(y),
                    reliable=len(y) >= self.min_reliable_samples
                )
            
            elif model_type == "polynomial":
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression
                
                # Fit polynomial (degree 2)
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                
                model = LinearRegression()
                model.fit(X_poly, y)
                
                # Residuals
                y_pred = model.predict(X_poly)
                residuals = y - y_pred
                noise_std = np.std(residuals, ddof=len(model.coef_))
                
                logger.info(f"Polynomial residual noise estimate: {noise_std:.3f}")
                
                return NoiseEstimate(
                    std=noise_std,
                    method="polynomial_residuals",
                    sample_size=len(y),
                    reliable=len(y) >= self.min_reliable_samples
                )
            
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        
        except Exception as e:
            logger.error(f"Failed to estimate noise from residuals: {e}")
            return NoiseEstimate(
                std=0.0,
                method=f"failed_{model_type}",
                sample_size=len(y),
                reliable=False
            )
    
    def estimate(
        self,
        data: Dict[str, any],
        prefer_method: str = "auto"
    ) -> NoiseEstimate:
        """
        Automatically estimate noise using best available method.
        
        Args:
            data: Dictionary with one of:
                - "replicates": List[List[float]] - replicated measurements
                - "sequential": List[float] - sequential observations
                - "X" and "y": Arrays for model-based estimation
            prefer_method: "auto", "replicates", "sequential", or "residuals"
        
        Returns:
            NoiseEstimate using best available method
        """
        # Try methods in order of reliability
        if prefer_method == "auto":
            # 1. Replicates (best)
            if "replicates" in data:
                estimate = self.estimate_from_replicates(data["replicates"])
                if estimate.reliable:
                    return estimate
            
            # 2. Residuals (good)
            if "X" in data and "y" in data:
                estimate = self.estimate_from_residuals(
                    data["X"],
                    data["y"]
                )
                if estimate.reliable:
                    return estimate
            
            # 3. Sequential (last resort)
            if "sequential" in data:
                return self.estimate_from_sequential(data["sequential"])
            
            # Nothing worked
            logger.error("No valid data provided for noise estimation")
            return NoiseEstimate(std=0.0, method="no_data", reliable=False)
        
        elif prefer_method == "replicates":
            return self.estimate_from_replicates(data.get("replicates", []))
        
        elif prefer_method == "sequential":
            return self.estimate_from_sequential(data.get("sequential", []))
        
        elif prefer_method == "residuals":
            return self.estimate_from_residuals(
                data.get("X"),
                data.get("y")
            )
        
        else:
            raise ValueError(f"Unknown method: {prefer_method}")


def estimate_noise_simple(observations: List[float]) -> float:
    """
    Quick and dirty noise estimation for prototyping.
    
    CAUTION: Uses sequential differences - may be biased!
    For production, use NoiseEstimator class with proper validation.
    
    Args:
        observations: Sequential measurements
    
    Returns:
        Estimated noise standard deviation
    """
    if len(observations) < 3:
        return 0.0
    
    diffs = np.diff(observations)
    return np.std(diffs) / np.sqrt(2)

