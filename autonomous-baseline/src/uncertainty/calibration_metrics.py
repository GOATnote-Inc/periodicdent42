"""Calibration metrics for uncertainty quantification.

Provides metrics to evaluate prediction interval coverage and calibration quality:
- PICP: Prediction Interval Coverage Probability
- MPIW: Mean Prediction Interval Width
- ECE: Expected Calibration Error
- Miscalibration Area: Area between calibration curve and diagonal
"""

import numpy as np


def prediction_interval_coverage_probability(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Calculate Prediction Interval Coverage Probability (PICP).
    
    PICP measures the fraction of targets that fall within the prediction intervals.
    For well-calibrated 95% intervals, PICP should be ≈ 0.95.
    
    Args:
        y_true: True target values (N,)
        y_lower: Lower bounds of prediction intervals (N,)
        y_upper: Upper bounds of prediction intervals (N,)
        
    Returns:
        PICP: Coverage probability ∈ [0, 1]
        
    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_lower = np.array([0.5, 1.5, 2.5])
        >>> y_upper = np.array([1.5, 2.5, 3.5])
        >>> picp = prediction_interval_coverage_probability(y_true, y_lower, y_upper)
        >>> picp
        1.0  # All targets within intervals
    """
    if len(y_true) != len(y_lower) or len(y_true) != len(y_upper):
        raise ValueError("Arrays must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays")
    
    # Check if target is within interval
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    
    return float(in_interval.mean())


def mean_prediction_interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Calculate Mean Prediction Interval Width (MPIW).
    
    MPIW measures the average width of prediction intervals.
    Narrower intervals are preferable (more informative) if coverage is maintained.
    
    Args:
        y_lower: Lower bounds of prediction intervals (N,)
        y_upper: Upper bounds of prediction intervals (N,)
        
    Returns:
        MPIW: Mean interval width
        
    Example:
        >>> y_lower = np.array([0.5, 1.5, 2.5])
        >>> y_upper = np.array([1.5, 2.5, 3.5])
        >>> mpiw = mean_prediction_interval_width(y_lower, y_upper)
        >>> mpiw
        1.0
    """
    if len(y_lower) != len(y_upper):
        raise ValueError("Arrays must have the same length")
    
    if len(y_lower) == 0:
        raise ValueError("Empty arrays")
    
    widths = y_upper - y_lower
    
    if np.any(widths < 0):
        raise ValueError("Upper bounds must be >= lower bounds")
    
    return float(widths.mean())


def coverage_width_criterion(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    eta: float = 50.0,
) -> float:
    """
    Calculate Coverage Width Criterion (CWC).
    
    CWC balances coverage and interval width. Penalizes intervals that are:
    - Too wide (uninformative)
    - Under-covering (coverage < nominal)
    
    Lower CWC is better.
    
    Args:
        y_true: True target values (N,)
        y_lower: Lower bounds of prediction intervals (N,)
        y_upper: Upper bounds of prediction intervals (N,)
        eta: Penalty coefficient for under-coverage (default: 50.0)
        
    Returns:
        CWC: Coverage-width criterion
        
    Reference:
        Khosravi et al. (2011) "Comprehensive Review of Neural Network-Based
        Prediction Intervals and New Advances"
    """
    picp = prediction_interval_coverage_probability(y_true, y_lower, y_upper)
    mpiw = mean_prediction_interval_width(y_lower, y_upper)
    
    # Normalize MPIW by target range
    y_range = y_true.max() - y_true.min()
    if y_range > 0:
        mpiw_normalized = mpiw / y_range
    else:
        mpiw_normalized = mpiw
    
    # Penalty for under-coverage
    # If PICP < 0.95 (for 95% intervals), apply exponential penalty
    nominal_coverage = 0.95  # Assume 95% intervals
    if picp < nominal_coverage:
        penalty = eta * (nominal_coverage - picp)
    else:
        penalty = 0.0
    
    cwc = mpiw_normalized * (1 + penalty)
    
    return float(cwc)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) for regression.
    
    ECE measures how well predicted uncertainties match empirical errors.
    For well-calibrated models:
    - If predicted std = σ, then ~68% of errors should be within [-σ, +σ]
    - If predicted std = 2σ, then ~95% of errors should be within [-2σ, +2σ]
    
    Args:
        y_true: True target values (N,)
        y_pred: Predicted values (N,)
        y_std: Predicted standard deviations (N,)
        n_bins: Number of bins for calibration curve (default: 10)
        
    Returns:
        ECE: Expected calibration error ∈ [0, 1]
        
    Note:
        Lower ECE is better. ECE ≤ 0.05 indicates good calibration.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(y_std):
        raise ValueError("Arrays must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays")
    
    if np.any(y_std <= 0):
        raise ValueError("Standard deviations must be positive")
    
    # Calculate normalized errors (z-scores)
    errors = y_true - y_pred
    z_scores = np.abs(errors / y_std)
    
    # Define confidence levels (e.g., 1σ, 2σ, 3σ)
    # Use quantiles of z-scores for adaptive binning
    bin_edges = np.linspace(0, z_scores.max(), n_bins + 1)
    
    ece = 0.0
    
    for i in range(n_bins):
        # Samples in this bin
        mask = (z_scores >= bin_edges[i]) & (z_scores < bin_edges[i + 1])
        
        if not mask.any():
            continue
        
        # Expected coverage for this z-score range
        # For normal distribution: P(|Z| < z) = erf(z / sqrt(2))
        from scipy.special import erf
        
        z_lower = bin_edges[i]
        z_upper = bin_edges[i + 1]
        z_mid = (z_lower + z_upper) / 2
        
        # Theoretical coverage for z-score < z_mid
        expected_coverage = erf(z_mid / np.sqrt(2))
        
        # Empirical coverage in this bin
        empirical_coverage = mask.sum() / len(y_true)
        
        # Contribution to ECE
        ece += np.abs(expected_coverage - empirical_coverage)
    
    return float(ece)


def miscalibration_area(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_points: int = 100,
) -> float:
    """
    Calculate Miscalibration Area (MA).
    
    MA is the area between the calibration curve and the diagonal.
    Calibration curve: (expected coverage, observed coverage) for various confidence levels.
    
    For perfect calibration, the curve lies on the diagonal (y = x).
    
    Args:
        y_true: True target values (N,)
        y_pred: Predicted values (N,)
        y_std: Predicted standard deviations (N,)
        n_points: Number of points for calibration curve (default: 100)
        
    Returns:
        MA: Miscalibration area ∈ [0, 1]
        
    Note:
        Lower MA is better. MA ≤ 0.05 indicates good calibration.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(y_std):
        raise ValueError("Arrays must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays")
    
    if np.any(y_std <= 0):
        raise ValueError("Standard deviations must be positive")
    
    # Calculate normalized errors (z-scores)
    errors = y_true - y_pred
    z_scores = np.abs(errors / y_std)
    
    # Confidence levels (0 to 3σ)
    confidence_levels = np.linspace(0, 3, n_points)
    
    miscal_area = 0.0
    
    for conf in confidence_levels:
        # Expected coverage for this confidence level
        from scipy.special import erf
        expected_coverage = erf(conf / np.sqrt(2))
        
        # Observed coverage (fraction of z-scores < conf)
        observed_coverage = (z_scores <= conf).mean()
        
        # Absolute difference
        miscal_area += np.abs(expected_coverage - observed_coverage)
    
    # Normalize by number of points and range
    miscal_area /= n_points
    
    return float(miscal_area)


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve for visualization.
    
    Returns (expected_coverage, observed_coverage) pairs for plotting.
    Perfect calibration → curve on diagonal (y = x).
    
    Args:
        y_true: True target values (N,)
        y_pred: Predicted values (N,)
        y_std: Predicted standard deviations (N,)
        n_points: Number of points for curve (default: 100)
        
    Returns:
        Tuple of (expected_coverage, observed_coverage) arrays
        
    Example:
        >>> y_true, y_pred, y_std = ...
        >>> exp, obs = calibration_curve(y_true, y_pred, y_std)
        >>> plt.plot(exp, obs, label='Model')
        >>> plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
        >>> plt.xlabel('Expected Coverage')
        >>> plt.ylabel('Observed Coverage')
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(y_std):
        raise ValueError("Arrays must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays")
    
    if np.any(y_std <= 0):
        raise ValueError("Standard deviations must be positive")
    
    # Calculate normalized errors
    errors = y_true - y_pred
    z_scores = np.abs(errors / y_std)
    
    # Confidence levels
    confidence_levels = np.linspace(0, 3, n_points)
    
    expected_coverage = []
    observed_coverage = []
    
    for conf in confidence_levels:
        # Expected coverage (from normal distribution)
        from scipy.special import erf
        exp_cov = erf(conf / np.sqrt(2))
        
        # Observed coverage
        obs_cov = (z_scores <= conf).mean()
        
        expected_coverage.append(exp_cov)
        observed_coverage.append(obs_cov)
    
    return np.array(expected_coverage), np.array(observed_coverage)


def interval_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Calculate Interval Score (Winkler Score).
    
    Proper scoring rule that rewards:
    - Narrow intervals (lower score = better)
    - High coverage (penalizes under-coverage)
    
    Args:
        y_true: True target values (N,)
        y_lower: Lower bounds of prediction intervals (N,)
        y_upper: Upper bounds of prediction intervals (N,)
        alpha: Significance level (default: 0.05 for 95% intervals)
        
    Returns:
        Interval score (lower is better)
        
    Reference:
        Gneiting & Raftery (2007) "Strictly Proper Scoring Rules, Prediction,
        and Estimation"
    """
    if len(y_true) != len(y_lower) or len(y_true) != len(y_upper):
        raise ValueError("Arrays must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays")
    
    # Interval width
    width = y_upper - y_lower
    
    # Penalties for under-coverage
    penalty_lower = (2 / alpha) * (y_lower - y_true) * (y_true < y_lower)
    penalty_upper = (2 / alpha) * (y_true - y_upper) * (y_true > y_upper)
    
    # Interval score per sample
    scores = width + penalty_lower + penalty_upper
    
    # Mean interval score
    return float(scores.mean())


def sharpness(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate sharpness (inverse of interval width).
    
    Sharpness measures how informative the intervals are.
    Higher sharpness = narrower intervals = more informative.
    
    Should only be compared if coverage is similar.
    
    Args:
        y_lower: Lower bounds of prediction intervals (N,)
        y_upper: Upper bounds of prediction intervals (N,)
        
    Returns:
        Sharpness (higher is better, subject to coverage constraint)
    """
    mpiw = mean_prediction_interval_width(y_lower, y_upper)
    
    if mpiw == 0:
        return float('inf')
    
    return float(1.0 / mpiw)

