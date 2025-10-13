# cudadent42/bench/common/stats.py
"""
Publication-Grade Statistical Analysis Module

Provides:
- Bootstrap confidence intervals
- Effect sizes (Hedges' g, Cliff's Delta)
- Distribution comparison tests
- Robust statistics

Usage:
    from cudadent42.bench.common.stats import bootstrap_ci, hedges_g, compare_distributions
    
    ci_lower, ci_upper = bootstrap_ci(data, confidence=0.95)
    effect = hedges_g(group1, group2)
    result = compare_distributions(baseline, candidate)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.median,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: 1D array of samples
        statistic: Function to compute (default: median)
        confidence: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility
    
    Returns:
        (lower_bound, upper_bound) tuple
    
    Example:
        >>> data = np.array([1.2, 1.3, 1.1, 1.4, 1.2])
        >>> lower, upper = bootstrap_ci(data, confidence=0.95)
    """
    rng = np.random.RandomState(seed)
    
    bootstrap_stats = np.zeros(n_bootstrap)
    n = len(data)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(resample)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return ci_lower, ci_upper


def ci_non_overlap(a_ci: Tuple[float, float], b_ci: Tuple[float, float]) -> bool:
    """
    Check if two confidence intervals do not overlap.
    
    Args:
        a_ci: (lower, upper) for first interval
        b_ci: (lower, upper) for second interval
    
    Returns:
        True if CIs do not overlap, False otherwise
    
    Example:
        >>> a_ci = (1.0, 1.5)
        >>> b_ci = (1.6, 2.0)
        >>> ci_non_overlap(a_ci, b_ci)  # True
    """
    a_lower, a_upper = a_ci
    b_lower, b_upper = b_ci
    
    return (a_upper < b_lower) or (b_upper < a_lower)


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Hedges' g effect size (bias-corrected Cohen's d).
    
    Hedges' g is preferred over Cohen's d for small sample sizes as it
    applies a correction factor that reduces bias.
    
    Interpretation:
        |g| < 0.2:  Small effect
        |g| < 0.5:  Medium effect
        |g| >= 0.8: Large effect
    
    Args:
        group1: First group samples
        group2: Second group samples
    
    Returns:
        Hedges' g value (positive means group1 > group2)
    
    Example:
        >>> baseline = np.array([1.0, 1.1, 1.2])
        >>> candidate = np.array([0.8, 0.9, 0.85])
        >>> g = hedges_g(baseline, candidate)
    """
    n1 = len(group1)
    n2 = len(group2)
    
    # Calculate pooled standard deviation
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    mean_diff = np.mean(group1) - np.mean(group2)
    if pooled_std == 0:
        return 0.0
    cohens_d = mean_diff / pooled_std
    
    # Apply Hedges' correction factor
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g_value = cohens_d * correction
    
    return hedges_g_value


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cliff's Delta non-parametric effect size.
    
    Cliff's Delta measures the probability that a random value from group1
    is greater than a random value from group2.
    
    Interpretation:
        |δ| < 0.147: Negligible
        |δ| < 0.33:  Small
        |δ| < 0.474: Medium
        |δ| >= 0.474: Large
    
    Args:
        group1: First group samples
        group2: Second group samples
    
    Returns:
        Cliff's Delta value in range [-1, 1]
        Positive: group1 tends to be larger than group2
        Negative: group1 tends to be smaller than group2
    
    Example:
        >>> baseline = np.array([1.0, 1.1, 1.2])
        >>> candidate = np.array([0.8, 0.9, 0.85])
        >>> delta = cliffs_delta(baseline, candidate)
    """
    n1 = len(group1)
    n2 = len(group2)
    
    # Count how many times group1[i] > group2[j]
    greater = 0
    less = 0
    
    for val1 in group1:
        for val2 in group2:
            if val1 > val2:
                greater += 1
            elif val1 < val2:
                less += 1
            # Ties don't count
    
    delta = (greater - less) / (n1 * n2)
    
    return delta


def compare_distributions(
    baseline: np.ndarray,
    candidate: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Comprehensive comparison of two distributions.
    
    Provides:
    - Descriptive statistics (median, mean, std)
    - Bootstrap confidence intervals
    - Effect sizes (Hedges' g, Cliff's Delta)
    - CI overlap check
    
    Args:
        baseline: Baseline measurements
        candidate: Candidate measurements
        confidence: Confidence level for CIs
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed
    
    Returns:
        Dict with complete analysis results
    
    Example:
        >>> baseline = np.array([1.0, 1.1, 1.2, 1.15, 1.05])
        >>> candidate = np.array([0.8, 0.9, 0.85, 0.88, 0.82])
        >>> result = compare_distributions(baseline, candidate)
        >>> print(f"Speedup: {result['speedup']:.2f}x")
        >>> print(f"CIs overlap: {result['ci_overlap']}")
    """
    # Descriptive statistics
    baseline_median = np.median(baseline)
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline, ddof=1)
    
    candidate_median = np.median(candidate)
    candidate_mean = np.mean(candidate)
    candidate_std = np.std(candidate, ddof=1)
    
    # Bootstrap CIs
    baseline_ci = bootstrap_ci(baseline, confidence=confidence, 
                              n_bootstrap=n_bootstrap, seed=seed)
    candidate_ci = bootstrap_ci(candidate, confidence=confidence,
                               n_bootstrap=n_bootstrap, seed=seed)
    
    # Effect sizes
    hedges_g_value = hedges_g(baseline, candidate)
    cliffs_delta_value = cliffs_delta(baseline, candidate)
    
    # CI overlap
    overlap = not ci_non_overlap(baseline_ci, candidate_ci)
    
    # Speedup (assuming lower is better, e.g., latency)
    speedup = baseline_median / candidate_median if candidate_median > 0 else float('inf')
    improvement_pct = ((baseline_median - candidate_median) / baseline_median) * 100
    
    return {
        'baseline': {
            'median': baseline_median,
            'mean': baseline_mean,
            'std': baseline_std,
            'ci_lower': baseline_ci[0],
            'ci_upper': baseline_ci[1],
            'n': len(baseline)
        },
        'candidate': {
            'median': candidate_median,
            'mean': candidate_mean,
            'std': candidate_std,
            'ci_lower': candidate_ci[0],
            'ci_upper': candidate_ci[1],
            'n': len(candidate)
        },
        'comparison': {
            'speedup': speedup,
            'improvement_pct': improvement_pct,
            'hedges_g': hedges_g_value,
            'cliffs_delta': cliffs_delta_value,
            'ci_overlap': overlap,
            'significant': not overlap
        }
    }


if __name__ == "__main__":
    # Test module
    print("Testing statistics module...\n")
    
    # Simulate baseline and candidate data
    np.random.seed(42)
    baseline = np.random.normal(1.0, 0.1, 100)  # Slower
    candidate = np.random.normal(0.85, 0.08, 100)  # Faster
    
    result = compare_distributions(baseline, candidate, seed=42)
    
    print("Baseline:")
    print(f"  Median: {result['baseline']['median']:.4f} ms")
    print(f"  95% CI: [{result['baseline']['ci_lower']:.4f}, {result['baseline']['ci_upper']:.4f}]")
    
    print("\nCandidate:")
    print(f"  Median: {result['candidate']['median']:.4f} ms")
    print(f"  95% CI: [{result['candidate']['ci_lower']:.4f}, {result['candidate']['ci_upper']:.4f}]")
    
    print("\nComparison:")
    print(f"  Speedup: {result['comparison']['speedup']:.2f}x")
    print(f"  Improvement: {result['comparison']['improvement_pct']:.1f}%")
    print(f"  Hedges' g: {result['comparison']['hedges_g']:.3f}")
    print(f"  Cliff's Δ: {result['comparison']['cliffs_delta']:.3f}")
    print(f"  CIs overlap: {result['comparison']['ci_overlap']}")
    print(f"  Significant: {result['comparison']['significant']}")
    
    print("\n✓ Module test complete")

