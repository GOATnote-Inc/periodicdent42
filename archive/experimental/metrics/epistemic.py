"""Epistemic efficiency metrics for CI test selection.

Replaces heuristic model_uncertainty with calibrated posterior tracking
and information-theoretic efficiency metrics. Production-hardened with
numerical stability guarantees and edge case handling.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def bernoulli_entropy(p: float) -> float:
    """Compute Bernoulli entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
    
    Args:
        p: Probability (0 to 1)
        
    Returns:
        Entropy in bits (0 to 1)
        
    Notes:
        - Returns 0.0 for p=0 or p=1 (no uncertainty)
        - Returns 1.0 for p=0.5 (maximum uncertainty)
        - Numerically stable for p near 0 or 1
    """
    # Handle exact edge cases first (before clamping)
    if p == 0.0 or p == 1.0:
        return 0.0
    
    # Clamp to avoid log(0) for near-zero/near-one values
    p = max(1e-10, min(1 - 1e-10, p))
    
    if p <= 0 or p >= 1:
        return 0.0
    
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def compute_expected_information_gain(
    failure_prob: float,
    cost_usd: float
) -> float:
    """Compute Expected Information Gain (EIG) per unit cost.
    
    EIG = H(p) / cost, where H(p) is Bernoulli entropy.
    
    Args:
        failure_prob: Predicted failure probability (0 to 1)
        cost_usd: Test cost in USD
        
    Returns:
        EIG in bits/$
        
    Notes:
        - Returns 0.0 if cost <= 0 (avoid division by zero)
        - Clamps failure_prob to [0, 1]
    """
    # Avoid division by zero
    if cost_usd <= 0:
        return 0.0
    
    # Clamp failure probability
    failure_prob = max(0.0, min(1.0, failure_prob))
    
    entropy = bernoulli_entropy(failure_prob)
    return float(entropy / cost_usd)


def compute_detection_rate(
    selected_tests: List[Dict[str, Any]],
    all_tests: List[Dict[str, Any]]
) -> float:
    """Compute estimated failure detection rate.
    
    Detection rate = (sum of failure probs for selected) / (sum for all)
    
    Args:
        selected_tests: Tests selected for execution
        all_tests: All available tests
        
    Returns:
        Detection rate (0 to 1)
        
    Notes:
        - Returns 1.0 if total_failures_est == 0 (no failures to detect)
        - Uses model_uncertainty as failure probability estimate
    """
    selected_failures_est = sum(
        t.get("model_uncertainty", 0.1) for t in selected_tests
    )
    total_failures_est = sum(
        t.get("model_uncertainty", 0.1) for t in all_tests
    )
    
    if total_failures_est == 0:
        return 1.0
    
    return float(min(1.0, selected_failures_est / total_failures_est))


def compute_entropy_delta(
    initial_entropy: float,
    final_entropy: float
) -> float:
    """Compute information gained as entropy reduction.
    
    Delta = initial - final (positive means information gained)
    
    Args:
        initial_entropy: Entropy before test execution (bits)
        final_entropy: Entropy after test execution (bits)
        
    Returns:
        Entropy delta in bits
        
    Notes:
        - Positive delta = information gained
        - Negative delta = information lost (should not happen in practice)
    """
    return float(initial_entropy - final_entropy)


def compute_epistemic_efficiency(
    selected_tests: List[Dict[str, Any]],
    all_tests: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute comprehensive epistemic efficiency metrics.
    
    Provides 20+ metrics quantifying information gain, cost efficiency,
    and detection coverage. Used for experiment ledger and reporting.
    
    Args:
        selected_tests: Tests selected for execution
        all_tests: All available tests
        
    Returns:
        Dictionary of epistemic metrics
        
    Metrics:
        - tests_selected, tests_total, selection_fraction
        - bits_gained, bits_available, information_fraction
        - bits_per_dollar (selected, full, ratio)
        - bits_per_second (selected, full, ratio)
        - detection_rate
        - entropy metrics (before, after, delta)
        - resource savings (cost, time, percentages)
    """
    # Basic stats
    n_selected = len(selected_tests)
    n_total = len(all_tests)
    
    # Information gained
    selected_eig = sum(t.get("eig_bits", 0) for t in selected_tests)
    total_eig = sum(t.get("eig_bits", 0) for t in all_tests)
    
    # Cost and time
    selected_cost = sum(t.get("cost_usd", 0) for t in selected_tests)
    total_cost = sum(t.get("cost_usd", 0) for t in all_tests)
    selected_time = sum(t.get("duration_sec", 0) for t in selected_tests)
    total_time = sum(t.get("duration_sec", 0) for t in all_tests)
    
    # Efficiency ratios (avoid division by zero)
    bits_per_dollar_selected = selected_eig / max(selected_cost, 1e-6)
    bits_per_dollar_full = total_eig / max(total_cost, 1e-6)
    
    bits_per_second_selected = selected_eig / max(selected_time, 1e-6)
    bits_per_second_full = total_eig / max(total_time, 1e-6)
    
    # Detection rate
    detection_rate = compute_detection_rate(selected_tests, all_tests)
    
    # Entropy metrics (if available)
    entropy_before = sum(t.get("entropy_before", 0) for t in selected_tests)
    entropy_after = sum(t.get("entropy_after", 0) for t in selected_tests)
    entropy_delta = compute_entropy_delta(entropy_before, entropy_after) if entropy_before > 0 else None
    
    metrics = {
        # Selection stats
        "tests_selected": n_selected,
        "tests_total": n_total,
        "selection_fraction": n_selected / max(n_total, 1),
        
        # Information metrics
        "bits_gained": float(selected_eig),
        "bits_available": float(total_eig),
        "information_fraction": selected_eig / max(total_eig, 1e-6),
        
        # Efficiency metrics
        "bits_per_dollar_selected": float(bits_per_dollar_selected),
        "bits_per_dollar_full": float(bits_per_dollar_full),
        "cost_efficiency_ratio": bits_per_dollar_selected / max(bits_per_dollar_full, 1e-6),
        
        "bits_per_second_selected": float(bits_per_second_selected),
        "bits_per_second_full": float(bits_per_second_full),
        "time_efficiency_ratio": bits_per_second_selected / max(bits_per_second_full, 1e-6),
        
        # Detection
        "detection_rate": float(detection_rate),
        
        # Entropy (optional)
        "entropy_before": float(entropy_before) if entropy_before > 0 else None,
        "entropy_after": float(entropy_after) if entropy_after > 0 else None,
        "entropy_delta": float(entropy_delta) if entropy_delta is not None else None,
        
        # Resource savings
        "cost_saved_usd": float(total_cost - selected_cost),
        "time_saved_sec": float(total_time - selected_time),
        "cost_reduction_pct": ((total_cost - selected_cost) / max(total_cost, 1e-6)) * 100,
        "time_reduction_pct": ((total_time - selected_time) / max(total_time, 1e-6)) * 100,
    }
    
    return metrics


def enrich_tests_with_epistemic_features(
    tests: List[Dict[str, Any]],
    model_predictions: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Enrich tests with calibrated epistemic features.
    
    Replaces heuristic model_uncertainty with calibrated failure probabilities
    and computes EIG for each test.
    
    Args:
        tests: List of test dictionaries
        model_predictions: Dict mapping test names to calibrated failure probabilities
        
    Returns:
        Enriched test list with epistemic features
        
    Added fields:
        - model_uncertainty: Calibrated failure probability
        - eig_bits: Expected Information Gain
        - entropy_before: Entropy before test execution
        - entropy_after: Expected entropy after test (0 for binary outcomes)
    """
    enriched = []
    
    for test in tests:
        test_name = test.get("name", "unknown")
        
        # Get calibrated failure probability
        failure_prob = model_predictions.get(test_name, test.get("model_uncertainty", 0.1))
        
        # Clamp to valid probability range
        failure_prob = max(0.0, min(1.0, failure_prob))
        
        # Compute epistemic features
        cost = test.get("cost_usd", 0.001)
        eig = compute_expected_information_gain(failure_prob, cost)
        entropy_before = bernoulli_entropy(failure_prob)
        
        # Expected entropy after test (assuming binary outcome)
        # E[H_after] ≈ 0 (we learn the outcome)
        entropy_after = 0.0
        
        # Enrich test
        enriched_test = test.copy()
        enriched_test.update({
            "model_uncertainty": float(failure_prob),  # Calibrated
            "eig_bits": float(eig),
            "entropy_before": float(entropy_before),
            "entropy_after": float(entropy_after),
        })
        
        enriched.append(enriched_test)
    
    return enriched


def compute_calibrated_confidence(
    failure_probs: np.ndarray,
    calibration_metrics: Dict[str, float]
) -> float:
    """Compute calibrated model confidence using ECE-adjusted probabilities.
    
    Confidence = 1 - ECE_penalty - abs(p - 0.5) * 2
    
    Args:
        failure_probs: Array of predicted failure probabilities
        calibration_metrics: Calibration metrics (ECE, Brier)
        
    Returns:
        Mean calibrated confidence (0 to 1)
        
    Notes:
        - Penalizes confidence by calibration error (ECE)
        - Returns 0.0 if ECE >= 1.0 (severely miscalibrated)
    """
    ece = calibration_metrics.get("ece", 0.1)
    
    # Penalize confidence by calibration error
    raw_confidence = 1 - np.abs(failure_probs - 0.5) * 2
    calibrated_confidence = np.maximum(0, raw_confidence - ece)
    
    return float(np.mean(calibrated_confidence))
