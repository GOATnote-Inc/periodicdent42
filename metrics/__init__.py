"""Metrics package for epistemic efficiency tracking."""

from .epistemic import (
    bernoulli_entropy,
    compute_expected_information_gain,
    compute_detection_rate,
    compute_entropy_delta,
    compute_epistemic_efficiency,
    enrich_tests_with_epistemic_features,
    compute_calibrated_confidence,
)

__all__ = [
    "bernoulli_entropy",
    "compute_expected_information_gain",
    "compute_detection_rate",
    "compute_entropy_delta",
    "compute_epistemic_efficiency",
    "enrich_tests_with_epistemic_features",
    "compute_calibrated_confidence",
]
