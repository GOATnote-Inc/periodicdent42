"""Uncertainty quantification and calibration."""

from src.uncertainty.calibration_metrics import (
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    coverage_width_criterion,
    expected_calibration_error,
    miscalibration_area,
    calibration_curve,
    interval_score,
    sharpness,
)
from src.uncertainty.conformal import (
    SplitConformalPredictor,
    MondrianConformalPredictor,
    stratify_by_prediction_bins,
    stratify_by_uncertainty,
)

__all__ = [
    # Metrics
    "prediction_interval_coverage_probability",
    "mean_prediction_interval_width",
    "coverage_width_criterion",
    "expected_calibration_error",
    "miscalibration_area",
    "calibration_curve",
    "interval_score",
    "sharpness",
    # Conformal
    "SplitConformalPredictor",
    "MondrianConformalPredictor",
    "stratify_by_prediction_bins",
    "stratify_by_uncertainty",
]

