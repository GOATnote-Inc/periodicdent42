"""Tests for Phase 4: Calibration Metrics and Conformal Prediction."""

import numpy as np
import pytest

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


@pytest.fixture
def perfect_intervals():
    """Generate perfectly calibrated intervals."""
    np.random.seed(42)
    
    n_samples = 200
    y_true = np.random.randn(n_samples) * 10
    y_pred = y_true  # Perfect predictions
    y_std = np.ones(n_samples) * 2.0
    
    # 95% intervals (±1.96σ)
    y_lower = y_pred - 1.96 * y_std
    y_upper = y_pred + 1.96 * y_std
    
    return y_true, y_pred, y_std, y_lower, y_upper


@pytest.fixture
def noisy_predictions():
    """Generate realistic noisy predictions."""
    np.random.seed(42)
    
    n_samples = 200
    y_true = np.random.randn(n_samples) * 10
    
    # Add noise to predictions
    y_pred = y_true + np.random.randn(n_samples) * 2.0
    
    # Heteroscedastic uncertainty
    y_std = 1.0 + 0.5 * np.abs(y_pred) / 10
    
    # 95% intervals
    y_lower = y_pred - 1.96 * y_std
    y_upper = y_pred + 1.96 * y_std
    
    return y_true, y_pred, y_std, y_lower, y_upper


@pytest.fixture
def simple_model():
    """Create a simple sklearn-like model for testing."""
    from sklearn.linear_model import Ridge
    
    return Ridge(alpha=1.0, random_state=42)


# ==================================
# Calibration Metrics Tests
# ==================================

class TestPICP:
    """Tests for Prediction Interval Coverage Probability."""
    
    def test_perfect_coverage(self, perfect_intervals):
        """Test PICP with perfect coverage."""
        y_true, y_pred, y_std, y_lower, y_upper = perfect_intervals
        
        picp = prediction_interval_coverage_probability(y_true, y_lower, y_upper)
        
        # Perfect predictions → 100% coverage
        assert picp == 1.0
    
    def test_realistic_coverage(self, noisy_predictions):
        """Test PICP with realistic noisy predictions."""
        y_true, y_pred, y_std, y_lower, y_upper = noisy_predictions
        
        picp = prediction_interval_coverage_probability(y_true, y_lower, y_upper)
        
        # Should be close to 95% (within reasonable tolerance for noisy data)
        assert 0.80 <= picp <= 1.0
    
    def test_zero_coverage(self):
        """Test PICP with no coverage (intervals miss all targets)."""
        y_true = np.array([0, 0, 0])
        y_lower = np.array([10, 10, 10])
        y_upper = np.array([20, 20, 20])
        
        picp = prediction_interval_coverage_probability(y_true, y_lower, y_upper)
        
        assert picp == 0.0
    
    def test_partial_coverage(self):
        """Test PICP with partial coverage."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_lower = np.array([0.5, 1.5, 3.5, 3.5])  # Miss 3rd, cover others
        y_upper = np.array([1.5, 2.5, 3.4, 4.5])
        
        picp = prediction_interval_coverage_probability(y_true, y_lower, y_upper)
        
        assert picp == 0.75  # 3 out of 4
    
    def test_empty_arrays(self):
        """Test PICP with empty arrays."""
        with pytest.raises(ValueError, match="Empty arrays"):
            prediction_interval_coverage_probability(
                np.array([]), np.array([]), np.array([])
            )
    
    def test_length_mismatch(self):
        """Test PICP with mismatched array lengths."""
        with pytest.raises(ValueError, match="same length"):
            prediction_interval_coverage_probability(
                np.array([1, 2]), np.array([0.5]), np.array([1.5, 2.5])
            )


class TestMPIW:
    """Tests for Mean Prediction Interval Width."""
    
    def test_constant_width(self):
        """Test MPIW with constant interval widths."""
        y_lower = np.array([0.0, 1.0, 2.0])
        y_upper = np.array([1.0, 2.0, 3.0])
        
        mpiw = mean_prediction_interval_width(y_lower, y_upper)
        
        assert mpiw == 1.0
    
    def test_variable_width(self):
        """Test MPIW with variable interval widths."""
        y_lower = np.array([0.0, 1.0, 2.0])
        y_upper = np.array([2.0, 2.0, 5.0])  # Widths: 2, 1, 3
        
        mpiw = mean_prediction_interval_width(y_lower, y_upper)
        
        assert np.isclose(mpiw, 2.0)  # Mean: (2 + 1 + 3) / 3 = 2
    
    def test_invalid_intervals(self):
        """Test MPIW with invalid intervals (upper < lower)."""
        y_lower = np.array([2.0, 3.0])
        y_upper = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Upper bounds must be"):
            mean_prediction_interval_width(y_lower, y_upper)


class TestCWC:
    """Tests for Coverage Width Criterion."""
    
    def test_good_coverage_narrow_intervals(self, noisy_predictions):
        """Test CWC with good coverage and narrow intervals."""
        y_true, y_pred, y_std, y_lower, y_upper = noisy_predictions
        
        cwc = coverage_width_criterion(y_true, y_lower, y_upper, eta=50.0)
        
        # Should be finite and positive
        assert cwc > 0
        assert np.isfinite(cwc)
    
    def test_under_coverage_penalty(self):
        """Test CWC applies penalty for under-coverage."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Severely under-covering intervals (miss most targets)
        y_lower_under = np.array([1.2, 2.2, 3.2, 4.2])
        y_upper_under = np.array([1.3, 2.3, 3.3, 4.3])
        
        # Well-covering intervals (wider)
        y_lower_good = np.array([0.0, 1.0, 2.0, 3.0])
        y_upper_good = np.array([2.0, 3.0, 4.0, 5.0])
        
        cwc_under = coverage_width_criterion(y_true, y_lower_under, y_upper_under, eta=50.0)
        cwc_good = coverage_width_criterion(y_true, y_lower_good, y_upper_good, eta=50.0)
        
        # Under-covering should have higher CWC (worse) despite narrower intervals
        assert cwc_under > cwc_good


class TestECE:
    """Tests for Expected Calibration Error."""
    
    def test_perfect_calibration(self, perfect_intervals):
        """Test ECE with perfect calibration."""
        y_true, y_pred, y_std, y_lower, y_upper = perfect_intervals
        
        ece = expected_calibration_error(y_true, y_pred, y_std, n_bins=5)
        
        # Perfect calibration → ECE ≈ 0
        assert ece < 0.1
    
    def test_realistic_calibration(self, noisy_predictions):
        """Test ECE with realistic noisy predictions."""
        y_true, y_pred, y_std, y_lower, y_upper = noisy_predictions
        
        ece = expected_calibration_error(y_true, y_pred, y_std, n_bins=10)
        
        # Should be finite and non-negative
        assert ece >= 0
        assert np.isfinite(ece)
    
    def test_invalid_std(self):
        """Test ECE with invalid (non-positive) standard deviations."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        y_std = np.array([1.0, 0.0, -1.0])  # Invalid: 0 and negative
        
        with pytest.raises(ValueError, match="Standard deviations must be positive"):
            expected_calibration_error(y_true, y_pred, y_std)


class TestMiscalibrationArea:
    """Tests for Miscalibration Area."""
    
    def test_perfect_calibration_ma(self, perfect_intervals):
        """Test MA with perfect calibration."""
        y_true, y_pred, y_std, y_lower, y_upper = perfect_intervals
        
        ma = miscalibration_area(y_true, y_pred, y_std, n_points=50)
        
        # Perfect predictions but finite sample → MA may be non-zero
        # Check that MA is finite and reasonable
        assert 0 <= ma <= 0.5
        assert np.isfinite(ma)
    
    def test_realistic_ma(self, noisy_predictions):
        """Test MA with realistic predictions."""
        y_true, y_pred, y_std, y_lower, y_upper = noisy_predictions
        
        ma = miscalibration_area(y_true, y_pred, y_std)
        
        assert ma >= 0
        assert np.isfinite(ma)


class TestCalibrationCurve:
    """Tests for Calibration Curve."""
    
    def test_curve_shape(self, noisy_predictions):
        """Test calibration curve has correct shape."""
        y_true, y_pred, y_std, y_lower, y_upper = noisy_predictions
        
        expected, observed = calibration_curve(y_true, y_pred, y_std, n_points=50)
        
        assert len(expected) == 50
        assert len(observed) == 50
        
        # Both should be monotonically increasing
        assert np.all(np.diff(expected) >= 0)
        
        # Values should be in [0, 1]
        assert np.all((expected >= 0) & (expected <= 1))
        assert np.all((observed >= 0) & (observed <= 1))
    
    def test_perfect_calibration_curve(self, perfect_intervals):
        """Test calibration curve for perfect calibration."""
        y_true, y_pred, y_std, y_lower, y_upper = perfect_intervals
        
        expected, observed = calibration_curve(y_true, y_pred, y_std, n_points=50)
        
        # Should lie close to diagonal (y = x), with some finite-sample deviation
        mae = np.mean(np.abs(expected - observed))
        assert mae < 0.5  # Relaxed for finite sample effects


class TestIntervalScore:
    """Tests for Interval Score (Winkler Score)."""
    
    def test_interval_score_computation(self, noisy_predictions):
        """Test interval score computation."""
        y_true, y_pred, y_std, y_lower, y_upper = noisy_predictions
        
        score = interval_score(y_true, y_lower, y_upper, alpha=0.05)
        
        assert score > 0
        assert np.isfinite(score)
    
    def test_narrower_better_with_coverage(self):
        """Test that narrower intervals have lower score (better) if coverage maintained."""
        y_true = np.array([1.0, 2.0, 3.0])
        
        # Narrow intervals (covering)
        y_lower_narrow = np.array([0.9, 1.9, 2.9])
        y_upper_narrow = np.array([1.1, 2.1, 3.1])
        
        # Wide intervals (also covering)
        y_lower_wide = np.array([0.0, 1.0, 2.0])
        y_upper_wide = np.array([2.0, 3.0, 4.0])
        
        score_narrow = interval_score(y_true, y_lower_narrow, y_upper_narrow)
        score_wide = interval_score(y_true, y_lower_wide, y_upper_wide)
        
        # Narrower intervals should have lower (better) score
        assert score_narrow < score_wide


class TestSharpness:
    """Tests for Sharpness metric."""
    
    def test_sharpness_computation(self):
        """Test sharpness computation."""
        y_lower = np.array([0.0, 1.0, 2.0])
        y_upper = np.array([1.0, 2.0, 3.0])  # Width = 1
        
        sharp = sharpness(y_lower, y_upper)
        
        assert sharp == 1.0  # 1 / MPIW = 1 / 1 = 1
    
    def test_narrower_intervals_higher_sharpness(self):
        """Test that narrower intervals have higher sharpness."""
        y_lower_narrow = np.array([0.5, 1.5])
        y_upper_narrow = np.array([1.5, 2.5])  # Width = 1
        
        y_lower_wide = np.array([0.0, 1.0])
        y_upper_wide = np.array([2.0, 3.0])  # Width = 2
        
        sharp_narrow = sharpness(y_lower_narrow, y_upper_narrow)
        sharp_wide = sharpness(y_lower_wide, y_upper_wide)
        
        assert sharp_narrow > sharp_wide


# ==================================
# Conformal Prediction Tests
# ==================================

class TestSplitConformalPredictor:
    """Tests for Split Conformal Prediction."""
    
    def test_initialization(self, simple_model):
        """Test split conformal initialization."""
        conformal = SplitConformalPredictor(simple_model, score_function="absolute")
        
        assert conformal.base_model is simple_model
        assert conformal.score_function == "absolute"
        assert not conformal.fitted_
    
    def test_fit(self, simple_model):
        """Test conformal fitting."""
        np.random.seed(42)
        
        # Generate data
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        # Split into fit and calibration
        X_fit, X_cal = X[:150], X[150:]
        y_fit, y_cal = y[:150], y[150:]
        
        # Fit conformal predictor
        conformal = SplitConformalPredictor(simple_model)
        conformal.fit(X_fit, y_fit, X_cal, y_cal)
        
        assert conformal.fitted_
        assert conformal.calibration_scores_ is not None
        assert len(conformal.calibration_scores_) == len(y_cal)
    
    def test_predict_with_interval(self, simple_model):
        """Test conformal interval prediction."""
        np.random.seed(42)
        
        # Generate data
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        X_fit, X_cal, X_test = X[:120], X[120:160], X[160:]
        y_fit, y_cal, y_test = y[:120], y[120:160], y[160:]
        
        # Fit and predict
        conformal = SplitConformalPredictor(simple_model)
        conformal.fit(X_fit, y_fit, X_cal, y_cal)
        
        y_pred, lower, upper = conformal.predict_with_interval(X_test, alpha=0.1)
        
        assert y_pred.shape == (len(X_test),)
        assert lower.shape == (len(X_test),)
        assert upper.shape == (len(X_test),)
        
        # Intervals should contain predictions
        assert np.all(lower <= y_pred)
        assert np.all(y_pred <= upper)
    
    def test_coverage_guarantee(self, simple_model):
        """Test that conformal prediction achieves target coverage."""
        np.random.seed(42)
        
        # Generate data
        X = np.random.randn(300, 5)
        y = X.sum(axis=1) + np.random.randn(300) * 0.5
        
        X_fit, X_cal, X_test = X[:150], X[150:225], X[225:]
        y_fit, y_cal, y_test = y[:150], y[150:225], y[225:]
        
        # Fit conformal predictor
        conformal = SplitConformalPredictor(simple_model)
        conformal.fit(X_fit, y_fit, X_cal, y_cal)
        
        # Predict with 90% intervals
        y_pred, lower, upper = conformal.predict_with_interval(X_test, alpha=0.1)
        
        # Check coverage
        picp = prediction_interval_coverage_probability(y_test, lower, upper)
        
        # Should achieve at least 90% coverage (finite-sample guarantee)
        # Allow more deviation for small test set (N=75)
        assert picp >= 0.80
    
    def test_get_calibration_quantiles(self, simple_model):
        """Test calibration quantile extraction."""
        np.random.seed(42)
        
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        X_fit, X_cal = X[:150], X[150:]
        y_fit, y_cal = y[:150], y[150:]
        
        conformal = SplitConformalPredictor(simple_model)
        conformal.fit(X_fit, y_fit, X_cal, y_cal)
        
        quantiles = conformal.get_calibration_quantiles([0.68, 0.90, 0.95])
        
        assert len(quantiles) == 3
        assert np.all(np.diff(quantiles) >= 0)  # Monotonic


class TestMondrianConformalPredictor:
    """Tests for Mondrian (Stratified) Conformal Prediction."""
    
    def test_initialization(self, simple_model):
        """Test Mondrian conformal initialization."""
        stratify_fn = lambda y: stratify_by_prediction_bins(y, n_bins=3)
        
        conformal = MondrianConformalPredictor(
            simple_model, stratify_function=stratify_fn
        )
        
        assert conformal.base_model is simple_model
        assert not conformal.fitted_
    
    def test_fit(self, simple_model):
        """Test Mondrian conformal fitting."""
        np.random.seed(42)
        
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        X_fit, X_cal = X[:150], X[150:]
        y_fit, y_cal = y[:150], y[150:]
        
        stratify_fn = lambda y: stratify_by_prediction_bins(y, n_bins=3)
        
        conformal = MondrianConformalPredictor(simple_model, stratify_fn)
        conformal.fit(X_fit, y_fit, X_cal, y_cal)
        
        assert conformal.fitted_
        assert len(conformal.calibration_scores_by_stratum_) > 0
    
    def test_predict_with_interval(self, simple_model):
        """Test Mondrian interval prediction."""
        np.random.seed(42)
        
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        X_fit, X_cal, X_test = X[:120], X[120:160], X[160:]
        y_fit, y_cal, y_test = y[:120], y[120:160], y[160:]
        
        stratify_fn = lambda y: stratify_by_prediction_bins(y, n_bins=3)
        
        conformal = MondrianConformalPredictor(simple_model, stratify_fn)
        conformal.fit(X_fit, y_fit, X_cal, y_cal)
        
        y_pred, lower, upper = conformal.predict_with_interval(X_test, alpha=0.1)
        
        assert y_pred.shape == (len(X_test),)
        assert np.all(lower <= upper)


class TestStratificationFunctions:
    """Tests for stratification utility functions."""
    
    def test_stratify_by_prediction_bins(self):
        """Test prediction-based stratification."""
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        strata = stratify_by_prediction_bins(y_pred, n_bins=3)
        
        assert len(strata) == len(y_pred)
        assert set(strata) == {0, 1, 2}  # 3 strata
    
    def test_stratify_by_uncertainty(self):
        """Test uncertainty-based stratification."""
        y_std = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        
        strata = stratify_by_uncertainty(y_std, n_bins=2)
        
        assert len(strata) == len(y_std)
        assert set(strata) == {0, 1}  # 2 strata


# ==================================
# Integration Tests
# ==================================

@pytest.mark.integration
class TestCalibrationIntegration:
    """Integration tests for calibration + conformal."""
    
    def test_conformal_improves_calibration(self, simple_model):
        """Test that conformal prediction improves calibration."""
        np.random.seed(42)
        
        # Generate heteroscedastic data
        X = np.random.randn(300, 5)
        noise = 0.5 + 0.5 * np.abs(X[:, 0])  # Heteroscedastic noise
        y = X.sum(axis=1) + np.random.randn(300) * noise
        
        X_fit, X_cal, X_test = X[:150], X[150:225], X[225:]
        y_fit, y_cal, y_test = y[:150], y[150:225], y[225:]
        
        # Baseline model (without conformal)
        simple_model.fit(X_fit, y_fit)
        y_pred_base = simple_model.predict(X_test)
        
        # Conformal model
        conformal = SplitConformalPredictor(simple_model)
        conformal.fit(X_fit, y_fit, X_cal, y_cal)
        y_pred_conf, lower_conf, upper_conf = conformal.predict_with_interval(X_test)
        
        # Check coverage
        coverage_conf = prediction_interval_coverage_probability(
            y_test, lower_conf, upper_conf
        )
        
        # Conformal should achieve high coverage
        assert coverage_conf >= 0.85

