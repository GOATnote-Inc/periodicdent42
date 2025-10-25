"""Tests for Phase 5: OOD Detection."""

import numpy as np
import pytest

from src.guards.ood_detectors import (
    MahalanobisOODDetector,
    KDEOODDetector,
    ConformalNoveltyDetector,
    create_ood_detector,
)


@pytest.fixture
def in_distribution_data():
    """Generate in-distribution data (2D Gaussian)."""
    np.random.seed(42)
    
    # In-distribution: N(μ=[0, 0], Σ=[[1, 0.5], [0.5, 1]])
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    X_train = np.random.multivariate_normal(mean, cov, size=200)
    X_test_in = np.random.multivariate_normal(mean, cov, size=50)
    
    return X_train, X_test_in


@pytest.fixture
def out_of_distribution_data():
    """Generate OOD data (shifted distribution)."""
    np.random.seed(42)
    
    # OOD: N(μ=[5, 5], Σ=[[1, 0], [0, 1]]) - far from training distribution
    mean_ood = np.array([5.0, 5.0])
    cov_ood = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    X_test_ood = np.random.multivariate_normal(mean_ood, cov_ood, size=50)
    
    return X_test_ood


@pytest.fixture
def simple_regression_model():
    """Create a simple regression model for conformal novelty detection."""
    from sklearn.linear_model import Ridge
    
    return Ridge(alpha=1.0, random_state=42)


# ==================================
# Mahalanobis OOD Detector Tests
# ==================================

class TestMahalanobisOODDetector:
    """Tests for Mahalanobis distance-based OOD detection."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = MahalanobisOODDetector(alpha=0.01, robust=True)
        
        assert detector.alpha == 0.01
        assert detector.robust is True
        assert not detector.fitted_
    
    def test_fit(self, in_distribution_data):
        """Test fitting the detector."""
        X_train, _ = in_distribution_data
        
        detector = MahalanobisOODDetector(alpha=0.01)
        detector.fit(X_train)
        
        assert detector.fitted_
        assert detector.mean_ is not None
        assert detector.cov_ is not None
        assert detector.cov_inv_ is not None
        assert detector.threshold_ is not None
    
    def test_predict_in_distribution(self, in_distribution_data):
        """Test that in-distribution samples are not flagged as OOD."""
        X_train, X_test_in = in_distribution_data
        
        detector = MahalanobisOODDetector(alpha=0.01)
        detector.fit(X_train)
        
        is_ood = detector.predict(X_test_in)
        
        # Most in-distribution samples should not be flagged (< 5% due to alpha=0.01)
        ood_rate = is_ood.mean()
        assert ood_rate < 0.10  # Allow some false positives
    
    def test_predict_out_of_distribution(
        self, in_distribution_data, out_of_distribution_data
    ):
        """Test that OOD samples are correctly flagged."""
        X_train, _ = in_distribution_data
        X_test_ood = out_of_distribution_data
        
        detector = MahalanobisOODDetector(alpha=0.01)
        detector.fit(X_train)
        
        is_ood = detector.predict(X_test_ood)
        
        # Most OOD samples should be flagged (> 90%)
        ood_rate = is_ood.mean()
        assert ood_rate > 0.90
    
    def test_mahalanobis_distance(self, in_distribution_data):
        """Test Mahalanobis distance computation."""
        X_train, X_test_in = in_distribution_data
        
        detector = MahalanobisOODDetector()
        detector.fit(X_train)
        
        distances = detector.mahalanobis_distance(X_test_in)
        
        assert distances.shape == (len(X_test_in),)
        assert np.all(distances >= 0)  # Distances are non-negative
    
    def test_predict_proba(self, in_distribution_data, out_of_distribution_data):
        """Test OOD probability prediction."""
        X_train, X_test_in = in_distribution_data
        X_test_ood = out_of_distribution_data
        
        detector = MahalanobisOODDetector(alpha=0.01)
        detector.fit(X_train)
        
        proba_in = detector.predict_proba(X_test_in)
        proba_ood = detector.predict_proba(X_test_ood)
        
        # OOD samples should have higher OOD probability
        assert proba_ood.mean() > proba_in.mean()
        
        # Probabilities should be in [0, 1]
        assert np.all((proba_in >= 0) & (proba_in <= 1))
        assert np.all((proba_ood >= 0) & (proba_ood <= 1))
    
    def test_insufficient_samples_error(self):
        """Test error when n_samples < n_features."""
        X_train = np.random.randn(5, 10)  # 5 samples, 10 features
        
        detector = MahalanobisOODDetector()
        
        with pytest.raises(ValueError, match="Insufficient samples"):
            detector.fit(X_train)
    
    def test_reproducibility(self, in_distribution_data):
        """Test that results are reproducible with same seed."""
        X_train, X_test_in = in_distribution_data
        
        detector1 = MahalanobisOODDetector(random_state=42)
        detector1.fit(X_train)
        is_ood1 = detector1.predict(X_test_in)
        
        detector2 = MahalanobisOODDetector(random_state=42)
        detector2.fit(X_train)
        is_ood2 = detector2.predict(X_test_in)
        
        assert np.array_equal(is_ood1, is_ood2)


# ==================================
# KDE OOD Detector Tests
# ==================================

class TestKDEOODDetector:
    """Tests for KDE-based OOD detection."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = KDEOODDetector(bandwidth="scott", alpha=0.01)
        
        assert detector.bandwidth == "scott"
        assert detector.alpha == 0.01
        assert not detector.fitted_
    
    def test_fit(self, in_distribution_data):
        """Test fitting the detector."""
        X_train, _ = in_distribution_data
        
        detector = KDEOODDetector(alpha=0.01)
        detector.fit(X_train)
        
        assert detector.fitted_
        assert detector.kde_ is not None
        assert detector.threshold_ is not None
    
    def test_predict_in_distribution(self, in_distribution_data):
        """Test that in-distribution samples are not flagged as OOD."""
        X_train, X_test_in = in_distribution_data
        
        detector = KDEOODDetector(alpha=0.01)
        detector.fit(X_train)
        
        is_ood = detector.predict(X_test_in)
        
        # Most in-distribution samples should not be flagged
        ood_rate = is_ood.mean()
        assert ood_rate < 0.15  # Allow some false positives
    
    def test_predict_out_of_distribution(
        self, in_distribution_data, out_of_distribution_data
    ):
        """Test that OOD samples are correctly flagged."""
        X_train, _ = in_distribution_data
        X_test_ood = out_of_distribution_data
        
        detector = KDEOODDetector(alpha=0.01)
        detector.fit(X_train)
        
        is_ood = detector.predict(X_test_ood)
        
        # Most OOD samples should be flagged
        ood_rate = is_ood.mean()
        assert ood_rate > 0.80  # Allow some misses for KDE
    
    def test_predict_proba(self, in_distribution_data, out_of_distribution_data):
        """Test OOD probability prediction."""
        X_train, X_test_in = in_distribution_data
        X_test_ood = out_of_distribution_data
        
        detector = KDEOODDetector(alpha=0.01)
        detector.fit(X_train)
        
        proba_in = detector.predict_proba(X_test_in)
        proba_ood = detector.predict_proba(X_test_ood)
        
        # OOD samples should have higher OOD probability
        assert proba_ood.mean() > proba_in.mean()
        
        # Probabilities should be in [0, 1]
        assert np.all((proba_in >= 0) & (proba_in <= 1))
        assert np.all((proba_ood >= 0) & (proba_ood <= 1))
    
    def test_different_bandwidths(self, in_distribution_data):
        """Test different bandwidth selection methods."""
        X_train, X_test_in = in_distribution_data
        
        for bandwidth in ["scott", "silverman", 0.5]:
            detector = KDEOODDetector(bandwidth=bandwidth, alpha=0.01)
            detector.fit(X_train)
            
            is_ood = detector.predict(X_test_in)
            
            assert is_ood.shape == (len(X_test_in),)
            assert is_ood.dtype == int


# ==================================
# Conformal Novelty Detector Tests
# ==================================

class TestConformalNoveltyDetector:
    """Tests for conformal novelty detection."""
    
    def test_initialization(self, simple_regression_model):
        """Test detector initialization."""
        detector = ConformalNoveltyDetector(
            simple_regression_model, score_function="absolute", alpha=0.01
        )
        
        assert detector.base_model is simple_regression_model
        assert detector.score_function == "absolute"
        assert not detector.fitted_
    
    def test_fit(self, simple_regression_model):
        """Test fitting the detector."""
        np.random.seed(42)
        
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        X_fit, X_cal = X[:150], X[150:]
        y_fit, y_cal = y[:150], y[150:]
        
        detector = ConformalNoveltyDetector(simple_regression_model)
        detector.fit(X_fit, y_fit, X_cal, y_cal)
        
        assert detector.fitted_
        assert detector.calibration_scores_ is not None
        assert detector.threshold_ is not None
    
    def test_predict_with_targets(self, simple_regression_model):
        """Test novelty prediction with targets (for validation)."""
        np.random.seed(42)
        
        # Generate data
        X = np.random.randn(300, 5)
        y = X.sum(axis=1) + np.random.randn(300) * 0.5
        
        X_fit, X_cal, X_test = X[:150], X[150:225], X[225:]
        y_fit, y_cal, y_test = y[:150], y[150:225], y[225:]
        
        # Fit detector
        detector = ConformalNoveltyDetector(simple_regression_model, alpha=0.01)
        detector.fit(X_fit, y_fit, X_cal, y_cal)
        
        # Predict with targets
        is_novel = detector.predict_with_targets(X_test, y_test)
        
        assert is_novel.shape == (len(X_test),)
        
        # Most in-distribution samples should not be flagged
        novelty_rate = is_novel.mean()
        assert novelty_rate < 0.10  # Allow some false positives
    
    def test_predict_with_ood(self, simple_regression_model):
        """Test novelty detection on OOD data."""
        np.random.seed(42)
        
        # In-distribution data
        X_train = np.random.randn(200, 5)
        y_train = X_train.sum(axis=1) + np.random.randn(200) * 0.5
        
        # OOD data (different relationship)
        X_test_ood = np.random.randn(50, 5) * 3 + 10  # Shifted and scaled
        y_test_ood = X_test_ood.sum(axis=1) ** 2  # Different relationship
        
        X_fit, X_cal = X_train[:150], X_train[150:]
        y_fit, y_cal = y_train[:150], y_train[150:]
        
        # Fit detector
        detector = ConformalNoveltyDetector(simple_regression_model, alpha=0.01)
        detector.fit(X_fit, y_fit, X_cal, y_cal)
        
        # Predict novelty
        is_novel = detector.predict_with_targets(X_test_ood, y_test_ood)
        
        # Most OOD samples should be flagged
        novelty_rate = is_novel.mean()
        assert novelty_rate > 0.50  # At least half should be detected


# ==================================
# Factory Function Tests
# ==================================

class TestOODDetectorFactory:
    """Tests for create_ood_detector factory function."""
    
    def test_create_mahalanobis(self):
        """Test creating Mahalanobis detector."""
        detector = create_ood_detector("mahalanobis", alpha=0.05)
        
        assert isinstance(detector, MahalanobisOODDetector)
        assert detector.alpha == 0.05
    
    def test_create_kde(self):
        """Test creating KDE detector."""
        detector = create_ood_detector("kde", bandwidth="scott", alpha=0.01)
        
        assert isinstance(detector, KDEOODDetector)
        assert detector.bandwidth == "scott"
    
    def test_create_conformal(self, simple_regression_model):
        """Test creating conformal detector."""
        detector = create_ood_detector(
            "conformal", base_model=simple_regression_model, alpha=0.01
        )
        
        assert isinstance(detector, ConformalNoveltyDetector)
        assert detector.alpha == 0.01
    
    def test_unknown_method_error(self):
        """Test error for unknown OOD detection method."""
        with pytest.raises(ValueError, match="Unknown OOD detection method"):
            create_ood_detector("unknown_method")


# ==================================
# Integration Tests
# ==================================

@pytest.mark.integration
class TestOODDetectorComparison:
    """Integration tests comparing different OOD detectors."""
    
    def test_all_detectors_on_same_data(
        self, in_distribution_data, out_of_distribution_data
    ):
        """Test all detectors on the same data."""
        X_train, X_test_in = in_distribution_data
        X_test_ood = out_of_distribution_data
        
        # Create detectors
        detectors = {
            "Mahalanobis": MahalanobisOODDetector(alpha=0.01),
            "KDE": KDEOODDetector(alpha=0.01),
        }
        
        results = {}
        
        for name, detector in detectors.items():
            detector.fit(X_train)
            
            is_ood_in = detector.predict(X_test_in)
            is_ood_out = detector.predict(X_test_ood)
            
            results[name] = {
                "fpr": is_ood_in.mean(),  # False positive rate
                "tpr": is_ood_out.mean(),  # True positive rate (detection rate)
            }
            
            print(f"{name}: FPR={results[name]['fpr']:.2%}, TPR={results[name]['tpr']:.2%}")
        
        # All detectors should have reasonable performance
        for name, metrics in results.items():
            assert metrics["fpr"] < 0.15, f"{name} has too many false positives"
            assert metrics["tpr"] > 0.70, f"{name} has low detection rate"
    
    def test_ensemble_voting(
        self, in_distribution_data, out_of_distribution_data
    ):
        """Test ensemble OOD detection via majority voting."""
        X_train, X_test_in = in_distribution_data
        X_test_ood = out_of_distribution_data
        
        # Create multiple detectors
        detectors = [
            MahalanobisOODDetector(alpha=0.01),
            KDEOODDetector(alpha=0.01),
        ]
        
        # Fit all detectors
        for detector in detectors:
            detector.fit(X_train)
        
        # Get predictions
        predictions_in = np.array([d.predict(X_test_in) for d in detectors])
        predictions_ood = np.array([d.predict(X_test_ood) for d in detectors])
        
        # Majority voting
        ensemble_in = (predictions_in.sum(axis=0) >= len(detectors) // 2 + 1).astype(int)
        ensemble_ood = (predictions_ood.sum(axis=0) >= len(detectors) // 2 + 1).astype(int)
        
        # Ensemble should have good performance
        fpr = ensemble_in.mean()
        tpr = ensemble_ood.mean()
        
        print(f"Ensemble: FPR={fpr:.2%}, TPR={tpr:.2%}")
        
        assert fpr < 0.15
        assert tpr > 0.75

