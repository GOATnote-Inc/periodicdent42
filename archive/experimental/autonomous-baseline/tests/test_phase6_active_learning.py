"""Tests for Phase 6: Active Learning."""

import numpy as np
import pytest

from src.active_learning.acquisition import (
    upper_confidence_bound,
    expected_improvement,
    maximum_variance,
    expected_information_gain_proxy,
    thompson_sampling,
    create_acquisition_function,
)
from src.active_learning.diversity import (
    k_medoids_selection,
    greedy_diversity_selection,
    dpp_selection,
    create_diversity_selector,
)
from src.active_learning.loop import (
    ActiveLearningLoop,
    go_no_go_gate,
)


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    np.random.seed(42)
    
    n_samples = 50
    y_pred = np.random.randn(n_samples) * 10 + 50
    y_std = np.abs(np.random.randn(n_samples)) + 0.5
    
    return y_pred, y_std


@pytest.fixture
def sample_features():
    """Generate sample features for diversity testing."""
    np.random.seed(42)
    
    n_samples = 50
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    return X


@pytest.fixture
def simple_model():
    """Create a simple model for active learning."""
    from sklearn.linear_model import Ridge
    
    return Ridge(alpha=1.0, random_state=42)


# ==================================
# Acquisition Function Tests
# ==================================

class TestUpperConfidenceBound:
    """Tests for UCB acquisition function."""
    
    def test_ucb_maximize(self, sample_predictions):
        """Test UCB for maximization."""
        y_pred, y_std = sample_predictions
        
        ucb_scores = upper_confidence_bound(y_pred, y_std, kappa=2.0, maximize=True)
        
        assert ucb_scores.shape == y_pred.shape
        assert np.all(ucb_scores >= y_pred)  # UCB should be >= mean for maximize
    
    def test_ucb_minimize(self, sample_predictions):
        """Test UCB for minimization."""
        y_pred, y_std = sample_predictions
        
        ucb_scores = upper_confidence_bound(y_pred, y_std, kappa=2.0, maximize=False)
        
        assert ucb_scores.shape == y_pred.shape
        assert np.all(ucb_scores <= y_pred)  # UCB should be <= mean for minimize
    
    def test_ucb_kappa_effect(self, sample_predictions):
        """Test that higher kappa increases exploration."""
        y_pred, y_std = sample_predictions
        
        ucb_low = upper_confidence_bound(y_pred, y_std, kappa=0.5)
        ucb_high = upper_confidence_bound(y_pred, y_std, kappa=3.0)
        
        # Higher kappa should give more weight to uncertainty
        assert (ucb_high - y_pred).mean() > (ucb_low - y_pred).mean()
    
    def test_ucb_validation_errors(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="same length"):
            upper_confidence_bound(np.array([1, 2]), np.array([0.5]), kappa=2.0)
        
        with pytest.raises(ValueError, match="non-negative"):
            upper_confidence_bound(np.array([1, 2]), np.array([0.5, -0.1]), kappa=2.0)


class TestExpectedImprovement:
    """Tests for EI acquisition function."""
    
    def test_ei_basic(self, sample_predictions):
        """Test basic EI computation."""
        y_pred, y_std = sample_predictions
        y_best = y_pred.max()
        
        ei_scores = expected_improvement(y_pred, y_std, y_best, maximize=True)
        
        assert ei_scores.shape == y_pred.shape
        assert np.all(ei_scores >= 0)  # EI is always non-negative
    
    def test_ei_best_point(self, sample_predictions):
        """Test that EI is low at current best."""
        y_pred, y_std = sample_predictions
        
        # Set best to a point with low uncertainty
        best_idx = np.argmin(y_std)
        y_best = y_pred[best_idx]
        
        ei_scores = expected_improvement(y_pred, y_std, y_best, xi=0.0)
        
        # EI at best point should be very low
        assert ei_scores[best_idx] < ei_scores.mean()
    
    def test_ei_maximize_vs_minimize(self, sample_predictions):
        """Test EI for maximization vs minimization."""
        y_pred, y_std = sample_predictions
        
        y_best_max = y_pred.max()
        y_best_min = y_pred.min()
        
        ei_max = expected_improvement(y_pred, y_std, y_best_max, maximize=True)
        ei_min = expected_improvement(y_pred, y_std, y_best_min, maximize=False)
        
        # Different optimization directions should prefer different points
        assert ei_max.argmax() != ei_min.argmax()


class TestMaximumVariance:
    """Tests for MaxVar acquisition function."""
    
    def test_maxvar_basic(self, sample_predictions):
        """Test MaxVar computation."""
        _, y_std = sample_predictions
        
        var_scores = maximum_variance(y_std)
        
        assert var_scores.shape == y_std.shape
        assert np.all(var_scores >= 0)
        assert np.allclose(var_scores, y_std ** 2)
    
    def test_maxvar_selects_uncertain(self, sample_predictions):
        """Test that MaxVar selects most uncertain samples."""
        _, y_std = sample_predictions
        
        var_scores = maximum_variance(y_std)
        
        most_uncertain_idx = y_std.argmax()
        highest_var_idx = var_scores.argmax()
        
        assert most_uncertain_idx == highest_var_idx


class TestEIGProxy:
    """Tests for EIG proxy."""
    
    def test_eig_proxy_basic(self, sample_predictions):
        """Test EIG proxy computation."""
        y_pred, y_std = sample_predictions
        
        eig_scores = expected_information_gain_proxy(y_std, y_pred, alpha=1.0)
        
        assert eig_scores.shape == y_std.shape
        assert np.all(eig_scores >= 0)
    
    def test_eig_proxy_without_pred(self, sample_predictions):
        """Test EIG proxy without predictions (fallback to uncertainty)."""
        _, y_std = sample_predictions
        
        eig_scores = expected_information_gain_proxy(y_std, y_pred=None)
        
        assert np.allclose(eig_scores, y_std)


class TestThompsonSampling:
    """Tests for Thompson sampling."""
    
    def test_thompson_basic(self, sample_predictions):
        """Test Thompson sampling."""
        y_pred, y_std = sample_predictions
        
        probs = thompson_sampling(y_pred, y_std, n_samples=100, random_state=42)
        
        assert probs.shape == y_pred.shape
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.isclose(probs.sum(), 1.0)  # Should sum to 1
    
    def test_thompson_reproducibility(self, sample_predictions):
        """Test that Thompson sampling is reproducible."""
        y_pred, y_std = sample_predictions
        
        probs1 = thompson_sampling(y_pred, y_std, random_state=42)
        probs2 = thompson_sampling(y_pred, y_std, random_state=42)
        
        assert np.allclose(probs1, probs2)


class TestAcquisitionFactory:
    """Tests for acquisition function factory."""
    
    def test_create_ucb(self, sample_predictions):
        """Test creating UCB function."""
        y_pred, y_std = sample_predictions
        
        acq_fn = create_acquisition_function("ucb", kappa=2.0, maximize=True)
        scores = acq_fn(y_pred=y_pred, y_std=y_std)
        
        assert scores.shape == y_pred.shape
    
    def test_create_ei(self, sample_predictions):
        """Test creating EI function."""
        y_pred, y_std = sample_predictions
        
        acq_fn = create_acquisition_function("ei", y_best=50.0, maximize=True)
        scores = acq_fn(y_pred=y_pred, y_std=y_std, y_best=50.0)
        
        assert scores.shape == y_pred.shape
    
    def test_create_unknown_method(self):
        """Test error for unknown method."""
        with pytest.raises(ValueError, match="Unknown acquisition method"):
            create_acquisition_function("unknown_method")


# ==================================
# Diversity Selection Tests
# ==================================

class TestKMedoidsSelection:
    """Tests for k-Medoids diversity selection."""
    
    def test_k_medoids_basic(self, sample_features, sample_predictions):
        """Test basic k-Medoids selection."""
        X = sample_features
        _, y_std = sample_predictions
        acq_scores = y_std
        
        batch_size = 10
        selected = k_medoids_selection(X, acq_scores, batch_size, random_state=42)
        
        assert len(selected) == batch_size
        assert len(set(selected)) == batch_size  # No duplicates
        assert np.all(selected >= 0) and np.all(selected < len(X))
    
    def test_k_medoids_covers_space(self, sample_features, sample_predictions):
        """Test that k-Medoids covers the feature space."""
        X = sample_features
        _, y_std = sample_predictions
        acq_scores = y_std
        
        batch_size = 10
        selected = k_medoids_selection(X, acq_scores, batch_size, random_state=42)
        
        # Selected points should be well-distributed
        # Check that pairwise distances are reasonable
        from scipy.spatial.distance import pdist
        
        distances = pdist(X[selected])
        
        # Average distance should be non-trivial (not all clustered)
        assert distances.mean() > 0.1
    
    def test_k_medoids_validation(self, sample_features, sample_predictions):
        """Test input validation."""
        X = sample_features
        _, y_std = sample_predictions
        
        with pytest.raises(ValueError, match="same length"):
            k_medoids_selection(X, y_std[:10], batch_size=5)
        
        with pytest.raises(ValueError, match="cannot exceed"):
            k_medoids_selection(X, y_std, batch_size=100)


class TestGreedyDiversitySelection:
    """Tests for greedy diversity selection."""
    
    def test_greedy_basic(self, sample_features, sample_predictions):
        """Test basic greedy selection."""
        X = sample_features
        _, y_std = sample_predictions
        acq_scores = y_std
        
        batch_size = 10
        selected = greedy_diversity_selection(
            X, acq_scores, batch_size, alpha=0.5
        )
        
        assert len(selected) == batch_size
        assert len(set(selected)) == batch_size
    
    def test_greedy_alpha_extremes(self, sample_features, sample_predictions):
        """Test greedy selection with extreme alpha values."""
        X = sample_features
        _, y_std = sample_predictions
        acq_scores = y_std
        
        batch_size = 5
        
        # Alpha = 1.0: pure acquisition (should select highest scores)
        selected_pure_acq = greedy_diversity_selection(X, acq_scores, batch_size, alpha=1.0)
        top_k = np.argsort(acq_scores)[-batch_size:]
        
        # First element should be from top-k
        assert selected_pure_acq[0] == acq_scores.argmax()
        
        # Alpha = 0.0: pure diversity
        selected_pure_div = greedy_diversity_selection(X, acq_scores, batch_size, alpha=0.0)
        
        # Should be different from pure acquisition
        assert not np.array_equal(sorted(selected_pure_acq), sorted(selected_pure_div))


class TestDPPSelection:
    """Tests for DPP selection."""
    
    def test_dpp_basic(self, sample_features, sample_predictions):
        """Test basic DPP selection."""
        X = sample_features
        _, y_std = sample_predictions
        acq_scores = y_std
        
        batch_size = 10
        selected = dpp_selection(X, acq_scores, batch_size, lambda_diversity=1.0)
        
        assert len(selected) == batch_size
        assert len(set(selected)) == batch_size
    
    def test_dpp_diversity_vs_quality(self, sample_features, sample_predictions):
        """Test DPP balances diversity and quality."""
        X = sample_features
        _, y_std = sample_predictions
        acq_scores = y_std
        
        batch_size = 5
        
        # Low diversity weight: favor quality
        selected_low_div = dpp_selection(X, acq_scores, batch_size, lambda_diversity=0.1)
        
        # High diversity weight: favor diversity
        selected_high_div = dpp_selection(X, acq_scores, batch_size, lambda_diversity=10.0)
        
        # Should select different batches
        assert not np.array_equal(sorted(selected_low_div), sorted(selected_high_div))


class TestDiversityFactory:
    """Tests for diversity selector factory."""
    
    def test_create_k_medoids(self, sample_features, sample_predictions):
        """Test creating k-Medoids selector."""
        X = sample_features
        _, y_std = sample_predictions
        
        selector = create_diversity_selector("k_medoids", random_state=42)
        selected = selector(X, y_std, batch_size=10)
        
        assert len(selected) == 10
    
    def test_create_unknown_method(self):
        """Test error for unknown method."""
        with pytest.raises(ValueError, match="Unknown diversity method"):
            create_diversity_selector("unknown_method")


# ==================================
# Active Learning Loop Tests
# ==================================

class TestActiveLearningLoop:
    """Tests for active learning loop."""
    
    def test_loop_initialization(self, simple_model):
        """Test loop initialization."""
        acq_fn = create_acquisition_function("maxvar")
        
        loop = ActiveLearningLoop(
            base_model=simple_model,
            acquisition_fn=acq_fn,
            budget=100,
            batch_size=10,
        )
        
        assert loop.base_model is simple_model
        assert loop.budget == 100
        assert loop.batch_size == 10
        assert loop.budget_used_ == 0
    
    def test_loop_run(self, simple_model):
        """Test running active learning loop."""
        np.random.seed(42)
        
        # Generate data
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        # Split into labeled and unlabeled
        X_labeled, X_unlabeled = X[:20], X[20:]
        y_labeled, y_unlabeled = y[:20], y[20:]
        
        # Create loop
        acq_fn = create_acquisition_function("maxvar")
        
        loop = ActiveLearningLoop(
            base_model=simple_model,
            acquisition_fn=acq_fn,
            budget=50,
            batch_size=10,
        )
        
        # Run
        result = loop.run(X_labeled, y_labeled, X_unlabeled, y_unlabeled, n_iterations=3)
        
        assert "X_train" in result
        assert "y_train" in result
        assert "history" in result
        assert len(result["X_train"]) == 20 + 3 * 10  # Initial + 3 iterations
        assert loop.budget_used_ == 30
    
    def test_loop_with_diversity(self, simple_model):
        """Test loop with diversity selector."""
        np.random.seed(42)
        
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        X_labeled, X_unlabeled = X[:20], X[20:]
        y_labeled, y_unlabeled = y[:20], y[20:]
        
        acq_fn = create_acquisition_function("maxvar")
        div_selector = create_diversity_selector("greedy", alpha=0.5)
        
        loop = ActiveLearningLoop(
            base_model=simple_model,
            acquisition_fn=acq_fn,
            diversity_selector=div_selector,
            budget=30,
            batch_size=10,
        )
        
        result = loop.run(X_labeled, y_labeled, X_unlabeled, y_unlabeled, n_iterations=2)
        
        assert len(result["X_train"]) == 20 + 2 * 10
        assert loop.budget_used_ == 20
    
    def test_loop_budget_exhausted(self, simple_model):
        """Test that loop respects budget."""
        np.random.seed(42)
        
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.5
        
        X_labeled, X_unlabeled = X[:20], X[20:]
        y_labeled, y_unlabeled = y[:20], y[20:]
        
        acq_fn = create_acquisition_function("maxvar")
        
        loop = ActiveLearningLoop(
            base_model=simple_model,
            acquisition_fn=acq_fn,
            budget=15,  # Small budget
            batch_size=10,
        )
        
        result = loop.run(X_labeled, y_labeled, X_unlabeled, y_unlabeled, n_iterations=10)
        
        # Should stop after using budget
        assert loop.budget_used_ <= 15
        assert len(result["X_train"]) <= 20 + 15


class TestGoNoGoGate:
    """Tests for GO/NO-GO decision gate."""
    
    def test_go_decision(self):
        """Test GO decision (interval above threshold)."""
        y_pred = np.array([100.0, 110.0, 120.0])
        y_std = np.array([5.0, 5.0, 5.0])
        y_lower = np.array([90.0, 100.0, 110.0])
        y_upper = np.array([110.0, 120.0, 130.0])
        
        decisions = go_no_go_gate(
            y_pred, y_std, y_lower, y_upper,
            threshold_min=80.0,
            threshold_max=150.0
        )
        
        # All intervals are within [80, 150] and above 80
        assert np.all(decisions == 1)  # All GO
    
    def test_no_go_decision(self):
        """Test NO-GO decision (interval below threshold)."""
        y_pred = np.array([10.0, 20.0, 30.0])
        y_std = np.array([5.0, 5.0, 5.0])
        y_lower = np.array([5.0, 15.0, 25.0])
        y_upper = np.array([15.0, 25.0, 35.0])
        
        decisions = go_no_go_gate(
            y_pred, y_std, y_lower, y_upper,
            threshold_min=50.0,
            threshold_max=np.inf
        )
        
        # All intervals are below 50
        assert np.all(decisions == -1)  # All NO-GO
    
    def test_maybe_decision(self):
        """Test MAYBE decision (interval overlaps threshold)."""
        y_pred = np.array([75.0, 80.0, 90.0])
        y_std = np.array([10.0, 10.0, 5.0])
        y_lower = np.array([65.0, 70.0, 85.0])
        y_upper = np.array([85.0, 90.0, 95.0])
        
        decisions = go_no_go_gate(
            y_pred, y_std, y_lower, y_upper,
            threshold_min=77.0,  # LN2 temperature
            threshold_max=np.inf
        )
        
        # Intervals overlap or are above 77K threshold
        assert decisions[0] == 0  # MAYBE (65-85 overlaps 77)
        assert decisions[1] == 0  # MAYBE (70-90 overlaps 77)
        assert decisions[2] == 1  # GO (85-95 entirely above 77)
    
    def test_mixed_decisions(self):
        """Test mix of GO/MAYBE/NO-GO decisions."""
        y_pred = np.array([50.0, 77.0, 100.0])
        y_std = np.array([5.0, 5.0, 5.0])
        y_lower = np.array([45.0, 72.0, 95.0])
        y_upper = np.array([55.0, 82.0, 105.0])
        
        decisions = go_no_go_gate(
            y_pred, y_std, y_lower, y_upper,
            threshold_min=77.0,
            threshold_max=np.inf
        )
        
        assert decisions[0] == -1  # NO-GO (entirely below 77)
        assert decisions[1] == 0   # MAYBE (overlaps 77)
        assert decisions[2] == 1   # GO (entirely above 77)


# ==================================
# Integration Tests
# ==================================

@pytest.mark.integration
class TestActiveLearningIntegration:
    """Integration tests for active learning."""
    
    def test_full_pipeline(self, simple_model):
        """Test full active learning pipeline."""
        np.random.seed(42)
        
        # Generate synthetic Tc prediction problem
        X = np.random.randn(300, 10)
        y = X[:, 0] ** 2 + X[:, 1] + np.random.randn(300) * 2
        
        # Start with small labeled set
        X_labeled, X_unlabeled = X[:30], X[30:]
        y_labeled, y_unlabeled = y[:30], y[30:]
        
        # Create acquisition + diversity
        acq_fn = create_acquisition_function("ucb", kappa=2.0, maximize=True)
        div_selector = create_diversity_selector("greedy", alpha=0.5)
        
        # Create loop
        loop = ActiveLearningLoop(
            base_model=simple_model,
            acquisition_fn=acq_fn,
            diversity_selector=div_selector,
            budget=100,
            batch_size=20,
        )
        
        # Run
        result = loop.run(X_labeled, y_labeled, X_unlabeled, y_unlabeled, n_iterations=3)
        
        # Verify results
        assert len(result["X_train"]) == 30 + 3 * 20  # 90 total
        assert loop.budget_used_ == 60
        assert len(result["history"]) == 3
        
        # Verify model improves
        # Train final model
        simple_model.fit(result["X_train"], result["y_train"])
        
        # Should have reasonable R²
        from sklearn.metrics import r2_score
        
        y_test_pred = simple_model.predict(X[:30])  # Test on initial labeled set
        r2 = r2_score(y[:30], y_test_pred)
        
        assert r2 > -1.0  # At least better than predicting mean
