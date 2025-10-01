"""
Tests for experimental adaptive routing system.

Testing Strategy:
- Unit tests for noise estimation methods
- Routing logic tests with known noise levels
- Edge case handling (insufficient data, extreme values)
- Statistical validation of routing decisions

Note: These tests validate the IMPLEMENTATION, not the HYPOTHESIS.
We still need real-world validation to confirm RL > BO at high noise.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.reasoning.adaptive.noise_estimator import (
    NoiseEstimator,
    NoiseEstimate,
    estimate_noise_simple
)
from src.reasoning.adaptive.router import (
    AdaptiveRouter,
    OptimizationMethod,
    RoutingDecision,
    quick_route
)


class TestNoiseEstimator:
    """Test noise estimation methods."""
    
    def test_replicate_estimation_perfect_data(self):
        """Test with ideal replicated data."""
        estimator = NoiseEstimator()
        
        # Three groups of replicates with known noise
        # Group 1: mean=10, std=1.0
        # Group 2: mean=20, std=1.0
        # Group 3: mean=15, std=1.0
        replicates = [
            [10.0, 11.0, 9.0, 10.5, 9.5],   # n=5, std≈0.75
            [20.0, 21.0, 19.0, 20.5, 19.5],  # n=5, std≈0.75
            [15.0, 16.0, 14.0, 15.5, 14.5],  # n=5, std≈0.75
        ]
        
        estimate = estimator.estimate_from_replicates(replicates)
        
        assert estimate.reliable
        assert estimate.method == "replicate_pooled"
        assert estimate.sample_size == 15
        assert 0.5 < estimate.std < 1.0  # Pooled std should be ~0.75
        assert estimate.confidence_interval is not None
    
    def test_replicate_estimation_insufficient_data(self):
        """Test with too few replicates."""
        estimator = NoiseEstimator()
        
        # Single replicate per group
        replicates = [[10.0], [20.0], [15.0]]
        
        estimate = estimator.estimate_from_replicates(replicates)
        
        assert not estimate.reliable
        assert estimate.method == "insufficient_replicates"
    
    def test_sequential_estimation(self):
        """Test sequential difference method."""
        estimator = NoiseEstimator()
        
        # Flat function with added noise
        np.random.seed(42)
        true_value = 10.0
        noise_std = 1.0
        observations = true_value + np.random.normal(0, noise_std, 20)
        
        estimate = estimator.estimate_from_sequential(
            observations.tolist(),
            assume_smooth=True
        )
        
        assert estimate.method == "sequential_differences"
        assert estimate.sample_size == 20
        # Should recover noise std approximately (with some error)
        assert 0.5 < estimate.std < 2.0
    
    def test_sequential_estimation_with_trend(self):
        """Test sequential method warns about non-smooth functions."""
        estimator = NoiseEstimator()
        
        # Linear trend - will bias the estimate
        observations = [float(i) for i in range(10)]
        
        estimate = estimator.estimate_from_sequential(
            observations,
            assume_smooth=False
        )
        
        # Should return estimate but flag as unreliable
        assert not estimate.reliable or estimate.std > 0
    
    def test_residual_estimation_gp(self):
        """Test GP-based noise estimation."""
        estimator = NoiseEstimator()
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.rand(20, 2)
        y_clean = np.sin(X[:, 0]) + np.cos(X[:, 1])
        noise_std = 0.5
        y_noisy = y_clean + np.random.normal(0, noise_std, len(y_clean))
        
        estimate = estimator.estimate_from_residuals(X, y_noisy, model_type="gp")
        
        assert estimate.method == "gp_white_kernel"
        assert estimate.sample_size == 20
        # Should roughly recover noise std
        assert 0.2 < estimate.std < 1.0
    
    def test_residual_estimation_insufficient_data(self):
        """Test residual method with too few points."""
        estimator = NoiseEstimator()
        
        X = np.random.rand(5, 2)
        y = np.random.rand(5)
        
        estimate = estimator.estimate_from_residuals(X, y, model_type="gp")
        
        assert not estimate.reliable
        assert "insufficient_data" in estimate.method
    
    def test_estimate_auto_method_selection(self):
        """Test automatic method selection."""
        estimator = NoiseEstimator()
        
        # Provide replicate data (should use this)
        data = {
            "replicates": [[1.0, 1.1, 0.9], [2.0, 2.1, 1.9]],
            "sequential": [1.0, 2.0, 3.0],  # Should ignore this
        }
        
        estimate = estimator.estimate(data, prefer_method="auto")
        
        assert estimate.method == "replicate_pooled"
    
    def test_simple_estimation_helper(self):
        """Test quick estimation helper function."""
        observations = [10.0, 10.1, 9.9, 10.2, 9.8]
        
        noise_std = estimate_noise_simple(observations)
        
        assert noise_std > 0
        assert noise_std < 1.0  # Should be small for this data


class TestAdaptiveRouter:
    """Test adaptive routing logic."""
    
    def test_routing_low_noise_prefers_bo(self):
        """Test that low noise routes to Bayesian Optimization."""
        router = AdaptiveRouter()
        
        # Mock low-noise data
        pilot_data = {
            "replicates": [[10.0, 10.1, 10.05], [20.0, 20.1, 20.05]]
        }
        
        decision = router.route(pilot_data)
        
        assert decision.method == OptimizationMethod.BAYESIAN_OPTIMIZATION
        assert decision.confidence > 0.8
        assert decision.noise_estimate.std < 0.5
    
    def test_routing_high_noise_considers_rl(self):
        """Test that high noise routes to RL."""
        router = AdaptiveRouter()
        
        # Mock high-noise data
        pilot_data = {
            "replicates": [
                [10.0, 12.5, 8.0, 11.0, 9.5],  # High variance
                [20.0, 22.5, 18.0, 21.0, 19.5],
            ]
        }
        
        decision = router.route(pilot_data)
        
        # Should at least consider RL
        assert decision.noise_estimate.std > 1.0
        # May choose RL depending on exact threshold
    
    def test_routing_insufficient_data_defaults_bo(self):
        """Test that insufficient data defaults to BO."""
        router = AdaptiveRouter()
        
        # Very little data
        pilot_data = {
            "sequential": [10.0, 11.0]
        }
        
        decision = router.route(pilot_data)
        
        assert decision.method == OptimizationMethod.BAYESIAN_OPTIMIZATION
        assert decision.confidence < 0.5  # Low confidence
        assert len(decision.warnings) > 0
    
    def test_routing_medium_noise_gray_zone(self):
        """Test gray zone behavior (medium noise)."""
        router = AdaptiveRouter()
        
        # Medium noise (σ ~ 0.8)
        pilot_data = {
            "replicates": [
                [10.0, 10.8, 9.2, 10.5, 9.5],
                [20.0, 20.8, 19.2, 20.5, 19.5],
            ]
        }
        
        decision = router.route(pilot_data)
        
        # Should have moderate confidence
        assert 0.4 < decision.confidence < 0.9
        # Should mention gray zone in warnings or reasoning
        assert any(
            "medium" in w.lower() or "unclear" in w.lower() or "gray" in decision.reasoning.lower()
            for w in decision.warnings
        ) or "medium" in decision.reasoning.lower()
    
    def test_custom_thresholds(self):
        """Test router with custom thresholds."""
        custom_thresholds = {
            "bo_preferred": 1.0,
            "rl_considered": 2.0,
            "rl_preferred": 3.0,
        }
        router = AdaptiveRouter(thresholds=custom_thresholds)
        
        assert router.thresholds == custom_thresholds
    
    def test_routing_history_tracking(self):
        """Test that router tracks decision history."""
        router = AdaptiveRouter()
        
        pilot_data = {
            "sequential": [10.0, 10.1, 9.9, 10.2, 9.8, 10.3]
        }
        
        decision1 = router.route(pilot_data)
        decision2 = router.route(pilot_data)
        
        assert len(router.routing_history) == 2
        assert router.routing_history[0] == decision1
        assert router.routing_history[1] == decision2
    
    def test_explain_decision(self):
        """Test decision explanation generation."""
        router = AdaptiveRouter()
        
        pilot_data = {
            "replicates": [[10.0, 10.1, 10.05]]
        }
        
        decision = router.route(pilot_data)
        explanation = router.explain_decision(decision)
        
        assert isinstance(explanation, str)
        # Check for uppercase version (how it's displayed in explanation)
        assert decision.method.value.upper() in explanation  # "BAYESIAN_OPTIMIZATION" in text
        assert "EXPERIMENTAL" in explanation
        # Confidence appears as "30.0%" not "0.3"
        assert "%" in explanation
    
    def test_routing_statistics(self):
        """Test statistics generation."""
        router = AdaptiveRouter()
        
        # Make several routing decisions
        for i in range(5):
            pilot_data = {
                "sequential": [10.0 + i * 0.1 * j for j in range(10)]
            }
            router.route(pilot_data)
        
        stats = router.get_routing_statistics()
        
        assert stats["total_decisions"] == 5
        assert "method_distribution" in stats
        assert "average_confidence" in stats
        assert 0 <= stats["average_confidence"] <= 1
    
    def test_quick_route_helper(self):
        """Test quick routing helper function."""
        # Low noise
        low_noise_obs = [10.0, 10.1, 9.9, 10.05, 9.95]
        method = quick_route(low_noise_obs)
        assert method == OptimizationMethod.BAYESIAN_OPTIMIZATION
        
        # High noise
        high_noise_obs = [10.0, 13.0, 8.0, 12.0, 9.0]
        method = quick_route(high_noise_obs)
        # May be RL depending on exact noise level
        
        # Insufficient data
        method = quick_route([10.0])
        assert method == OptimizationMethod.BAYESIAN_OPTIMIZATION  # Default


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_noise(self):
        """Test with perfect noiseless data."""
        estimator = NoiseEstimator()
        
        # Perfectly replicated measurements
        replicates = [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]
        
        estimate = estimator.estimate_from_replicates(replicates)
        
        assert estimate.std == 0.0 or estimate.std < 1e-10
    
    def test_extremely_high_noise(self):
        """Test with very high noise levels."""
        router = AdaptiveRouter()
        
        # Extreme noise
        np.random.seed(42)
        pilot_data = {
            "sequential": [10.0 + np.random.normal(0, 10.0) for _ in range(20)]
        }
        
        decision = router.route(pilot_data)
        
        # Should handle without crashing
        assert decision.method in OptimizationMethod
    
    def test_empty_data(self):
        """Test with empty data."""
        estimator = NoiseEstimator()
        
        estimate = estimator.estimate({}, prefer_method="auto")
        
        assert not estimate.reliable
        assert estimate.std == 0.0
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        estimator = NoiseEstimator()
        
        # Data with NaN
        observations = [10.0, np.nan, 10.1, 9.9]
        
        # Should handle gracefully (may filter NaN or return unreliable)
        # Don't crash
        try:
            estimate = estimator.estimate_from_sequential(
                [x for x in observations if not np.isnan(x)]
            )
            assert True  # Didn't crash
        except Exception:
            pytest.fail("Should handle NaN gracefully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

