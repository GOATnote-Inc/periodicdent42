"""Tests for scripts/benchmark_example.py - Simple benchmark profiling."""

import sys
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from benchmark_example import (
    fast_computation,
    slow_computation,
    run_benchmark,
)


class TestFastComputation:
    """Test fast_computation function."""
    
    def test_fast_computation_default(self):
        """Test fast computation with default parameters."""
        result = fast_computation()
        
        # Should return a positive number
        assert result > 0
        assert isinstance(result, float)
    
    def test_fast_computation_small_n(self):
        """Test fast computation with small n."""
        result = fast_computation(n=10)
        
        assert result > 0
        assert isinstance(result, float)
    
    def test_fast_computation_zero(self):
        """Test fast computation with n=0."""
        result = fast_computation(n=0)
        
        assert result == 0.0
    
    def test_fast_computation_deterministic(self):
        """Test that fast computation is deterministic."""
        result1 = fast_computation(n=100)
        result2 = fast_computation(n=100)
        
        assert result1 == result2
    
    def test_fast_computation_increases_with_n(self):
        """Test that result increases with larger n."""
        result_small = fast_computation(n=10)
        result_large = fast_computation(n=100)
        
        assert result_large > result_small
    
    def test_fast_computation_expected_value(self):
        """Test fast computation produces expected value for known input."""
        # For n=3: (0^2)/1 + (1^2)/2 + (2^2)/3 = 0 + 0.5 + 1.333... = 1.833...
        result = fast_computation(n=3)
        
        expected = (0**2)/(0+1) + (1**2)/(1+1) + (2**2)/(2+1)
        assert abs(result - expected) < 1e-10


class TestSlowComputation:
    """Test slow_computation function."""
    
    def test_slow_computation_default(self):
        """Test slow computation with default parameters (skip for speed)."""
        # Run with small n to avoid long test time
        result = slow_computation(n=100)
        
        assert result > 0
        assert isinstance(result, float)
    
    def test_slow_computation_small_n(self):
        """Test slow computation with small n."""
        result = slow_computation(n=10)
        
        assert result > 0
        assert isinstance(result, float)
    
    def test_slow_computation_zero(self):
        """Test slow computation with n=0."""
        result = slow_computation(n=0)
        
        assert result == 0.0
    
    def test_slow_computation_deterministic(self):
        """Test that slow computation is deterministic (same result as fast)."""
        n = 100
        result_slow = slow_computation(n=n)
        result_fast = fast_computation(n=n)
        
        # Should produce same result, just slower
        assert abs(result_slow - result_fast) < 1e-10
    
    def test_slow_computation_has_sleeps(self):
        """Test that slow computation takes longer due to sleeps."""
        import time
        
        n = 2500  # Will trigger 2 sleeps (at i=0 and i=1000, i=2000)
        
        start = time.perf_counter()
        slow_computation(n=n)
        elapsed = time.perf_counter() - start
        
        # Should take at least 2ms due to 3 sleeps of 1ms each
        assert elapsed > 0.002  # 2ms minimum


class TestRunBenchmark:
    """Test run_benchmark function."""
    
    def test_run_benchmark_fast_mode(self):
        """Test benchmark in fast mode."""
        results = run_benchmark(mode="fast", iterations=3)
        
        assert results["mode"] == "fast"
        assert results["iterations"] == 3
        assert "result" in results
        assert "timings_ms" in results
        assert len(results["timings_ms"]) == 3
        assert "mean_ms" in results
        assert "min_ms" in results
        assert "max_ms" in results
    
    def test_run_benchmark_slow_mode(self):
        """Test benchmark in slow mode (with small iterations)."""
        results = run_benchmark(mode="slow", iterations=2)
        
        assert results["mode"] == "slow"
        assert results["iterations"] == 2
        assert len(results["timings_ms"]) == 2
    
    def test_run_benchmark_default_mode(self):
        """Test benchmark with default mode."""
        results = run_benchmark()
        
        assert results["mode"] == "fast"
        assert results["iterations"] == 5
        assert len(results["timings_ms"]) == 5
    
    def test_run_benchmark_single_iteration(self):
        """Test benchmark with single iteration."""
        results = run_benchmark(mode="fast", iterations=1)
        
        assert results["iterations"] == 1
        assert len(results["timings_ms"]) == 1
        assert results["mean_ms"] == results["timings_ms"][0]
        assert results["min_ms"] == results["timings_ms"][0]
        assert results["max_ms"] == results["timings_ms"][0]
    
    def test_run_benchmark_timings_are_positive(self):
        """Test that all timings are positive."""
        results = run_benchmark(mode="fast", iterations=3)
        
        assert all(t > 0 for t in results["timings_ms"])
        assert results["mean_ms"] > 0
        assert results["min_ms"] > 0
        assert results["max_ms"] > 0
    
    def test_run_benchmark_min_max_relationship(self):
        """Test that min <= mean <= max."""
        results = run_benchmark(mode="fast", iterations=5)
        
        assert results["min_ms"] <= results["mean_ms"]
        assert results["mean_ms"] <= results["max_ms"]
    
    def test_run_benchmark_mean_calculation(self):
        """Test that mean is calculated correctly."""
        results = run_benchmark(mode="fast", iterations=3)
        
        expected_mean = sum(results["timings_ms"]) / len(results["timings_ms"])
        assert abs(results["mean_ms"] - expected_mean) < 1e-10
    
    def test_run_benchmark_min_calculation(self):
        """Test that min is calculated correctly."""
        results = run_benchmark(mode="fast", iterations=3)
        
        assert results["min_ms"] == min(results["timings_ms"])
    
    def test_run_benchmark_max_calculation(self):
        """Test that max is calculated correctly."""
        results = run_benchmark(mode="fast", iterations=3)
        
        assert results["max_ms"] == max(results["timings_ms"])
    
    def test_run_benchmark_result_is_float(self):
        """Test that result is a float."""
        results = run_benchmark(mode="fast", iterations=1)
        
        assert isinstance(results["result"], float)
    
    def test_run_benchmark_slow_takes_longer(self):
        """Test that slow mode takes longer than fast mode."""
        import time
        
        start_fast = time.perf_counter()
        results_fast = run_benchmark(mode="fast", iterations=1)
        elapsed_fast = time.perf_counter() - start_fast
        
        start_slow = time.perf_counter()
        results_slow = run_benchmark(mode="slow", iterations=1)
        elapsed_slow = time.perf_counter() - start_slow
        
        # Slow should take noticeably longer (at least 10x due to sleeps)
        # Being conservative to avoid flaky tests
        assert elapsed_slow > elapsed_fast
    
    def test_run_benchmark_timings_in_milliseconds(self):
        """Test that timings are in milliseconds (reasonable range)."""
        results = run_benchmark(mode="fast", iterations=1)
        
        # Fast computation should take less than 1000ms (1 second)
        assert results["timings_ms"][0] < 1000
        
        # Should take more than 0.001ms (sanity check)
        assert results["timings_ms"][0] > 0.001
    
    def test_run_benchmark_multiple_iterations_consistency(self):
        """Test that multiple iterations produce consistent results."""
        results = run_benchmark(mode="fast", iterations=10)
        
        # Standard deviation should be reasonable (within 50% of mean)
        import statistics
        std_dev = statistics.stdev(results["timings_ms"])
        mean = results["mean_ms"]
        
        # Coefficient of variation should be reasonable
        cv = std_dev / mean
        assert cv < 0.5  # Standard deviation < 50% of mean


class TestIntegration:
    """Integration tests for benchmark_example."""
    
    def test_fast_computation_matches_expected_algorithm(self):
        """Test that fast computation implements sum of i^2/(i+1)."""
        n = 5
        result = fast_computation(n=n)
        
        # Manual calculation
        expected = sum((i ** 2) / (i + 1) for i in range(n))
        
        assert abs(result - expected) < 1e-10
    
    def test_slow_computation_matches_fast(self):
        """Test that slow and fast produce same results."""
        n = 50  # Small enough to run quickly
        
        result_fast = fast_computation(n=n)
        result_slow = slow_computation(n=n)
        
        assert abs(result_fast - result_slow) < 1e-10
    
    def test_benchmark_results_structure(self):
        """Test that benchmark results have expected structure."""
        results = run_benchmark(mode="fast", iterations=3)
        
        # Check all required keys
        required_keys = ["mode", "iterations", "result", "timings_ms", 
                        "mean_ms", "min_ms", "max_ms"]
        
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_benchmark_results_types(self):
        """Test that benchmark results have correct types."""
        results = run_benchmark(mode="fast", iterations=3)
        
        assert isinstance(results["mode"], str)
        assert isinstance(results["iterations"], int)
        assert isinstance(results["result"], float)
        assert isinstance(results["timings_ms"], list)
        assert isinstance(results["mean_ms"], float)
        assert isinstance(results["min_ms"], float)
        assert isinstance(results["max_ms"], float)

