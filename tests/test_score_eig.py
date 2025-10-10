"""Tests for scripts/score_eig.py - Expected Information Gain scoring."""

import json
import numpy as np
from pathlib import Path
import sys
import tempfile

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_import_score_eig():
    """Test that score_eig module can be imported."""
    try:
        import score_eig
        assert hasattr(score_eig, "__file__")
    except ImportError as e:
        pytest.skip(f"score_eig not available: {e}")


def test_compute_entropy():
    """Test entropy computation."""
    from score_eig import compute_entropy
    
    # Uniform distribution should have max entropy
    p_uniform = np.array([0.5, 0.5])
    H_uniform = compute_entropy(p_uniform)
    assert H_uniform == pytest.approx(1.0, abs=0.01)
    
    # Deterministic distribution should have zero entropy
    p_det = np.array([1.0, 0.0])
    H_det = compute_entropy(p_det)
    assert H_det == pytest.approx(0.0, abs=0.01)


def test_compute_conditional_entropy():
    """Test conditional entropy computation."""
    from score_eig import compute_conditional_entropy
    
    # P(Y|X)
    p_y_given_x = np.array([[0.9, 0.1], [0.2, 0.8]])  # 2 values of X, 2 of Y
    p_x = np.array([0.5, 0.5])
    
    H_conditional = compute_conditional_entropy(p_y_given_x, p_x)
    
    # Should be positive
    assert H_conditional > 0
    # Should be less than max entropy
    assert H_conditional < 1.0


def test_compute_mutual_information():
    """Test mutual information computation."""
    from score_eig import compute_mutual_information
    
    # Perfect correlation: I(X;Y) = H(X) = H(Y)
    p_xy_perfect = np.array([[0.5, 0.0], [0.0, 0.5]])
    I_perfect = compute_mutual_information(p_xy_perfect)
    assert I_perfect == pytest.approx(1.0, abs=0.01)
    
    # No correlation: I(X;Y) = 0
    p_xy_indep = np.array([[0.25, 0.25], [0.25, 0.25]])
    I_indep = compute_mutual_information(p_xy_indep)
    assert I_indep == pytest.approx(0.0, abs=0.01)


def test_expected_information_gain():
    """Test EIG computation for test selection."""
    from score_eig import compute_eig
    
    # Prior failure probabilities for tests
    p_failures = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    eig_scores = compute_eig(p_failures)
    
    # Should have one score per test
    assert len(eig_scores) == 5
    
    # All scores should be non-negative
    assert all(score >= 0 for score in eig_scores)
    
    # Tests near p=0.5 should have highest EIG (most informative)
    max_eig_idx = np.argmax(eig_scores)
    assert max_eig_idx == 2  # p=0.5


def test_eig_symmetry():
    """Test that EIG is symmetric around p=0.5."""
    from score_eig import compute_eig
    
    p1 = np.array([0.2])
    p2 = np.array([0.8])
    
    eig1 = compute_eig(p1)[0]
    eig2 = compute_eig(p2)[0]
    
    assert eig1 == pytest.approx(eig2, abs=0.01)


def test_score_test_suite():
    """Test scoring entire test suite."""
    from score_eig import score_test_suite
    
    # Mock test suite with failure predictions
    tests = [
        {"name": "test_a", "failure_prob": 0.1},
        {"name": "test_b", "failure_prob": 0.5},
        {"name": "test_c", "failure_prob": 0.9},
    ]
    
    scored_tests = score_test_suite(tests)
    
    assert len(scored_tests) == 3
    for test in scored_tests:
        assert "eig_score" in test
        assert test["eig_score"] >= 0
    
    # Test with p=0.5 should have highest score
    test_b = [t for t in scored_tests if t["name"] == "test_b"][0]
    assert test_b["eig_score"] == max(t["eig_score"] for t in scored_tests)


def test_rank_tests_by_eig():
    """Test that tests are correctly ranked by EIG."""
    from score_eig import rank_tests_by_eig
    
    tests = [
        {"name": "test_a", "failure_prob": 0.1, "eig_score": 0.5},
        {"name": "test_b", "failure_prob": 0.5, "eig_score": 1.0},
        {"name": "test_c", "failure_prob": 0.9, "eig_score": 0.5},
    ]
    
    ranked = rank_tests_by_eig(tests)
    
    # Should be sorted by EIG (descending)
    assert ranked[0]["name"] == "test_b"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3


def test_save_load_rankings():
    """Test saving and loading EIG rankings."""
    from score_eig import save_rankings, load_rankings
    
    rankings = [
        {"name": "test_a", "eig_score": 0.8, "rank": 1},
        {"name": "test_b", "eig_score": 0.5, "rank": 2},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        rankings_file = Path(tmpdir) / "rankings.json"
        save_rankings(rankings, rankings_file)
        
        loaded = load_rankings(rankings_file)
        assert loaded == rankings


def test_eig_with_historical_data():
    """Test EIG computation with historical failure data."""
    from score_eig import compute_eig_from_history
    
    # Historical test results: [pass, fail, pass, fail, pass]
    history = {
        "test_a": [1, 0, 1, 0, 1],  # 60% pass rate
        "test_b": [1, 1, 1, 1, 1],  # 100% pass rate  
        "test_c": [1, 0, 1, 1, 0],  # 60% pass rate but different pattern
    }
    
    eig_scores = compute_eig_from_history(history)
    
    assert "test_a" in eig_scores
    assert "test_b" in eig_scores
    assert "test_c" in eig_scores
    
    # Deterministic test should have low EIG
    assert eig_scores["test_b"] < eig_scores["test_a"]


def test_compute_test_correlation():
    """Test correlation between test outcomes."""
    from score_eig import compute_test_correlation
    
    # Test outcomes over multiple runs
    test_results = {
        "test_a": [1, 1, 0, 0, 1],
        "test_b": [1, 1, 0, 0, 1],  # Perfectly correlated with test_a
        "test_c": [0, 0, 1, 1, 0],  # Negatively correlated with test_a
    }
    
    corr = compute_test_correlation(test_results)
    
    # test_a and test_b should be highly correlated
    assert corr["test_a"]["test_b"] > 0.9
    
    # test_a and test_c should be negatively correlated
    assert corr["test_a"]["test_c"] < -0.9


def test_adjust_eig_for_cost():
    """Test cost-adjusted EIG computation."""
    from score_eig import adjust_eig_for_cost
    
    tests = [
        {"name": "test_fast", "eig_score": 0.5, "duration_ms": 100},
        {"name": "test_slow", "eig_score": 0.5, "duration_ms": 10000},
    ]
    
    adjusted = adjust_eig_for_cost(tests)
    
    # Fast test should have higher cost-adjusted score
    fast = [t for t in adjusted if t["name"] == "test_fast"][0]
    slow = [t for t in adjusted if t["name"] == "test_slow"][0]
    
    assert fast["cost_adjusted_eig"] > slow["cost_adjusted_eig"]

