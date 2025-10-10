"""Tests for scripts/select_tests.py - Test selection based on EIG."""

import json
import tempfile
from pathlib import Path
import sys

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_import_select_tests():
    """Test that select_tests module can be imported."""
    try:
        import select_tests
        assert hasattr(select_tests, "__file__")
    except ImportError as e:
        pytest.skip(f"select_tests not available: {e}")


def test_select_top_k_tests():
    """Test selecting top K tests by EIG."""
    from select_tests import select_top_k
    
    rankings = [
        {"name": "test_a", "eig_score": 0.9, "rank": 1},
        {"name": "test_b", "eig_score": 0.7, "rank": 2},
        {"name": "test_c", "eig_score": 0.5, "rank": 3},
        {"name": "test_d", "eig_score": 0.3, "rank": 4},
    ]
    
    selected = select_top_k(rankings, k=2)
    
    assert len(selected) == 2
    assert selected[0]["name"] == "test_a"
    assert selected[1]["name"] == "test_b"


def test_select_by_threshold():
    """Test selecting tests above EIG threshold."""
    from select_tests import select_by_threshold
    
    rankings = [
        {"name": "test_a", "eig_score": 0.9},
        {"name": "test_b", "eig_score": 0.7},
        {"name": "test_c", "eig_score": 0.5},
        {"name": "test_d", "eig_score": 0.3},
    ]
    
    selected = select_by_threshold(rankings, threshold=0.6)
    
    assert len(selected) == 2
    assert all(t["eig_score"] >= 0.6 for t in selected)


def test_select_with_budget():
    """Test selecting tests within time budget."""
    from select_tests import select_with_budget
    
    rankings = [
        {"name": "test_a", "eig_score": 0.9, "duration_ms": 1000},
        {"name": "test_b", "eig_score": 0.8, "duration_ms": 2000},
        {"name": "test_c", "eig_score": 0.7, "duration_ms": 3000},
        {"name": "test_d", "eig_score": 0.6, "duration_ms": 1000},
    ]
    
    # Budget of 4000ms
    selected = select_with_budget(rankings, budget_ms=4000)
    
    # Should select tests maximizing EIG within budget
    total_duration = sum(t["duration_ms"] for t in selected)
    assert total_duration <= 4000
    
    # Should include highest EIG tests that fit
    assert any(t["name"] == "test_a" for t in selected)


def test_diversified_selection():
    """Test diverse test selection across domains."""
    from select_tests import select_diversified
    
    rankings = [
        {"name": "test_a", "eig_score": 0.9, "domain": "frontend"},
        {"name": "test_b", "eig_score": 0.85, "domain": "frontend"},
        {"name": "test_c", "eig_score": 0.8, "domain": "backend"},
        {"name": "test_d", "eig_score": 0.75, "domain": "backend"},
    ]
    
    # Select 2 tests with domain diversity
    selected = select_diversified(rankings, k=2, diversity_weight=0.5)
    
    # Should prefer one from each domain over two from same domain
    domains = [t["domain"] for t in selected]
    assert len(set(domains)) > 1


def test_greedy_selection():
    """Test greedy knapsack-style selection."""
    from select_tests import greedy_select
    
    rankings = [
        {"name": "test_a", "eig_score": 0.9, "duration_ms": 3000},
        {"name": "test_b", "eig_score": 0.7, "duration_ms": 1000},  # Best EIG/time ratio
        {"name": "test_c", "eig_score": 0.5, "duration_ms": 500},   # Second best ratio
    ]
    
    selected = greedy_select(rankings, budget_ms=2000)
    
    # Should select test_b and test_c (best ratios that fit)
    names = [t["name"] for t in selected]
    assert "test_b" in names
    assert "test_c" in names


def test_save_load_selected():
    """Test saving and loading selected tests."""
    from select_tests import save_selected, load_selected
    
    selected = [
        {"name": "test_a", "eig_score": 0.9},
        {"name": "test_b", "eig_score": 0.8},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "selected.json"
        save_selected(selected, output_file)
        
        loaded = load_selected(output_file)
        assert loaded == selected


def test_compute_selection_metrics():
    """Test computation of selection quality metrics."""
    from select_tests import compute_selection_metrics
    
    all_tests = [
        {"name": "test_a", "eig_score": 0.9, "duration_ms": 1000},
        {"name": "test_b", "eig_score": 0.7, "duration_ms": 2000},
        {"name": "test_c", "eig_score": 0.5, "duration_ms": 1000},
    ]
    
    selected = [all_tests[0], all_tests[1]]  # Selected top 2
    
    metrics = compute_selection_metrics(selected, all_tests)
    
    assert "coverage_pct" in metrics
    assert "total_eig" in metrics
    assert "avg_eig" in metrics
    assert "time_saved_pct" in metrics
    
    assert metrics["coverage_pct"] == pytest.approx(66.67, abs=0.1)  # 2/3 tests


def test_adaptive_selection():
    """Test adaptive selection based on historical performance."""
    from select_tests import adaptive_select
    
    # Historical failure data
    history = [
        {"run_id": 1, "test_a": "failed", "test_b": "passed", "test_c": "passed"},
        {"run_id": 2, "test_a": "failed", "test_b": "passed", "test_c": "failed"},
        {"run_id": 3, "test_a": "passed", "test_b": "passed", "test_c": "failed"},
    ]
    
    rankings = [
        {"name": "test_a", "eig_score": 0.7},
        {"name": "test_b", "eig_score": 0.5},
        {"name": "test_c", "eig_score": 0.8},
    ]
    
    selected = adaptive_select(rankings, history, k=2)
    
    # Should boost tests with recent failures
    names = [t["name"] for t in selected]
    assert "test_a" in names or "test_c" in names


def test_force_include_critical():
    """Test that critical tests are always included."""
    from select_tests import select_with_critical
    
    rankings = [
        {"name": "test_health", "eig_score": 0.1, "critical": True},
        {"name": "test_feature_a", "eig_score": 0.9, "critical": False},
        {"name": "test_feature_b", "eig_score": 0.8, "critical": False},
    ]
    
    selected = select_with_critical(rankings, k=2)
    
    # Should include critical test even with low EIG
    names = [t["name"] for t in selected]
    assert "test_health" in names


def test_selection_reproducibility():
    """Test that selection is reproducible with same inputs."""
    from select_tests import select_top_k
    
    rankings = [
        {"name": f"test_{i}", "eig_score": i/10} 
        for i in range(10, 0, -1)
    ]
    
    selected1 = select_top_k(rankings, k=5)
    selected2 = select_top_k(rankings, k=5)
    
    assert selected1 == selected2


def test_empty_rankings():
    """Test handling of empty rankings."""
    from select_tests import select_top_k
    
    selected = select_top_k([], k=5)
    assert selected == []


def test_k_larger_than_available():
    """Test selecting more tests than available."""
    from select_tests import select_top_k
    
    rankings = [
        {"name": "test_a", "eig_score": 0.9},
        {"name": "test_b", "eig_score": 0.8},
    ]
    
    selected = select_top_k(rankings, k=10)
    
    # Should return all available tests
    assert len(selected) == 2

