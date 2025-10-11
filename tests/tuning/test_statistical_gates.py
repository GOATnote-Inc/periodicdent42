"""
Tests for statistical validation gates (Bootstrap CI + MAE decision aid).

© 2025 GOATnote Autonomous Research Lab Initiative
"""
import numpy as np
import pandas as pd
import pytest
from app.src.htc.tuning.statistical_gates import compute_delta_mape_ci


def test_bootstrap_ci_determinism():
    """Same seed should give identical CIs."""
    # Create sample data
    current = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb'],
        'tier': ['A', 'A', 'A'],
        'rel_error_pct': [10.0, 15.0, 20.0],
        'abs_error': [1.0, 1.5, 2.0]
    })
    baseline = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb'],
        'rel_error_pct': [12.0, 18.0, 22.0],
        'abs_error': [1.2, 1.8, 2.2]
    })
    
    result1 = compute_delta_mape_ci(current, baseline, n_bootstrap=100, seed=42)
    result2 = compute_delta_mape_ci(current, baseline, n_bootstrap=100, seed=42)
    
    assert result1['delta_mape_mean'] == result2['delta_mape_mean']
    assert result1['delta_mape_ci_lower'] == result2['delta_mape_ci_lower']
    assert result1['delta_mape_ci_upper'] == result2['delta_mape_ci_upper']


def test_improvement_detected():
    """Should detect improvement when current < baseline."""
    current = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb', 'V'],
        'tier': ['A', 'A', 'A', 'A'],
        'rel_error_pct': [5.0, 6.0, 7.0, 8.0],
        'abs_error': [0.5, 0.6, 0.7, 0.8]
    })
    baseline = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb', 'V'],
        'rel_error_pct': [10.0, 12.0, 14.0, 16.0],
        'abs_error': [1.0, 1.2, 1.4, 1.6]
    })
    
    result = compute_delta_mape_ci(current, baseline, n_bootstrap=1000, seed=42)
    
    # MAPE decreased (improvement)
    assert result['delta_mape_mean'] < 0
    assert result['delta_mape_ci_upper'] < 0  # CI excludes zero
    assert result['delta_mape_excludes_zero'] is True
    
    # MAE decreased (improvement)
    assert result['delta_mae_mean'] < 0
    assert result['delta_mae_ci_upper'] < 0
    assert result['delta_mae_excludes_zero'] is True


def test_no_improvement_detected():
    """Should detect no improvement when current ≈ baseline."""
    current = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb'],
        'tier': ['A', 'A', 'A'],
        'rel_error_pct': [10.0, 11.0, 12.0],
        'abs_error': [1.0, 1.1, 1.2]
    })
    baseline = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb'],
        'rel_error_pct': [9.5, 11.5, 12.0],
        'abs_error': [0.95, 1.15, 1.20]
    })
    
    result = compute_delta_mape_ci(current, baseline, n_bootstrap=1000, seed=42)
    
    # MAPE change is small, CI includes zero
    assert result['delta_mape_excludes_zero'] is False


def test_stratified_sampling():
    """Stratified sampling should preserve tier ratios."""
    # Create data with unbalanced tiers
    current = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb', 'NbN', 'TiN'],
        'tier': ['A', 'A', 'A', 'B', 'B'],
        'rel_error_pct': [5.0, 6.0, 7.0, 15.0, 16.0],
        'abs_error': [0.5, 0.6, 0.7, 1.5, 1.6]
    })
    baseline = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb', 'NbN', 'TiN'],
        'rel_error_pct': [10.0, 11.0, 12.0, 20.0, 21.0],
        'abs_error': [1.0, 1.1, 1.2, 2.0, 2.1]
    })
    
    # Stratified
    result_stratified = compute_delta_mape_ci(
        current, baseline, n_bootstrap=1000, seed=42, stratify_by_tier=True
    )
    
    # Non-stratified
    result_non_stratified = compute_delta_mape_ci(
        current, baseline, n_bootstrap=1000, seed=42, stratify_by_tier=False
    )
    
    # Both should show improvement, but CI widths may differ
    assert result_stratified['delta_mape_mean'] < 0
    assert result_non_stratified['delta_mape_mean'] < 0


def test_ci_width_increases_with_fewer_samples():
    """Fewer materials should give wider CIs."""
    # Large sample
    np.random.seed(42)
    current_large = pd.DataFrame({
        'material': [f'M{i}' for i in range(20)],
        'tier': ['A'] * 20,
        'rel_error_pct': np.random.normal(10, 2, 20),
        'abs_error': np.random.normal(1, 0.2, 20)
    })
    baseline_large = pd.DataFrame({
        'material': [f'M{i}' for i in range(20)],
        'rel_error_pct': np.random.normal(15, 2, 20),
        'abs_error': np.random.normal(1.5, 0.2, 20)
    })
    
    # Small sample
    current_small = current_large.head(5).copy()
    baseline_small = baseline_large.head(5).copy()
    
    result_large = compute_delta_mape_ci(current_large, baseline_large, n_bootstrap=1000, seed=42)
    result_small = compute_delta_mape_ci(current_small, baseline_small, n_bootstrap=1000, seed=42)
    
    ci_width_large = result_large['delta_mape_ci_upper'] - result_large['delta_mape_ci_lower']
    ci_width_small = result_small['delta_mape_ci_upper'] - result_small['delta_mape_ci_lower']
    
    # Small sample should have wider CI
    assert ci_width_small > ci_width_large


def test_p_value_calculation():
    """p-value should reflect improvement probability."""
    # Strong improvement (p-value near 0)
    current_strong = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb', 'V'],
        'tier': ['A', 'A', 'A', 'A'],
        'rel_error_pct': [5.0, 6.0, 7.0, 8.0],
        'abs_error': [0.5, 0.6, 0.7, 0.8]
    })
    baseline_strong = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb', 'V'],
        'rel_error_pct': [20.0, 22.0, 24.0, 26.0],
        'abs_error': [2.0, 2.2, 2.4, 2.6]
    })
    
    result_strong = compute_delta_mape_ci(current_strong, baseline_strong, n_bootstrap=1000, seed=42)
    
    # Strong improvement should have p-value near 0
    assert result_strong['delta_mape_p_value'] < 0.05


def test_missing_data_handling():
    """Should handle partial overlap in materials."""
    # Current has Al, Pb, Nb
    current = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb'],
        'tier': ['A', 'A', 'A'],
        'rel_error_pct': [10.0, 11.0, 12.0],
        'abs_error': [1.0, 1.1, 1.2]
    })
    # Baseline has Al, Nb, V (missing Pb)
    baseline = pd.DataFrame({
        'material': ['Al', 'Nb', 'V'],
        'rel_error_pct': [15.0, 17.0, 18.0],
        'abs_error': [1.5, 1.7, 1.8]
    })
    
    # Should only use overlapping materials (Al, Nb)
    result = compute_delta_mape_ci(current, baseline, n_bootstrap=100, seed=42)
    
    # Should only have 2 materials (Al, Nb)
    assert result['n_materials'] == 2
    assert result['delta_mape_mean'] < 0  # Both improved


def test_no_overlapping_materials():
    """Should raise error when no materials overlap."""
    current = pd.DataFrame({
        'material': ['Al', 'Pb'],
        'tier': ['A', 'A'],
        'rel_error_pct': [10.0, 12.0],
        'abs_error': [1.0, 1.2]
    })
    baseline = pd.DataFrame({
        'material': ['Nb', 'V'],
        'rel_error_pct': [15.0, 16.0],
        'abs_error': [1.5, 1.6]
    })
    
    with pytest.raises(ValueError, match="No overlapping materials"):
        compute_delta_mape_ci(current, baseline, n_bootstrap=100, seed=42)


def test_ci_includes_mean():
    """CI should always include the mean."""
    current = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb', 'V'],
        'tier': ['A', 'A', 'A', 'A'],
        'rel_error_pct': [10.0, 11.0, 12.0, 13.0],
        'abs_error': [1.0, 1.1, 1.2, 1.3]
    })
    baseline = pd.DataFrame({
        'material': ['Al', 'Pb', 'Nb', 'V'],
        'rel_error_pct': [15.0, 16.0, 17.0, 18.0],
        'abs_error': [1.5, 1.6, 1.7, 1.8]
    })
    
    result = compute_delta_mape_ci(current, baseline, n_bootstrap=1000, seed=42)
    
    # CI should bracket the mean
    assert result['delta_mape_ci_lower'] <= result['delta_mape_mean'] <= result['delta_mape_ci_upper']
    assert result['delta_mae_ci_lower'] <= result['delta_mae_mean'] <= result['delta_mae_ci_upper']


def test_alpha_parameter():
    """Different alpha should give different CI widths."""
    # More materials with variability
    np.random.seed(42)
    n = 15
    current = pd.DataFrame({
        'material': [f'M{i}' for i in range(n)],
        'tier': ['A'] * n,
        'rel_error_pct': np.random.normal(10, 3, n),  # More variance
        'abs_error': np.random.normal(1.0, 0.3, n)
    })
    baseline = pd.DataFrame({
        'material': [f'M{i}' for i in range(n)],
        'rel_error_pct': np.random.normal(15, 4, n),  # More variance
        'abs_error': np.random.normal(1.5, 0.4, n)
    })
    
    # 95% CI (α=0.05)
    result_95 = compute_delta_mape_ci(current, baseline, n_bootstrap=1000, seed=42, alpha=0.05)
    
    # 99% CI (α=0.01) should be wider
    result_99 = compute_delta_mape_ci(current, baseline, n_bootstrap=1000, seed=42, alpha=0.01)
    
    ci_width_95 = result_95['delta_mape_ci_upper'] - result_95['delta_mape_ci_lower']
    ci_width_99 = result_99['delta_mape_ci_upper'] - result_99['delta_mape_ci_lower']
    
    # 99% CI should be wider (more conservative)
    assert ci_width_99 > ci_width_95, f"99% CI width ({ci_width_99:.3f}) should be > 95% CI width ({ci_width_95:.3f})"

