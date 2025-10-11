"""
Tests for Allen-Dynes exact f₁/f₂ corrections.

© 2025 GOATnote Autonomous Research Lab Initiative
"""
import numpy as np
import pytest
from app.src.htc.tuning.allen_dynes_corrections import (
    compute_f1_factor,
    compute_f2_factor,
    allen_dynes_corrected_tc,
    get_omega2_ratio,
    OMEGA2_RATIO_DB,
)


def test_f1_monotonicity_in_lambda():
    """f₁ should increase with λ (μ* fixed)."""
    mu = 0.13
    f1_values = [compute_f1_factor(lam, mu) for lam in [0.5, 1.0, 1.5, 2.0]]
    assert all(f1_values[i] < f1_values[i+1] for i in range(len(f1_values)-1)), \
        f"f1 monotonicity violated: {f1_values}"


def test_f2_bounds():
    """f₂ should be ≥ 1.0 and ≤ 1.5."""
    for lam in [0.5, 1.0, 1.5, 2.0]:
        for mu in [0.10, 0.13, 0.15]:
            for ratio in [1.1, 1.5, 2.0, 2.5]:
                f2 = compute_f2_factor(lam, mu, ratio)
                assert 1.0 <= f2 <= 1.5, f"f2={f2} out of bounds for λ={lam}, μ*={mu}, r={ratio}"


def test_mu_star_monotonicity():
    """Increasing μ* should DECREASE Tc (λ, ω_log fixed)."""
    omega_log = 300.0  # K
    lam = 1.0
    ratio = 1.5
    
    tc_values = []
    for mu in [0.08, 0.10, 0.13, 0.16, 0.20]:
        result = allen_dynes_corrected_tc(lam, mu, omega_log, ratio)
        tc_values.append(result['Tc'])
    
    # Higher μ* → lower Tc
    assert all(tc_values[i] > tc_values[i+1] for i in range(len(tc_values)-1)), \
        f"μ* monotonicity violated: Tc={tc_values}"


def test_allen_dynes_tc_increases_with_lambda():
    """Tc should increase with λ (μ*, ω_log fixed)."""
    omega_log = 300.0  # K
    mu = 0.13
    ratio = 1.5
    
    tc_values = []
    for lam in [0.6, 0.8, 1.0, 1.2]:
        result = allen_dynes_corrected_tc(lam, mu, omega_log, ratio)
        tc_values.append(result['Tc'])
    
    assert all(tc_values[i] < tc_values[i+1] for i in range(len(tc_values)-1)), \
        f"λ monotonicity violated: Tc={tc_values}"


def test_allen_dynes_warning_for_large_lambda():
    """Should warn when λ > 1.5."""
    result = allen_dynes_corrected_tc(1.8, 0.13, 300.0, 1.5)
    assert any('extrapolation' in w.lower() for w in result['warnings']), \
        f"Missing extrapolation warning for λ=1.8: {result['warnings']}"


def test_omega2_ratio_sanity():
    """All ratios should be ≥ 1.0, including default fallback."""
    assert get_omega2_ratio('default') >= 1.0
    assert get_omega2_ratio('Al') >= 1.0
    assert get_omega2_ratio('MgB2') >= 1.0
    # Micro-edit #8a: Test unknown material returns default
    assert get_omega2_ratio('nonexistent_material') == OMEGA2_RATIO_DB['default']
    assert get_omega2_ratio('nonexistent_material') >= 1.0


def test_omega_log_guard():
    """Should raise ValueError for omega_log ≤ 0."""
    with pytest.raises(ValueError, match="omega_log must be > 0"):
        allen_dynes_corrected_tc(1.0, 0.13, 0.0, 1.5)
    with pytest.raises(ValueError, match="omega_log must be > 0"):
        allen_dynes_corrected_tc(1.0, 0.13, -50.0, 1.5)


def test_extreme_spectrum_warning():
    """Should warn when r > 3.5 (extreme spectrum)."""
    result = allen_dynes_corrected_tc(1.0, 0.13, 300.0, 4.0)
    assert any('extreme spectrum' in w.lower() for w in result['warnings']), \
        f"Missing extreme spectrum warning for r=4.0: {result['warnings']}"


def test_f1_factor_range():
    """f₁ should be ≥ 1.0 for all valid inputs."""
    for lam in [0.5, 1.0, 1.5, 2.0]:
        for mu in [0.08, 0.13, 0.20]:
            f1 = compute_f1_factor(lam, mu)
            assert f1 >= 1.0, f"f1={f1} < 1.0 for λ={lam}, μ*={mu}"


def test_denominator_guard():
    """Should raise ValueError when denominator ≤ 0 (unphysical)."""
    # For high μ* and low λ, denominator can go negative
    # Need λ ≤ μ*/(1 - 0.62μ*); for μ*=0.20 → λ ≤ 0.228
    with pytest.raises(ValueError, match="denominator"):
        allen_dynes_corrected_tc(0.15, 0.20, 300.0, 1.5)


def test_tc_positive():
    """Tc should always be positive for valid inputs."""
    for lam in [0.6, 1.0, 1.5]:
        for mu in [0.10, 0.13, 0.15]:
            for omega in [200.0, 500.0, 800.0]:
                for r in [1.2, 1.5, 2.0]:
                    result = allen_dynes_corrected_tc(lam, mu, omega, r)
                    assert result['Tc'] > 0, \
                        f"Tc ≤ 0 for λ={lam}, μ*={mu}, ω={omega}, r={r}"


def test_determinism():
    """Same inputs should give identical results."""
    result1 = allen_dynes_corrected_tc(1.0, 0.13, 300.0, 1.5)
    result2 = allen_dynes_corrected_tc(1.0, 0.13, 300.0, 1.5)
    
    assert result1['Tc'] == result2['Tc']
    assert result1['f1_factor'] == result2['f1_factor']
    assert result1['f2_factor'] == result2['f2_factor']


def test_comparison_with_known_values():
    """Test against Nb (λ≈0.82, μ*≈0.13, ω_log≈276K, Tc≈9.25K)."""
    # Approximate values for Nb
    lam = 0.82
    mu = 0.13
    omega = 276.0
    ratio = get_omega2_ratio('Nb')
    
    result = allen_dynes_corrected_tc(lam, mu, omega, ratio)
    Tc_pred = result['Tc']
    
    # Should be within ±50% of experimental 9.25K (BCS accuracy)
    assert 4.0 < Tc_pred < 14.0, \
        f"Nb Tc prediction {Tc_pred:.2f}K outside expected range [4, 14]K"

