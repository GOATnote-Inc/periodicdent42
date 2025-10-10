"""
Tests for HTC domain module (superconductor prediction).

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

from __future__ import annotations

import pytest
import numpy as np

# Mark all tests in this module as htc
pytestmark = pytest.mark.htc


def test_imports():
    """Test that HTC modules can be imported."""
    from app.src.htc.domain import (
        SuperconductorPrediction,
        SuperconductorPredictor,
        allen_dynes_tc,
        mcmillan_tc,
    )

    assert SuperconductorPrediction is not None
    assert SuperconductorPredictor is not None
    assert allen_dynes_tc is not None
    assert mcmillan_tc is not None


def test_mcmillan_formula():
    """Test McMillan's Tc formula."""
    from app.src.htc.domain import mcmillan_tc

    # Test case: typical values for conventional superconductor
    tc = mcmillan_tc(omega_log=200.0, lambda_ep=0.5, mu_star=0.13)

    assert tc > 0, "Tc should be positive"
    assert tc < 50, "Tc should be reasonable for conventional superconductor"


def test_allen_dynes_formula():
    """Test Allen-Dynes Tc formula."""
    from app.src.htc.domain import allen_dynes_tc

    # Test case: MgB2-like parameters
    # λ ≈ 0.62, ω_log ≈ 660 K → Tc ≈ 39 K
    tc = allen_dynes_tc(omega_log=660.0, lambda_ep=0.62, mu_star=0.13)

    assert tc > 30, "Tc should be > 30 K for MgB2-like parameters"
    assert tc < 50, "Tc should be < 50 K for MgB2-like parameters"

    # Test strong-coupling correction impact
    tc_basic = allen_dynes_tc(
        omega_log=660.0, lambda_ep=0.62, mu_star=0.13, include_strong_coupling=False
    )
    tc_corrected = allen_dynes_tc(
        omega_log=660.0, lambda_ep=0.62, mu_star=0.13, include_strong_coupling=True
    )

    assert abs(tc_corrected - tc_basic) < 10, "Strong-coupling correction should be modest"


def test_superconductor_prediction_dataclass():
    """Test SuperconductorPrediction dataclass."""
    from app.src.htc.domain import SuperconductorPrediction

    pred = SuperconductorPrediction(
        composition="MgB2",
        reduced_formula="MgB2",
        structure_info={"space_group": 191},
        tc_predicted=39.0,
        tc_lower_95ci=35.0,
        tc_upper_95ci=43.0,
        tc_uncertainty=2.0,
        pressure_required_gpa=0.0,
        lambda_ep=0.62,
        omega_log=660.0,
    )

    # Check ξ parameter is calculated
    assert pred.xi_parameter > 0, "ξ should be positive"
    assert pred.xi_parameter < 1.0, "ξ should be < 1 for λ < 1"

    # Check constraint satisfaction
    assert pred.satisfies_constraints(
        max_pressure_gpa=1.0, min_tc_kelvin=30.0, xi_threshold=4.0
    )

    # Check dict conversion
    pred_dict = pred.to_dict()
    assert "composition" in pred_dict
    assert "tc_predicted" in pred_dict


def test_constraint_satisfaction():
    """Test constraint checking logic."""
    from app.src.htc.domain import SuperconductorPrediction

    # Valid material
    good_pred = SuperconductorPrediction(
        "MgB2", "MgB2", {}, 39.0, 35.0, 43.0, 2.0, 0.0, 0.62, 660.0
    )

    assert good_pred.satisfies_constraints(max_pressure_gpa=1.0, min_tc_kelvin=30.0)

    # Fails Tc requirement
    low_tc_pred = SuperconductorPrediction(
        "Material", "Mat", {}, 20.0, 18.0, 22.0, 1.0, 0.0, 0.3, 400.0
    )

    assert not low_tc_pred.satisfies_constraints(max_pressure_gpa=1.0, min_tc_kelvin=77.0)

    # Fails pressure requirement
    high_p_pred = SuperconductorPrediction(
        "LaH10", "LaH10", {}, 250.0, 240.0, 260.0, 5.0, 170.0, 2.0, 800.0
    )

    assert not high_p_pred.satisfies_constraints(max_pressure_gpa=1.0, min_tc_kelvin=77.0)


def test_xi_constraint_validator():
    """Test ξ constraint validator."""
    from app.src.htc.domain import XiConstraintValidator, SuperconductorPrediction

    validator = XiConstraintValidator(threshold=4.0)

    # Valid material (ξ < 4.0)
    valid_pred = SuperconductorPrediction(
        "MgB2", "MgB2", {}, 39.0, 35.0, 43.0, 2.0, 0.0, 0.62, 660.0
    )

    satisfied, msg = validator.validate(valid_pred)
    assert satisfied, "ξ = 0.38 should satisfy threshold"
    assert "✓" in msg

    # Check violation rate
    predictions = [valid_pred, valid_pred, valid_pred]
    rate = validator.compute_violation_rate(predictions)
    assert rate == 0.0, "No violations should be detected"


def test_pareto_front_computation():
    """Test Pareto front computation."""
    from app.src.htc.domain import compute_pareto_front, SuperconductorPrediction

    # Create test materials
    materials = [
        # High Tc, high pressure (LaH10-like)
        SuperconductorPrediction(
            "LaH10", "LaH10", {}, 250.0, 240.0, 260.0, 5.0, 170.0, 2.0, 800.0
        ),
        # Low Tc, low pressure (MgB2-like)
        SuperconductorPrediction("MgB2", "MgB2", {}, 39.0, 35.0, 43.0, 2.0, 0.0, 0.62, 660.0),
        # Dominated point (worse in both objectives)
        SuperconductorPrediction(
            "Dominated", "Dom", {}, 30.0, 28.0, 32.0, 1.0, 50.0, 0.4, 500.0
        ),
    ]

    pareto = compute_pareto_front(
        materials, objectives=["tc_predicted", "pressure_required_gpa"], directions=["maximize", "minimize"]
    )

    # LaH10 and MgB2 should be in Pareto front, dominated point should not
    pareto_names = [p.composition for p in pareto]
    assert "LaH10" in pareto_names, "LaH10 should be Pareto optimal"
    assert "MgB2" in pareto_names, "MgB2 should be Pareto optimal"
    assert "Dominated" not in pareto_names, "Dominated point should not be in Pareto front"


@pytest.mark.slow
def test_superconductor_predictor():
    """Test SuperconductorPredictor class."""
    pytest.importorskip("pymatgen", reason="pymatgen required for predictor tests")

    from app.src.htc.domain import SuperconductorPredictor, load_benchmark_materials

    # Load benchmark materials
    materials = load_benchmark_materials(include_ambient=True)

    if not materials:
        pytest.skip("No benchmark materials available")

    # Initialize predictor
    predictor = SuperconductorPredictor(random_state=42)

    # Make prediction
    material = materials[0]
    pred = predictor.predict(material["structure"], pressure_gpa=material["pressure"])

    # Check prediction structure
    assert pred.tc_predicted > 0, "Tc should be positive"
    assert pred.tc_uncertainty > 0, "Uncertainty should be positive"
    assert pred.lambda_ep > 0, "λ should be positive"
    assert pred.omega_log > 0, "ω_log should be positive"
    assert 0 <= pred.xi_parameter <= 1, "ξ should be in [0,1] for physical λ"


@pytest.mark.slow
def test_validate_against_known_materials():
    """Test validation against known superconductors."""
    pytest.importorskip("pymatgen", reason="pymatgen required for validation tests")

    from app.src.htc.domain import (
        predict_tc_with_uncertainty,
        validate_against_known_materials,
        load_benchmark_materials,
    )

    # Load known materials
    materials = load_benchmark_materials(include_ambient=True)

    if not materials:
        pytest.skip("No benchmark materials available")

    # Make predictions
    predictions = []
    for mat in materials:
        pred = predict_tc_with_uncertainty(mat["structure"], mat["pressure"])
        predictions.append(pred)

    # Validate
    errors = validate_against_known_materials(predictions)

    # Check we have validation results
    assert len(errors) > 0, "Should have validation results"

    # Check errors are reasonable (within an order of magnitude)
    for material, error in errors.items():
        assert error < 200.0, f"Error for {material} too large: {error} K"


def test_edge_cases():
    """Test edge cases and error handling."""
    from app.src.htc.domain import allen_dynes_tc

    # Zero λ should give zero Tc
    tc = allen_dynes_tc(omega_log=500.0, lambda_ep=0.0)
    assert tc == 0.0, "Tc should be zero for λ=0"

    # Negative λ should give zero Tc (unphysical)
    tc = allen_dynes_tc(omega_log=500.0, lambda_ep=-0.5)
    assert tc == 0.0, "Tc should be zero for negative λ"


def test_uncertainty_estimation():
    """Test uncertainty estimation in predictions."""
    pytest.importorskip("pymatgen", reason="pymatgen required for uncertainty tests")

    from app.src.htc.domain import SuperconductorPredictor, load_benchmark_materials

    materials = load_benchmark_materials(include_ambient=True)

    if not materials:
        pytest.skip("No benchmark materials available")

    predictor = SuperconductorPredictor(random_state=42)
    material = materials[0]

    # Prediction with uncertainty
    pred_with_uncertainty = predictor.predict(
        material["structure"], pressure_gpa=0.0, include_uncertainty=True
    )

    # Prediction without uncertainty
    pred_without_uncertainty = predictor.predict(
        material["structure"], pressure_gpa=0.0, include_uncertainty=False
    )

    # Check uncertainty is calculated when requested
    assert pred_with_uncertainty.tc_uncertainty > 0, "Should have uncertainty when requested"
    assert pred_without_uncertainty.tc_uncertainty == 0, "Should have no uncertainty when not requested"

    # Confidence intervals should reflect uncertainty
    ci_width = pred_with_uncertainty.tc_upper_95ci - pred_with_uncertainty.tc_lower_95ci
    expected_width = 2 * 1.96 * pred_with_uncertainty.tc_uncertainty
    assert abs(ci_width - expected_width) < 1e-6, "CI width should match 2*1.96*uncertainty"

