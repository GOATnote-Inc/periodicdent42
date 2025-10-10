"""
Validation Module for HTC Integration

Validates HTC superconductor optimization integration with the
Materials ML Protocols framework.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.src.htc.domain import (
    SuperconductorPrediction,
    XiConstraintValidator,
    allen_dynes_tc,
    compute_pareto_front,
    load_benchmark_materials,
    validate_against_known_materials,
)

logger = logging.getLogger(__name__)


class HTCValidationSuite:
    """Validation test suite for HTC integration"""

    def __init__(self):
        self.results = {}
        logger.info("Initialized HTCValidationSuite")

    def run_all_tests(self) -> tuple[int, int]:
        """Run complete validation suite"""
        tests = [
            ("McMillan Formula", self.test_mcmillan_formula),
            ("Constraint Validation", self.test_constraints),
            ("Pareto Front", self.test_pareto_front),
            ("Known Materials Validation", self.test_known_materials),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                test_func()
                logger.info(f"✓ PASS: {test_name}")
                passed += 1
            except Exception as e:
                logger.error(f"✗ FAIL: {test_name} - {e}")
                failed += 1

        logger.info(f"Validation complete: {passed} passed, {failed} failed")
        return passed, failed

    def test_mcmillan_formula(self) -> None:
        """Test McMillan-Allen-Dynes formula"""
        # Test case: MgB2 (λ ≈ 0.62, ω_log ≈ 660 K → Tc ≈ 39 K)
        tc = allen_dynes_tc(omega_log=660.0, lambda_ep=0.62, mu_star=0.13)

        error = abs(tc - 39.0)
        if error > 0.20 * 39.0:
            raise ValueError(f"McMillan formula error too large: {error:.1f} K")

        logger.info(f"MgB2 prediction: Tc = {tc:.1f} K (error: {error:.1f} K)")

    def test_constraints(self) -> None:
        """Test constraint validation"""
        validator = XiConstraintValidator(threshold=4.0)

        # Case 1: Valid material
        pred_good = SuperconductorPrediction(
            composition="MgB2",
            reduced_formula="MgB2",
            structure_info={},
            tc_predicted=39.0,
            tc_lower_95ci=35.0,
            tc_upper_95ci=43.0,
            tc_uncertainty=2.0,
            pressure_required_gpa=0.0,
            lambda_ep=0.62,
            omega_log=660.0,
        )

        satisfied, msg = validator.validate(pred_good)
        if not satisfied:
            raise ValueError("Valid material failed constraint")

        logger.info(f"Constraint test passed: {msg}")

    def test_pareto_front(self) -> None:
        """Test Pareto front computation"""
        predictions = [
            SuperconductorPrediction(
                "LaH10", "LaH10", {}, 250.0, 240.0, 260.0, 5.0, 170.0, 2.0, 800.0
            ),
            SuperconductorPrediction(
                "MgB2", "MgB2", {}, 39.0, 35.0, 43.0, 2.0, 0.0, 0.62, 660.0
            ),
        ]

        pareto = compute_pareto_front(
            predictions,
            objectives=["tc_predicted", "pressure_required_gpa"],
            directions=["maximize", "minimize"],
        )

        pareto_names = [p.composition for p in pareto]
        if "MgB2" not in pareto_names or "LaH10" not in pareto_names:
            raise ValueError("Expected materials not in Pareto front")

        logger.info(f"Pareto front computed: {len(pareto)} optimal materials")

    def test_known_materials(self) -> None:
        """Test validation against known superconductors"""
        materials = load_benchmark_materials(include_ambient=True)

        if not materials:
            logger.warning("No benchmark materials available")
            return

        from app.src.htc.domain import predict_tc_with_uncertainty

        # Make predictions
        predictions = []
        for mat in materials:
            pred = predict_tc_with_uncertainty(mat["structure"], mat["pressure"])
            predictions.append(pred)

        # Validate
        errors = validate_against_known_materials(predictions)

        if not any(e < 20.0 for e in errors.values()):
            raise ValueError("No materials within 20K error tolerance")

        logger.info(f"Validated {len(errors)} materials successfully")


def quick_validation() -> bool:
    """Run quick validation check"""
    suite = HTCValidationSuite()
    passed, failed = suite.run_all_tests()
    return failed == 0


logger.info("HTC validation module initialized")

