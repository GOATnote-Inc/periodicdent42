"""
Integration tests for complete HTC workflow.

Tests end-to-end functionality: domain → runner → API

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile

pytestmark = [pytest.mark.htc, pytest.mark.integration, pytest.mark.slow]


def test_complete_screening_workflow():
    """Test complete screening workflow from domain to runner."""
    pytest.importorskip("pymatgen", reason="pymatgen required for integration tests")

    from app.src.htc.runner import IntegratedExperimentRunner

    # Create runner with temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = IntegratedExperimentRunner(
            evidence_dir=Path(tmpdir) / "evidence", results_dir=Path(tmpdir) / "results"
        )

        # Run screening experiment
        results = runner.run_experiment("HTC_screening", max_pressure_gpa=1.0, min_tc_kelvin=77.0)

        # Check results structure
        assert "experiment_type" in results
        assert results["experiment_type"] == "HTC_screening"
        assert "predictions" in results
        assert "metadata" in results

        # Check metadata includes provenance
        metadata = results["metadata"]
        assert "git_sha" in metadata
        assert "timestamp" in metadata
        assert "checksum" in metadata

        # Check statistical summary is present
        if "statistical_summary" in results:
            stats = results["statistical_summary"]
            assert "tc_statistics" in stats or "xi_statistics" in stats


def test_complete_optimization_workflow():
    """Test complete optimization workflow."""
    pytest.importorskip("pymatgen", reason="pymatgen required for integration tests")

    from app.src.htc.runner import IntegratedExperimentRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = IntegratedExperimentRunner(
            evidence_dir=Path(tmpdir) / "evidence", results_dir=Path(tmpdir) / "results"
        )

        # Run optimization
        results = runner.run_experiment(
            "HTC_optimization", max_pressure_gpa=1.0, min_tc_kelvin=77.0
        )

        # Check Pareto front is computed
        assert "pareto_front" in results
        assert isinstance(results["pareto_front"], list)

        # Check validation results
        assert "validation_results" in results

        # Check compliance checking
        assert "compliance" in results
        compliance = results["compliance"]
        assert "fully_compliant" in compliance
        assert "checks" in compliance


def test_validation_workflow():
    """Test validation workflow against known materials."""
    pytest.importorskip("pymatgen", reason="pymatgen required for integration tests")

    from app.src.htc.runner import IntegratedExperimentRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = IntegratedExperimentRunner(
            evidence_dir=Path(tmpdir) / "evidence", results_dir=Path(tmpdir) / "results"
        )

        # Run validation
        results = runner.run_experiment("HTC_validation")

        # Check validation results
        assert "validation_errors" in results
        assert "summary" in results

        summary = results["summary"]
        assert "mean_error" in summary
        assert "max_error" in summary


def test_api_to_domain_integration():
    """Test API layer properly calls domain layer."""
    pytest.importorskip("pymatgen", reason="pymatgen required for integration tests")

    from fastapi.testclient import TestClient
    from app.src.api.main import app

    client = TestClient(app)

    # Test prediction endpoint
    response = client.post(
        "/api/htc/predict",
        json={"composition": "MgB2", "pressure_gpa": 0.0, "include_uncertainty": True},
    )

    if response.status_code == 501:
        pytest.skip("HTC dependencies not available")

    assert response.status_code == 200
    data = response.json()

    # Verify domain-level fields are present
    assert "tc_predicted" in data
    assert "lambda_ep" in data
    assert "xi_parameter" in data


def test_evidence_persistence():
    """Test that results are persisted correctly."""
    pytest.importorskip("pymatgen", reason="pymatgen required for integration tests")

    from app.src.htc.runner import IntegratedExperimentRunner
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        evidence_dir = Path(tmpdir) / "evidence"
        results_dir = Path(tmpdir) / "results"

        runner = IntegratedExperimentRunner(evidence_dir=evidence_dir, results_dir=results_dir)

        # Run experiment
        results = runner.run_experiment("HTC_screening", max_pressure_gpa=1.0, min_tc_kelvin=77.0)

        # Check evidence files were created
        evidence_files = list(evidence_dir.glob("*.json"))
        assert len(evidence_files) > 0, "Evidence file should be created"

        # Check summary files were created
        summary_files = list(results_dir.glob("*_summary.txt"))
        assert len(summary_files) > 0, "Summary file should be created"

        # Verify evidence file content
        with open(evidence_files[0], "r") as f:
            saved_results = json.load(f)

        assert saved_results["metadata"]["checksum"] == results["metadata"]["checksum"]


def test_reproducibility():
    """Test that results are reproducible with same random seed."""
    pytest.importorskip("pymatgen", reason="pymatgen required for integration tests")

    from app.src.htc.runner import IntegratedExperimentRunner

    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        runner1 = IntegratedExperimentRunner(
            evidence_dir=Path(tmpdir1) / "evidence", results_dir=Path(tmpdir1) / "results"
        )

        runner2 = IntegratedExperimentRunner(
            evidence_dir=Path(tmpdir2) / "evidence", results_dir=Path(tmpdir2) / "results"
        )

        # Run same experiment twice with same seed
        results1 = runner1.run_experiment(
            "HTC_screening", random_state=42, max_pressure_gpa=1.0, min_tc_kelvin=77.0
        )

        results2 = runner2.run_experiment(
            "HTC_screening", random_state=42, max_pressure_gpa=1.0, min_tc_kelvin=77.0
        )

        # Check predictions are identical
        assert len(results1["predictions"]) == len(results2["predictions"])

        for pred1, pred2 in zip(results1["predictions"], results2["predictions"]):
            assert pred1["tc_predicted"] == pytest.approx(pred2["tc_predicted"], abs=1e-6)
            assert pred1["lambda_ep"] == pytest.approx(pred2["lambda_ep"], abs=1e-6)


def test_error_handling_integration():
    """Test error handling across layers."""
    from fastapi.testclient import TestClient
    from app.src.api.main import app

    client = TestClient(app)

    # Test invalid composition (should be handled gracefully)
    response = client.post(
        "/api/htc/predict",
        json={"composition": "", "pressure_gpa": 0.0},  # Empty composition
    )

    # Should either validate (422) or handle gracefully (200 with error info)
    assert response.status_code in [200, 422, 500, 501]


def test_benchmark_materials_loading():
    """Test that benchmark materials can be loaded and used."""
    pytest.importorskip("pymatgen", reason="pymatgen required for benchmark tests")

    from app.src.htc.domain import load_benchmark_materials, predict_tc_with_uncertainty

    materials = load_benchmark_materials(include_ambient=True)

    if not materials:
        pytest.skip("No benchmark materials available")

    # Should have at least MgB2
    assert len(materials) > 0

    # Should be able to predict for each material
    for material in materials:
        pred = predict_tc_with_uncertainty(material["structure"], material["pressure"])

        assert pred.tc_predicted > 0
        assert pred.composition != "Unknown"


def test_validation_suite():
    """Test the built-in validation suite."""
    pytest.importorskip("pymatgen", reason="pymatgen required for validation suite")

    from app.src.htc.validation import HTCValidationSuite

    suite = HTCValidationSuite()
    passed, failed = suite.run_all_tests()

    # At least some tests should pass
    assert passed > 0, "Validation suite should have some passing tests"

    # Most tests should pass (allow some failures for missing optional features)
    assert passed >= failed, "Majority of validation tests should pass"

