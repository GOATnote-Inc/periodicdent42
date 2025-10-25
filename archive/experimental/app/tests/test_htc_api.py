"""
Tests for HTC API endpoints.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.htc


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from app.src.api.main import app

    return TestClient(app)


def test_htc_health_endpoint(client):
    """Test HTC health check endpoint."""
    response = client.get("/api/htc/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "module" in data
    assert data["module"] == "HTC Superconductor Optimization"
    assert "enabled" in data
    assert "features" in data


def test_predict_endpoint_basic(client):
    """Test basic Tc prediction endpoint."""
    response = client.post(
        "/api/htc/predict",
        json={
            "composition": "MgB2",
            "pressure_gpa": 0.0,
            "include_uncertainty": True,
        },
    )

    # Should return 200 if HTC enabled, 501 if not
    if response.status_code == 501:
        pytest.skip("HTC dependencies not available")

    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "composition" in data
    assert "tc_predicted" in data
    assert "tc_uncertainty" in data
    assert "lambda_ep" in data
    assert "omega_log" in data
    assert "xi_parameter" in data
    assert "timestamp" in data


def test_predict_endpoint_validation(client):
    """Test prediction endpoint input validation."""
    # Missing composition
    response = client.post(
        "/api/htc/predict",
        json={"pressure_gpa": 0.0},
    )

    assert response.status_code == 422, "Should fail validation for missing composition"

    # Negative pressure
    response = client.post(
        "/api/htc/predict",
        json={"composition": "MgB2", "pressure_gpa": -10.0},
    )

    assert response.status_code == 422, "Should fail validation for negative pressure"

    # Pressure too high
    response = client.post(
        "/api/htc/predict",
        json={"composition": "MgB2", "pressure_gpa": 1000.0},
    )

    assert response.status_code == 422, "Should fail validation for excessive pressure"


def test_screen_endpoint(client):
    """Test materials screening endpoint."""
    response = client.post(
        "/api/htc/screen",
        json={
            "max_pressure_gpa": 1.0,
            "min_tc_kelvin": 77.0,
            "use_benchmark_materials": True,
        },
    )

    if response.status_code == 501:
        pytest.skip("HTC dependencies not available")

    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "run_id" in data
    assert "n_candidates" in data
    assert "n_passing" in data
    assert "success_rate" in data
    assert "predictions" in data
    assert "passing_candidates" in data
    assert "statistical_summary" in data


def test_optimize_endpoint(client):
    """Test multi-objective optimization endpoint."""
    response = client.post(
        "/api/htc/optimize",
        json={
            "max_pressure_gpa": 1.0,
            "min_tc_kelvin": 77.0,
            "use_benchmark_materials": True,
        },
    )

    if response.status_code == 501:
        pytest.skip("HTC dependencies not available")

    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "run_id" in data
    assert "n_evaluated" in data
    assert "n_pareto_optimal" in data
    assert "pareto_front" in data
    assert "validation_results" in data
    assert "compliance" in data


def test_validate_endpoint(client):
    """Test validation endpoint."""
    response = client.post("/api/htc/validate")

    if response.status_code == 501:
        pytest.skip("HTC dependencies not available")

    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "validation_errors" in data
    assert "mean_error" in data
    assert "max_error" in data
    assert "materials_within_20K" in data
    assert "total_materials" in data


@pytest.mark.slow
def test_results_retrieval(client):
    """Test results retrieval endpoint."""
    # First run a screening to get a run_id
    screen_response = client.post(
        "/api/htc/screen",
        json={
            "max_pressure_gpa": 1.0,
            "min_tc_kelvin": 77.0,
            "use_benchmark_materials": True,
        },
    )

    if screen_response.status_code == 501:
        pytest.skip("HTC dependencies not available")

    assert screen_response.status_code == 200
    run_id = screen_response.json()["run_id"]

    # Retrieve results
    results_response = client.get(f"/api/htc/results/{run_id}")

    assert results_response.status_code == 200
    results_data = results_response.json()

    # Check it matches screening response structure
    assert "experiment_type" in results_data
    assert results_data["experiment_type"] == "HTC_screening"


def test_results_not_found(client):
    """Test results retrieval with invalid run_id."""
    response = client.get("/api/htc/results/invalid-uuid-12345")

    if response.status_code == 501:
        pytest.skip("HTC dependencies not available")

    assert response.status_code == 404, "Should return 404 for non-existent run_id"


def test_api_documentation_includes_htc(client):
    """Test that HTC endpoints appear in OpenAPI docs."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi_spec = response.json()

    # Check HTC endpoints are documented
    paths = openapi_spec.get("paths", {})

    assert "/api/htc/predict" in paths, "Predict endpoint should be documented"
    assert "/api/htc/screen" in paths, "Screen endpoint should be documented"
    assert "/api/htc/optimize" in paths, "Optimize endpoint should be documented"
    assert "/api/htc/validate" in paths, "Validate endpoint should be documented"
    assert "/api/htc/results/{run_id}" in paths, "Results endpoint should be documented"


def test_concurrent_requests(client):
    """Test handling multiple concurrent requests."""
    pytest.importorskip("pymatgen", reason="pymatgen required for this test")

    import concurrent.futures

    def make_prediction():
        return client.post(
            "/api/htc/predict",
            json={"composition": "MgB2", "pressure_gpa": 0.0},
        )

    # Make 5 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_prediction) for _ in range(5)]
        responses = [f.result() for f in futures]

    # All should succeed (or all fail if HTC not enabled)
    status_codes = [r.status_code for r in responses]

    if status_codes[0] == 501:
        pytest.skip("HTC dependencies not available")

    assert all(code == 200 for code in status_codes), "All concurrent requests should succeed"

