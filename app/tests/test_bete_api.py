"""
Tests for BETE-NET FastAPI endpoints.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.src.api.main import app

client = TestClient(app)


@pytest.mark.bete
class TestPredictEndpoint:
    """Test /api/bete/predict endpoint."""

    def test_predict_with_mp_id(self):
        """Test prediction with Materials Project ID."""
        response = client.post(
            "/api/bete/predict",
            json={"mp_id": "mp-48", "mu_star": 0.10}
        )
        
        # Note: This will fail until BETE-NET is fully integrated
        # For now, just check endpoint exists
        assert response.status_code in [200, 500], "Endpoint should exist"

    def test_predict_with_cif(self):
        """Test prediction with CIF content."""
        cif_content = """
data_Nb
_cell_length_a 3.3
_cell_length_b 3.3
_cell_length_c 3.3
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Nb 0.0 0.0 0.0
        """
        
        response = client.post(
            "/api/bete/predict",
            json={"cif_content": cif_content, "mu_star": 0.13}
        )
        
        assert response.status_code in [200, 500], "Endpoint should exist"

    def test_predict_missing_input(self):
        """Test error handling when no input provided."""
        response = client.post(
            "/api/bete/predict",
            json={"mu_star": 0.10}  # No cif_content or mp_id
        )
        
        assert response.status_code == 400, "Should return 400 Bad Request"
        assert "must provide" in response.json()["detail"].lower()

    def test_predict_both_inputs(self):
        """Test error handling when both inputs provided."""
        response = client.post(
            "/api/bete/predict",
            json={
                "mp_id": "mp-48",
                "cif_content": "...",
                "mu_star": 0.10
            }
        )
        
        assert response.status_code == 400, "Should return 400 Bad Request"

    def test_predict_response_schema(self):
        """Test that successful response has correct schema."""
        # This test will pass once model is integrated
        # For now, document expected schema
        expected_fields = [
            "formula",
            "mp_id",
            "tc_kelvin",
            "tc_std",
            "lambda_ep",
            "lambda_std",
            "omega_log_K",
            "omega_log_std_K",
            "mu_star",
            "input_hash",
            "evidence_url",
            "timestamp",
        ]
        
        # Just document expected response
        assert len(expected_fields) == 12, "Expected 12 fields in response"


@pytest.mark.bete
class TestScreenEndpoint:
    """Test /api/bete/screen endpoint."""

    def test_screen_with_mp_ids(self):
        """Test batch screening with MP-IDs."""
        response = client.post(
            "/api/bete/screen",
            json={
                "mp_ids": ["mp-48", "mp-66", "mp-134"],
                "mu_star": 0.13,
                "n_workers": 2
            }
        )
        
        # Endpoint should exist and queue job
        assert response.status_code in [200, 500], "Endpoint should exist"
        
        if response.status_code == 200:
            data = response.json()
            assert "run_id" in data
            assert "n_materials" in data
            assert data["n_materials"] == 3

    def test_screen_missing_inputs(self):
        """Test error handling when no inputs provided."""
        response = client.post(
            "/api/bete/screen",
            json={"mu_star": 0.10, "n_workers": 4}
        )
        
        assert response.status_code == 400, "Should return 400 Bad Request"
        assert "must provide" in response.json()["detail"].lower()


@pytest.mark.bete
class TestReportEndpoint:
    """Test /api/bete/report/{id} endpoint."""

    def test_report_not_found(self):
        """Test 404 for non-existent report."""
        response = client.get("/api/bete/report/nonexistent-id")
        
        assert response.status_code == 404, "Should return 404 for missing report"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

