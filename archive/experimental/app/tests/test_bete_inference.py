"""
Tests for BETE-NET inference wrapper.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from app.src.bete_net_io.inference import (
    BETEPrediction,
    allen_dynes_tc,
    compute_structure_hash,
    load_structure,
    predict_tc,
)


class TestAllenDynesFormula:
    """Test Allen-Dynes Tc calculation."""

    def test_typical_superconductor(self):
        """Test Tc calculation for typical values (Nb-like)."""
        # Nb: λ~1.0, ω_log~250K, μ*=0.10 → Tc~9K
        tc = allen_dynes_tc(lambda_ep=1.0, omega_log_K=250.0, mu_star=0.10)
        assert 7.0 < tc < 12.0, "Tc should be ~9K for Nb-like parameters"

    def test_weak_coupling(self):
        """Test Tc calculation for weak coupling (λ < μ*)."""
        tc = allen_dynes_tc(lambda_ep=0.05, omega_log_K=300.0, mu_star=0.10)
        assert tc == 0.0, "Tc should be 0 when λ < μ*"

    def test_strong_coupling(self):
        """Test Tc calculation for strong coupling."""
        # MgB2-like: λ~0.7, ω_log~500K → Tc~39K
        tc = allen_dynes_tc(lambda_ep=0.7, omega_log_K=500.0, mu_star=0.10)
        assert 30.0 < tc < 50.0, "Tc should be ~39K for MgB2-like parameters"

    def test_numerical_stability(self):
        """Test numerical stability at edge cases."""
        # λ exactly equals μ*
        tc = allen_dynes_tc(lambda_ep=0.10, omega_log_K=250.0, mu_star=0.10)
        assert tc == 0.0

        # Very large λ
        tc = allen_dynes_tc(lambda_ep=2.5, omega_log_K=300.0, mu_star=0.10)
        assert tc > 0.0 and tc < 100.0

    def test_mu_star_sensitivity(self):
        """Test sensitivity to μ* parameter."""
        tc1 = allen_dynes_tc(lambda_ep=1.0, omega_log_K=250.0, mu_star=0.10)
        tc2 = allen_dynes_tc(lambda_ep=1.0, omega_log_K=250.0, mu_star=0.13)
        assert tc2 < tc1, "Higher μ* should reduce Tc"
        assert 0.7 < tc2 / tc1 < 0.95, "Tc reduction should be 5-30%"


@pytest.mark.bete
class TestStructureLoading:
    """Test crystal structure loading (requires pymatgen)."""

    def test_cif_file_loading(self, tmp_path):
        """Test loading structure from CIF file."""
        # Create minimal CIF
        cif_content = """
data_Nb
_cell_length_a 3.3008
_cell_length_b 3.3008
_cell_length_c 3.3008
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'Im-3m'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Nb 0.0 0.0 0.0
        """
        cif_path = tmp_path / "Nb.cif"
        cif_path.write_text(cif_content)

        # Load structure
        structure, formula, mp_id = load_structure(str(cif_path))

        assert formula == "Nb", f"Expected Nb, got {formula}"
        assert mp_id is None, "CIF file should not have MP-ID"
        assert len(structure) >= 1, "Structure should have at least 1 atom"

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="must be CIF file path or MP-ID"):
            load_structure("not_a_valid_input")

    def test_missing_file(self):
        """Test error handling for missing CIF file."""
        with pytest.raises(ValueError, match="Failed to parse CIF file"):
            load_structure("/nonexistent/file.cif")

    @pytest.mark.integration
    def test_materials_project_loading(self):
        """Test loading structure from Materials Project (requires API key)."""
        pytest.importorskip("pymatgen")
        
        try:
            structure, formula, mp_id = load_structure("mp-48")  # Nb
            assert formula == "Nb"
            assert mp_id == "mp-48"
        except ValueError as e:
            if "API" in str(e) or "fetch" in str(e):
                pytest.skip("Materials Project API not available")
            raise


class TestStructureHash:
    """Test structure hashing for provenance."""

    def test_hash_determinism(self, tmp_path):
        """Test that same structure produces same hash."""
        cif_content = """
data_Al
_cell_length_a 4.05
_cell_length_b 4.05
_cell_length_c 4.05
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al 0.0 0.0 0.0
        """
        cif_path = tmp_path / "Al.cif"
        cif_path.write_text(cif_content)

        # Load twice
        structure1, _, _ = load_structure(str(cif_path))
        structure2, _, _ = load_structure(str(cif_path))

        hash1 = compute_structure_hash(structure1)
        hash2 = compute_structure_hash(structure2)

        assert hash1 == hash2, "Same structure should produce same hash"
        assert len(hash1) == 64, "SHA-256 hash should be 64 hex characters"


@pytest.mark.bete
class TestBETEPrediction:
    """Test BETEPrediction dataclass and serialization."""

    def test_to_dict(self):
        """Test conversion to JSON-serializable dict."""
        import numpy as np
        from datetime import datetime

        prediction = BETEPrediction(
            formula="Nb",
            input_hash="abc123",
            mp_id="mp-48",
            omega_grid=np.linspace(0, 0.1, 10),
            alpha2F_mean=np.ones(10) * 0.5,
            alpha2F_std=np.ones(10) * 0.05,
            lambda_ep=1.0,
            lambda_std=0.1,
            omega_log=250.0,
            omega_log_std=25.0,
            tc_kelvin=9.2,
            tc_std=1.4,
            mu_star=0.10,
            timestamp=datetime.utcnow().isoformat(),
        )

        result = prediction.to_dict()

        assert result["formula"] == "Nb"
        assert result["tc_kelvin"] == 9.2
        assert result["lambda_ep"] == 1.0
        assert "alpha2F" in result
        assert len(result["alpha2F"]["mean"]) == 10

        # Test JSON serialization
        json_str = json.dumps(result)
        assert "Nb" in json_str


@pytest.mark.bete
class TestPredictTc:
    """Test end-to-end Tc prediction (mocked model)."""

    def test_predict_tc_mock(self, tmp_path):
        """Test predict_tc with mock BETE-NET model."""
        # Create test CIF
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
        cif_path = tmp_path / "Nb.cif"
        cif_path.write_text(cif_content)

        # Run prediction (uses mock data internally)
        prediction = predict_tc(str(cif_path), mu_star=0.10)

        assert prediction.formula == "Nb"
        assert prediction.tc_kelvin > 0.0
        assert prediction.lambda_ep > 0.0
        assert prediction.omega_log > 0.0
        assert len(prediction.input_hash) == 64
        assert prediction.mu_star == 0.10

    def test_golden_prediction_reproducibility(self, tmp_path):
        """Test that predictions are reproducible with same inputs."""
        cif_content = """
data_Al
_cell_length_a 4.05
_cell_length_b 4.05
_cell_length_c 4.05
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Al 0.0 0.0 0.0
        """
        cif_path = tmp_path / "Al.cif"
        cif_path.write_text(cif_content)

        # Run twice with same seed
        pred1 = predict_tc(str(cif_path), mu_star=0.10, seed=42)
        pred2 = predict_tc(str(cif_path), mu_star=0.10, seed=42)

        assert pred1.input_hash == pred2.input_hash
        assert pred1.tc_kelvin == pred2.tc_kelvin
        assert pred1.lambda_ep == pred2.lambda_ep


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

