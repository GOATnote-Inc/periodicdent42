"""
Golden tests for BETE-NET predictions against known superconductors.

These tests validate model predictions against experimentally measured T_c values.
For publication-quality research, these MUST pass with real BETE-NET weights.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
"""

import pytest
from pathlib import Path

# Check if real weights are available
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "third_party" / "bete_net" / "models"
REAL_WEIGHTS_AVAILABLE = (WEIGHTS_DIR / "model_ensemble_0.pt").exists()

pytestmark = pytest.mark.bete


@pytest.mark.skipif(
    not REAL_WEIGHTS_AVAILABLE,
    reason="Real BETE-NET weights not downloaded. Using mock models for development only."
)
class TestGoldenMaterialsRealWeights:
    """
    Golden tests with REAL BETE-NET weights.
    
    These tests validate predictions against experimentally known T_c values.
    Required for publication-quality research.
    """
    
    def test_niobium_tc_prediction(self):
        """
        Test niobium (Nb) T_c prediction.
        
        Experimental T_c: 9.2 K
        Materials Project ID: mp-48
        Expected range: 8.0-10.5 K (±15% tolerance)
        """
        from src.bete_net_io.inference import predict_tc
        
        result = predict_tc("mp-48", mu_star=0.10, seed=42)
        
        # Assertions for publication
        assert result.formula == "Nb", f"Expected Nb, got {result.formula}"
        assert 8.0 <= result.tc_kelvin <= 10.5, \
            f"Nb T_c = {result.tc_kelvin:.2f} K (expected 8.0-10.5 K, experimental: 9.2 K)"
        assert result.tc_std < 2.0, f"Uncertainty too high: {result.tc_std:.2f} K"
        assert result.lambda_ep > 0.8, f"λ too low for Nb: {result.lambda_ep:.3f}"
        
        print(f"✅ Niobium: T_c = {result.tc_kelvin:.2f} ± {result.tc_std:.2f} K")
        print(f"   λ = {result.lambda_ep:.3f} ± {result.lambda_std:.3f}")
        print(f"   ⟨ω_log⟩ = {result.omega_log:.1f} K")
    
    def test_mgb2_tc_prediction(self):
        """
        Test magnesium diboride (MgB₂) T_c prediction.
        
        Experimental T_c: 39 K
        Materials Project ID: mp-5486
        Expected range: 35.0-43.0 K (±10% tolerance)
        """
        from src.bete_net_io.inference import predict_tc
        
        result = predict_tc("mp-5486", mu_star=0.10, seed=42)
        
        # Assertions for publication
        assert "Mg" in result.formula and "B" in result.formula, \
            f"Expected MgB₂, got {result.formula}"
        assert 35.0 <= result.tc_kelvin <= 43.0, \
            f"MgB₂ T_c = {result.tc_kelvin:.2f} K (expected 35.0-43.0 K, experimental: 39 K)"
        assert result.tc_std < 3.0, f"Uncertainty too high: {result.tc_std:.2f} K"
        assert result.lambda_ep > 0.6, f"λ too low for MgB₂: {result.lambda_ep:.3f}"
        
        print(f"✅ MgB₂: T_c = {result.tc_kelvin:.2f} ± {result.tc_std:.2f} K")
        print(f"   λ = {result.lambda_ep:.3f} ± {result.lambda_std:.3f}")
        print(f"   ⟨ω_log⟩ = {result.omega_log:.1f} K")
    
    def test_aluminum_tc_prediction(self):
        """
        Test aluminum (Al) T_c prediction.
        
        Experimental T_c: 1.2 K
        Materials Project ID: mp-134
        Expected range: 0.8-1.6 K (±30% tolerance for low T_c)
        """
        from src.bete_net_io.inference import predict_tc
        
        result = predict_tc("mp-134", mu_star=0.10, seed=42)
        
        # Assertions for publication
        assert result.formula == "Al", f"Expected Al, got {result.formula}"
        assert 0.8 <= result.tc_kelvin <= 1.6, \
            f"Al T_c = {result.tc_kelvin:.2f} K (expected 0.8-1.6 K, experimental: 1.2 K)"
        assert result.tc_std < 0.5, f"Uncertainty too high: {result.tc_std:.2f} K"
        assert 0.3 < result.lambda_ep < 0.5, f"λ out of range for Al: {result.lambda_ep:.3f}"
        
        print(f"✅ Aluminum: T_c = {result.tc_kelvin:.2f} ± {result.tc_std:.2f} K")
        print(f"   λ = {result.lambda_ep:.3f} ± {result.lambda_std:.3f}")
        print(f"   ⟨ω_log⟩ = {result.omega_log:.1f} K")


class TestGoldenMaterialsMockModels:
    """
    Golden tests with MOCK models (development only).
    
    These tests verify the API works but DO NOT validate scientific accuracy.
    NOT suitable for publication.
    """
    
    @pytest.mark.skipif(
        REAL_WEIGHTS_AVAILABLE,
        reason="Real weights available - use TestGoldenMaterialsRealWeights instead"
    )
    def test_mock_niobium_structure(self):
        """Test mock prediction returns valid structure for Niobium."""
        from src.bete_net_io.inference import predict_tc
        
        result = predict_tc("mp-48", mu_star=0.10, seed=42)
        
        # Basic structure validation (not scientific accuracy)
        assert result.formula == "Nb"
        assert result.tc_kelvin > 0, "T_c must be positive"
        assert result.lambda_ep > 0, "λ must be positive"
        assert len(result.omega_grid) > 0, "ω grid must not be empty"
        assert len(result.alpha2F_mean) == len(result.omega_grid)
        
        print(f"⚠️  MOCK: Nb T_c = {result.tc_kelvin:.2f} K (NOT scientifically validated)")
    
    @pytest.mark.skipif(
        REAL_WEIGHTS_AVAILABLE,
        reason="Real weights available - use TestGoldenMaterialsRealWeights instead"
    )
    def test_mock_mgb2_structure(self):
        """Test mock prediction returns valid structure for MgB₂."""
        from src.bete_net_io.inference import predict_tc
        
        result = predict_tc("mp-5486", mu_star=0.10, seed=42)
        
        # Basic structure validation (not scientific accuracy)
        assert "Mg" in result.formula or "B" in result.formula
        assert result.tc_kelvin > 0
        assert result.lambda_ep > 0
        assert len(result.omega_grid) > 0
        
        print(f"⚠️  MOCK: MgB₂ T_c = {result.tc_kelvin:.2f} K (NOT scientifically validated)")
    
    @pytest.mark.skipif(
        REAL_WEIGHTS_AVAILABLE,
        reason="Real weights available - use TestGoldenMaterialsRealWeights instead"
    )
    def test_mock_aluminum_structure(self):
        """Test mock prediction returns valid structure for Aluminum."""
        from src.bete_net_io.inference import predict_tc
        
        result = predict_tc("mp-134", mu_star=0.10, seed=42)
        
        # Basic structure validation (not scientific accuracy)
        assert result.formula == "Al"
        assert result.tc_kelvin > 0
        assert result.lambda_ep > 0
        assert len(result.omega_grid) > 0
        
        print(f"⚠️  MOCK: Al T_c = {result.tc_kelvin:.2f} K (NOT scientifically validated)")


def test_prediction_reproducibility():
    """
    Test that predictions are reproducible with fixed seed.
    
    This is CRITICAL for scientific reproducibility in publications.
    """
    from src.bete_net_io.inference import predict_tc
    
    # Run prediction twice with same seed
    result1 = predict_tc("mp-48", mu_star=0.10, seed=42)
    result2 = predict_tc("mp-48", mu_star=0.10, seed=42)
    
    # Must be bit-identical for reproducibility
    assert result1.tc_kelvin == result2.tc_kelvin, "T_c must be reproducible"
    assert result1.lambda_ep == result2.lambda_ep, "λ must be reproducible"
    assert (result1.alpha2F_mean == result2.alpha2F_mean).all(), "α²F must be reproducible"
    
    print(f"✅ Reproducibility verified: T_c = {result1.tc_kelvin:.2f} K (seed=42)")


def test_uncertainty_quantification():
    """
    Test that ensemble provides uncertainty estimates.
    
    Required for publication: must quantify prediction uncertainty.
    """
    from src.bete_net_io.inference import predict_tc
    
    result = predict_tc("mp-48", mu_star=0.10, seed=42)
    
    # Uncertainty must be provided
    assert result.tc_std is not None, "T_c uncertainty required"
    assert result.lambda_std is not None, "λ uncertainty required"
    assert result.alpha2F_std is not None, "α²F uncertainty required"
    
    # Uncertainty should be reasonable (not zero, not huge)
    assert result.tc_std > 0, "Uncertainty must be positive"
    assert result.tc_std < result.tc_kelvin, "Uncertainty should be < prediction"
    
    # Coefficient of variation should be reasonable (<50%)
    cv = result.tc_std / result.tc_kelvin
    assert cv < 0.5, f"Coefficient of variation too high: {cv:.2%}"
    
    print(f"✅ Uncertainty quantification: T_c = {result.tc_kelvin:.2f} ± {result.tc_std:.2f} K (CV={cv:.1%})")


@pytest.mark.parametrize("mu_star", [0.05, 0.10, 0.15, 0.20])
def test_mu_star_sensitivity(mu_star):
    """
    Test predictions across different μ* values (Coulomb pseudopotential).
    
    Physical expectation: T_c decreases with increasing μ*
    """
    from src.bete_net_io.inference import predict_tc
    
    result = predict_tc("mp-48", mu_star=mu_star, seed=42)
    
    assert result.mu_star == mu_star, f"μ* not set correctly"
    assert result.tc_kelvin > 0, f"T_c must be positive for μ*={mu_star}"
    
    print(f"μ*={mu_star:.2f}: T_c = {result.tc_kelvin:.2f} ± {result.tc_std:.2f} K")


def test_provenance_documentation():
    """
    Test that predictions include full provenance information.
    
    Required for publication: must document model version, checksums, timestamps.
    """
    from src.bete_net_io.inference import predict_tc
    
    result = predict_tc("mp-48", mu_star=0.10, seed=42)
    
    # Provenance fields required
    assert result.model_version is not None, "Model version required"
    assert result.model_checksum is not None, "Model checksum required"
    assert result.timestamp is not None, "Timestamp required"
    assert result.input_hash is not None, "Input hash required"
    assert result.seed is not None, "Seed required for reproducibility"
    
    print(f"✅ Provenance: {result.model_version} ({result.model_checksum[:8]}...)")
    print(f"   Timestamp: {result.timestamp}")
    print(f"   Input hash: {result.input_hash[:8]}...")
    print(f"   Seed: {result.seed}")


if __name__ == "__main__":
    # Print status message
    if REAL_WEIGHTS_AVAILABLE:
        print("✅ REAL BETE-NET WEIGHTS DETECTED")
        print("   Running publication-quality validation tests...")
    else:
        print("⚠️  MOCK MODELS IN USE")
        print("   Real weights not found at:", WEIGHTS_DIR)
        print("   Running development tests only (NOT suitable for publication)")
        print()
        print("To download real weights (required for publication):")
        print("  1. See: third_party/bete_net/WEIGHTS_INFO.md")
        print("  2. Run: bash scripts/download_bete_weights_real.sh")
        print("  3. Rerun tests: pytest app/tests/test_bete_golden.py -v")
    
    print()
    pytest.main([__file__, "-v", "--tb=short"])

