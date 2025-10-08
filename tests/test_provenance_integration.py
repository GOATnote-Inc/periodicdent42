"""Integration tests for provenance pipeline components.

Tests schemas, metrics, calibration, and dataset validation.
Ensures production-ready quality with edge case coverage.
"""

import json
import pathlib
import pytest
import sys
import tempfile
import hashlib
from datetime import datetime, timezone

# Add parent directory for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from schemas.ci_telemetry import CIRun, TestResult, CIProvenance, ExperimentLedgerEntry
from pydantic import ValidationError

# Import calibration and metrics
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
import calibration
from metrics.epistemic import (
    bernoulli_entropy,
    compute_expected_information_gain,
    compute_detection_rate,
    compute_epistemic_efficiency,
    enrich_tests_with_epistemic_features,
)


class TestCITelemetrySchemas:
    """Test Pydantic schemas for CI telemetry."""
    
    def test_test_result_valid(self):
        """Test valid TestResult creation."""
        test = TestResult(
            name="tests/test_foo.py::test_bar",
            suite="materials",
            domain="materials",
            duration_sec=5.2,
            result="pass",
            cost_usd=0.001,
            timestamp=datetime.now(timezone.utc),
            model_uncertainty=0.15,
        )
        
        assert test.name == "tests/test_foo.py::test_bar"
        assert test.result == "pass"
        assert test.cost_usd == 0.001
    
    def test_test_result_invalid_result(self):
        """Test TestResult with invalid result value."""
        with pytest.raises(ValidationError):
            TestResult(
                name="test",
                suite="generic",
                domain="generic",
                duration_sec=1.0,
                result="invalid_result",  # Should fail
                cost_usd=0.001,
                timestamp=datetime.now(timezone.utc),
            )
    
    def test_ci_run_valid(self):
        """Test valid CIRun creation."""
        tests = [
            TestResult(
                name=f"test_{i}",
                suite="generic",
                domain="generic",
                duration_sec=1.0,
                result="pass",
                cost_usd=0.001,
                timestamp=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]
        
        ci_run = CIRun(
            commit="abc123def456",
            branch="main",
            changed_files=["src/foo.py", "src/bar.py"],
            lines_added=50,
            lines_deleted=20,
            walltime_sec=3.0,
            tests=tests,
            budget_sec=1.5,
            runner_usd_per_hour=0.60,
            timestamp=datetime.now(timezone.utc),
        )
        
        assert ci_run.commit == "abc123def456"
        assert len(ci_run.tests) == 3
        assert ci_run.walltime_sec == 3.0
    
    def test_ci_run_walltime_validation(self):
        """Test CIRun walltime consistency check."""
        tests = [
            TestResult(
                name=f"test_{i}",
                suite="generic",
                domain="generic",
                duration_sec=10.0,  # Total: 30s
                result="pass",
                cost_usd=0.001,
                timestamp=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]
        
        # Walltime=5s but tests sum to 30s (6x) → should fail validation
        with pytest.raises(ValidationError, match="Sum of test durations.*exceeds.*walltime"):
            CIRun(
                commit="abc1234",  # 7 chars minimum
                branch="main",
                walltime_sec=5.0,  # Too small!
                tests=tests,
                budget_sec=2.5,
                runner_usd_per_hour=0.60,
                timestamp=datetime.now(timezone.utc),
            )


class TestCalibrationMetrics:
    """Test model calibration computation."""
    
    def test_brier_score(self):
        """Test Brier score computation."""
        import numpy as np
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
        
        brier = calibration.compute_brier_score(y_true, y_prob)
        
        # Brier = mean((y_prob - y_true)^2)
        # = mean([0.1^2, 0.1^2, 0.2^2, 0.2^2, 0.3^2])
        # = mean([0.01, 0.01, 0.04, 0.04, 0.09])
        # = 0.038
        assert brier == pytest.approx(0.038, abs=0.001)
    
    def test_ece(self):
        """Test Expected Calibration Error."""
        import numpy as np
        
        # Imperfect calibration example
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        
        ece = calibration.compute_ece(y_true, y_prob, n_bins=10)
        
        # ECE should be reasonable (< 0.25 for this data)
        assert 0.0 <= ece < 0.25
        assert isinstance(ece, float)
    
    def test_generate_calibration_metrics(self):
        """Test complete calibration metrics generation."""
        import numpy as np
        
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.15, 0.85, 0.95, 0.05, 0.75])
        
        metrics = calibration.generate_calibration_metrics(y_true, y_prob, n_bins=5)
        
        assert "brier_score" in metrics
        assert "ece" in metrics
        assert "mce" in metrics
        assert "model_confidence_mean" in metrics
        assert metrics["n_samples"] == 10


class TestEpistemicMetrics:
    """Test epistemic efficiency metrics."""
    
    def test_bernoulli_entropy(self):
        """Test Bernoulli entropy computation."""
        # H(0.5) = 1 bit (maximum entropy)
        assert bernoulli_entropy(0.5) == pytest.approx(1.0, abs=0.001)
        
        # H(0) = H(1) = 0 (no entropy)
        assert bernoulli_entropy(0.0) == 0.0
        assert bernoulli_entropy(1.0) == 0.0
        
        # H(0.25) ≈ 0.811 bits
        assert bernoulli_entropy(0.25) == pytest.approx(0.811, abs=0.01)
    
    def test_expected_information_gain(self):
        """Test EIG computation."""
        # High uncertainty (p=0.5), low cost → high EIG
        eig_high = compute_expected_information_gain(failure_prob=0.5, cost_usd=0.001)
        
        # Low uncertainty (p=0.1), low cost → lower EIG
        eig_low = compute_expected_information_gain(failure_prob=0.1, cost_usd=0.001)
        
        assert eig_high > eig_low
        
        # Same uncertainty, higher cost → lower EIG
        eig_expensive = compute_expected_information_gain(failure_prob=0.5, cost_usd=0.01)
        assert eig_expensive < eig_high
    
    def test_detection_rate(self):
        """Test detection rate computation."""
        selected_tests = [
            {"model_uncertainty": 0.5},
            {"model_uncertainty": 0.3},
        ]
        
        all_tests = [
            {"model_uncertainty": 0.5},
            {"model_uncertainty": 0.3},
            {"model_uncertainty": 0.1},
            {"model_uncertainty": 0.1},
        ]
        
        detection_rate = compute_detection_rate(selected_tests, all_tests)
        
        # Selected: 0.5 + 0.3 = 0.8
        # Total: 0.5 + 0.3 + 0.1 + 0.1 = 1.0
        # Detection rate = 0.8 / 1.0 = 0.8
        assert detection_rate == pytest.approx(0.8, abs=0.01)
    
    def test_compute_epistemic_efficiency(self):
        """Test comprehensive epistemic efficiency metrics."""
        selected_tests = [
            {"eig_bits": 0.5, "cost_usd": 0.001, "duration_sec": 1.0, "model_uncertainty": 0.5},
            {"eig_bits": 0.3, "cost_usd": 0.002, "duration_sec": 2.0, "model_uncertainty": 0.3},
        ]
        
        all_tests = selected_tests + [
            {"eig_bits": 0.1, "cost_usd": 0.001, "duration_sec": 1.0, "model_uncertainty": 0.1},
            {"eig_bits": 0.1, "cost_usd": 0.001, "duration_sec": 1.0, "model_uncertainty": 0.1},
        ]
        
        metrics = compute_epistemic_efficiency(selected_tests, all_tests)
        
        assert metrics["tests_selected"] == 2
        assert metrics["tests_total"] == 4
        assert metrics["selection_fraction"] == 0.5
        assert metrics["bits_gained"] == 0.8
        assert metrics["detection_rate"] > 0
        assert metrics["cost_reduction_pct"] > 0
    
    def test_enrich_tests_with_epistemic_features(self):
        """Test epistemic feature enrichment."""
        tests = [
            {"name": "test_1", "cost_usd": 0.001},
            {"name": "test_2", "cost_usd": 0.002},
        ]
        
        model_predictions = {
            "test_1": 0.3,
            "test_2": 0.6,
        }
        
        enriched = enrich_tests_with_epistemic_features(tests, model_predictions)
        
        assert len(enriched) == 2
        assert "model_uncertainty" in enriched[0]
        assert "eig_bits" in enriched[0]
        assert "entropy_before" in enriched[0]
        assert enriched[0]["model_uncertainty"] == 0.3
        assert enriched[1]["model_uncertainty"] == 0.6


class TestDatasetValidation:
    """Test dataset contract validation."""
    
    def test_checksum_computation(self):
        """Test checksum computation for files."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test data\n")
            temp_path = pathlib.Path(f.name)
        
        try:
            # Compute checksum
            with temp_path.open("rb") as f:
                expected_checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Now compute with validation script
            sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
            from validate_datasets import DatasetContractValidator
            
            # Create a minimal manifest
            manifest = {
                "datasets": {
                    "test_dataset": {
                        "path": str(temp_path),
                        "checksum_type": "sha256",
                        "checksum": expected_checksum,
                    }
                },
                "validation": {
                    "enforce_checksums": True,
                    "block_on_mismatch": True,
                },
            }
            
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as manifest_file:
                import yaml
                yaml.dump(manifest, manifest_file)
                manifest_path = pathlib.Path(manifest_file.name)
            
            try:
                validator = DatasetContractValidator(manifest_path=manifest_path)
                result = validator.validate_dataset("test_dataset")
                
                assert result is True
                assert len(validator.errors) == 0
            
            finally:
                manifest_path.unlink()
        
        finally:
            temp_path.unlink()


class TestDoubleBuildVerification:
    """Test reproducible double builds."""
    
    def test_deterministic_mock_data_generation(self):
        """Test that mock data generation with same seed is bit-identical."""
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
        from collect_ci_runs import generate_mock_run
        
        import random
        from datetime import datetime, timezone
        
        # Fixed timestamp for deterministic generation
        fixed_timestamp = datetime(2025, 10, 8, 0, 0, 0, tzinfo=timezone.utc)
        
        # Generate first run
        random.seed(42)
        run1 = generate_mock_run(n_tests=10, failure_prob=0.1, base_timestamp=fixed_timestamp)
        
        # Generate second run with same seed
        random.seed(42)
        run2 = generate_mock_run(n_tests=10, failure_prob=0.1, base_timestamp=fixed_timestamp)
        
        # Convert to JSON and compare
        json1 = json.dumps(run1, sort_keys=True)
        json2 = json.dumps(run2, sort_keys=True)
        
        assert json1 == json2, "Mock data generation is not deterministic!"
    
    def test_artifact_hashing(self):
        """Test artifact hash computation for hermetic builds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = pathlib.Path(tmpdir) / "artifacts"
            artifact_dir.mkdir()
            
            # Create some artifacts
            (artifact_dir / "file1.txt").write_text("content1")
            (artifact_dir / "file2.txt").write_text("content2")
            
            # Compute hash
            def compute_artifact_hash(path: pathlib.Path) -> str:
                hasher = hashlib.sha256()
                for file_path in sorted(path.rglob("*")):
                    if file_path.is_file():
                        with file_path.open("rb") as f:
                            hasher.update(f.read())
                return hasher.hexdigest()
            
            hash1 = compute_artifact_hash(artifact_dir)
            hash2 = compute_artifact_hash(artifact_dir)
            
            assert hash1 == hash2, "Artifact hashing is not deterministic!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
