"""Tests for CI gates script."""

import json
import pathlib
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from ci_gates import check_coverage_gate, check_calibration_gates, check_epistemic_gates, GateResult
from _config import get_thresholds


def test_gate_result():
    """Test GateResult creation."""
    result = GateResult("test", True, 90.0, 85.0, "Coverage test")
    assert result.name == "test"
    assert result.passed is True
    assert result.value == 90.0
    assert result.threshold == 85.0


def test_coverage_gate_pass():
    """Test coverage gate with passing coverage."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        coverage_data = {
            "totals": {
                "percent_covered": 90.5
            }
        }
        json.dump(coverage_data, f)
        temp_file = pathlib.Path(f.name)
    
    try:
        result = check_coverage_gate(temp_file, 85.0)
        assert result.passed is True
        assert result.value == 90.5
        assert result.threshold == 85.0
    finally:
        temp_file.unlink()


def test_coverage_gate_fail():
    """Test coverage gate with failing coverage."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        coverage_data = {
            "totals": {
                "percent_covered": 75.0
            }
        }
        json.dump(coverage_data, f)
        temp_file = pathlib.Path(f.name)
    
    try:
        result = check_coverage_gate(temp_file, 85.0)
        assert result.passed is False
        assert result.value == 75.0
    finally:
        temp_file.unlink()


def test_coverage_gate_missing_file():
    """Test coverage gate with missing file."""
    result = check_coverage_gate(pathlib.Path("/nonexistent/coverage.json"), 85.0)
    assert result.passed is False
    assert "not found" in result.message


def test_calibration_gates():
    """Test calibration gates with valid ledger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_dir = pathlib.Path(tmpdir)
        ledger_file = ledger_dir / "run_001.jsonl"
        
        entry = {
            "timestamp": "2025-10-08T00:00:00Z",
            "calibration": {
                "ece": 0.15,
                "brier_score": 0.10,
                "mce": 0.20
            }
        }
        
        with ledger_file.open("w") as f:
            f.write(json.dumps(entry) + "\n")
        
        thresholds = get_thresholds()
        results = check_calibration_gates(ledger_dir, thresholds)
        
        assert len(results) == 3
        assert all(r.passed for r in results)  # All should pass with these values


def test_epistemic_gates():
    """Test epistemic gates with valid ledger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_dir = pathlib.Path(tmpdir)
        ledger_file = ledger_dir / "run_001.jsonl"
        
        entry = {
            "timestamp": "2025-10-08T00:00:00Z",
            "uncertainty": {
                "entropy_before": 0.5,
                "entropy_after": 0.4
            },
            "selector_metrics": {
                "avg_eig_bits": 0.05
            }
        }
        
        with ledger_file.open("w") as f:
            f.write(json.dumps(entry) + "\n")
        
        thresholds = get_thresholds()
        results = check_epistemic_gates(ledger_dir, thresholds)
        
        assert len(results) == 2
        # Both should pass (entropy delta < 0.15, avg EIG > 0.01)
        assert all(r.passed for r in results)
