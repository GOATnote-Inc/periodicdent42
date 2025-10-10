"""Tests for scripts/ci_gates.py - CI quality gate enforcement."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ci_gates import (
    GateResult,
    check_coverage_gate,
    check_calibration_gates,
    check_epistemic_gates,
    print_gate_summary,
)


class TestGateResult:
    """Test GateResult class."""
    
    def test_init(self):
        """Test GateResult initialization."""
        result = GateResult("Coverage", True, 85.5, 80.0, "Good coverage")
        
        assert result.name == "Coverage"
        assert result.passed == True
        assert result.value == 85.5
        assert result.threshold == 80.0
        assert result.message == "Good coverage"
    
    def test_repr_passed(self):
        """Test repr for passed gate."""
        result = GateResult("Coverage", True, 85.5, 80.0, "Good")
        repr_str = repr(result)
        
        assert "✅ PASS" in repr_str
        assert "Coverage" in repr_str
        assert "85.5" in repr_str
        assert "80.0" in repr_str
    
    def test_repr_failed(self):
        """Test repr for failed gate."""
        result = GateResult("Coverage", False, 75.0, 80.0, "Low")
        repr_str = repr(result)
        
        assert "❌ FAIL" in repr_str
        assert "Coverage" in repr_str
        assert "75.0" in repr_str


class TestCheckCoverageGate:
    """Test check_coverage_gate function."""
    
    def test_coverage_passes(self):
        """Test coverage gate when coverage meets threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coverage_file = Path(tmpdir) / "coverage.json"
            coverage_file.write_text(json.dumps({
                "totals": {"percent_covered": 85.5}
            }))
            
            result = check_coverage_gate(coverage_file, min_coverage=80.0)
            
            assert result.passed == True
            assert result.value == 85.5
            assert result.threshold == 80.0
    
    def test_coverage_fails(self):
        """Test coverage gate when coverage below threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coverage_file = Path(tmpdir) / "coverage.json"
            coverage_file.write_text(json.dumps({
                "totals": {"percent_covered": 75.0}
            }))
            
            result = check_coverage_gate(coverage_file, min_coverage=80.0)
            
            assert result.passed == False
            assert result.value == 75.0
            assert result.threshold == 80.0
    
    def test_coverage_file_missing(self):
        """Test coverage gate when file doesn't exist."""
        missing_file = Path("/nonexistent/coverage.json")
        
        result = check_coverage_gate(missing_file, min_coverage=80.0)
        
        assert result.passed == False
        assert result.value == 0.0
        assert "not found" in result.message
    
    def test_coverage_malformed_json(self):
        """Test coverage gate with malformed JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coverage_file = Path(tmpdir) / "coverage.json"
            coverage_file.write_text("not valid json")
            
            result = check_coverage_gate(coverage_file, min_coverage=80.0)
            
            assert result.passed == False
            assert "Error" in result.message
    
    def test_coverage_missing_totals(self):
        """Test coverage gate with missing totals key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coverage_file = Path(tmpdir) / "coverage.json"
            coverage_file.write_text(json.dumps({"other_key": "value"}))
            
            result = check_coverage_gate(coverage_file, min_coverage=80.0)
            
            assert result.passed == False
            assert result.value == 0.0


class TestCheckCalibrationGates:
    """Test check_calibration_gates function."""
    
    def test_calibration_all_pass(self):
        """Test calibration gates when all metrics pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_dir = Path(tmpdir)
            ledger_file = ledger_dir / "ledger_20250101.jsonl"
            
            entry = {
                "calibration": {
                    "ece": 0.05,
                    "brier_score": 0.10,
                    "mce": 0.12
                }
            }
            ledger_file.write_text(json.dumps(entry) + "\n")
            
            thresholds = {"ece": 0.10, "brier": 0.15, "mce": 0.20}
            results = check_calibration_gates(ledger_dir, thresholds)
            
            assert len(results) == 3
            assert all(r.passed for r in results)
    
    def test_calibration_some_fail(self):
        """Test calibration gates when some metrics fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_dir = Path(tmpdir)
            ledger_file = ledger_dir / "ledger_20250101.jsonl"
            
            entry = {
                "calibration": {
                    "ece": 0.15,  # Exceeds threshold
                    "brier_score": 0.10,
                    "mce": 0.25  # Exceeds threshold
                }
            }
            ledger_file.write_text(json.dumps(entry) + "\n")
            
            thresholds = {"ece": 0.10, "brier": 0.15, "mce": 0.20}
            results = check_calibration_gates(ledger_dir, thresholds)
            
            assert len(results) == 3
            assert results[0].passed == False  # ECE failed
            assert results[1].passed == True   # Brier passed
            assert results[2].passed == False  # MCE failed
    
    def test_calibration_no_ledger_files(self):
        """Test calibration gates when no ledger files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_dir = Path(tmpdir)
            
            thresholds = {"ece": 0.10, "brier": 0.15, "mce": 0.20}
            results = check_calibration_gates(ledger_dir, thresholds)
            
            assert len(results) == 3
            assert all(not r.passed for r in results)
            assert all("No ledger files" in r.message for r in results)
    
    def test_calibration_empty_ledger(self):
        """Test calibration gates with empty ledger file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_dir = Path(tmpdir)
            ledger_file = ledger_dir / "ledger_20250101.jsonl"
            ledger_file.touch()  # Empty file
            
            thresholds = {"ece": 0.10, "brier": 0.15, "mce": 0.20}
            results = check_calibration_gates(ledger_dir, thresholds)
            
            assert len(results) == 3
            assert all(not r.passed for r in results)
            assert all("Empty ledger" in r.message for r in results)


class TestCheckEpistemicGates:
    """Test check_epistemic_gates function."""
    
    def test_epistemic_all_pass(self):
        """Test epistemic gates when all metrics pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_dir = Path(tmpdir)
            ledger_file = ledger_dir / "ledger_20250101.jsonl"
            
            entry = {
                "uncertainty": {
                    "entropy_before": 2.5,
                    "entropy_after": 2.0
                },
                "selector_metrics": {
                    "avg_eig_bits": 0.8
                }
            }
            ledger_file.write_text(json.dumps(entry) + "\n")
            
            thresholds = {"entropy_delta": 1.0, "eig_min": 0.5}
            results = check_epistemic_gates(ledger_dir, thresholds)
            
            assert len(results) == 2
            assert all(r.passed for r in results)
    
    def test_epistemic_entropy_delta_fails(self):
        """Test epistemic gates when entropy delta exceeds threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_dir = Path(tmpdir)
            ledger_file = ledger_dir / "ledger_20250101.jsonl"
            
            entry = {
                "uncertainty": {
                    "entropy_before": 3.0,
                    "entropy_after": 0.5  # Delta = 2.5
                },
                "selector_metrics": {
                    "avg_eig_bits": 0.8
                }
            }
            ledger_file.write_text(json.dumps(entry) + "\n")
            
            thresholds = {"entropy_delta": 1.0, "eig_min": 0.5}
            results = check_epistemic_gates(ledger_dir, thresholds)
            
            assert len(results) == 2
            assert results[0].passed == False  # Entropy delta failed
            assert results[1].passed == True   # EIG passed
    
    def test_epistemic_no_ledger(self):
        """Test epistemic gates with no ledger (should skip gracefully)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_dir = Path(tmpdir)
            
            thresholds = {"entropy_delta": 1.0, "eig_min": 0.5}
            results = check_epistemic_gates(ledger_dir, thresholds)
            
            assert len(results) == 2
            # No ledger means we skip (pass with warning)
            assert all(r.passed for r in results)
            assert all("skip" in r.message.lower() for r in results)


class TestPrintGateSummary:
    """Test print_gate_summary function."""
    
    def test_summary_all_passed(self, capsys):
        """Test summary printing when all gates pass."""
        results = [
            GateResult("Coverage", True, 85.0, 80.0, "Good"),
            GateResult("ECE", True, 0.05, 0.10, "Good"),
            GateResult("Brier", True, 0.08, 0.15, "Good"),
        ]
        
        passed, failed = print_gate_summary(results)
        
        assert passed == 3
        assert failed == 0
        
        captured = capsys.readouterr()
        assert "SUMMARY: 3 passed, 0 failed" in captured.out
    
    def test_summary_some_failed(self, capsys):
        """Test summary printing when some gates fail."""
        results = [
            GateResult("Coverage", True, 85.0, 80.0, "Good"),
            GateResult("ECE", False, 0.15, 0.10, "Bad"),
            GateResult("Brier", True, 0.08, 0.15, "Good"),
        ]
        
        passed, failed = print_gate_summary(results)
        
        assert passed == 2
        assert failed == 1
        
        captured = capsys.readouterr()
        assert "SUMMARY: 2 passed, 1 failed" in captured.out
    
    def test_summary_formatting(self, capsys):
        """Test that summary includes result details."""
        results = [
            GateResult("Coverage", True, 85.5, 80.0, "Test message"),
        ]
        
        print_gate_summary(results)
        
        captured = capsys.readouterr()
        assert "Coverage" in captured.out
        assert "85.5" in captured.out
        assert "✅ PASS" in captured.out
