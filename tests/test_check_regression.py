#!/usr/bin/env python3
"""Tests for scripts/check_regression.py

Comprehensive test suite for result regression detection.
"""

import json
import pytest
import tempfile
from pathlib import Path
from scripts.check_regression import (
    RegressionResult,
    RegressionReport,
    RegressionChecker
)


class TestRegressionResult:
    """Test RegressionResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a regression result."""
        result = RegressionResult(
            field="test_field",
            current_value=1.0,
            baseline_value=1.0,
            difference=0.0,
            relative_diff=0.0,
            passed=True,
            tolerance=1e-10
        )
        assert result.field == "test_field"
        assert result.passed is True
    
    def test_result_failed(self):
        """Test failed regression result."""
        result = RegressionResult(
            field="test_field",
            current_value=2.0,
            baseline_value=1.0,
            difference=1.0,
            relative_diff=1.0,
            passed=False,
            tolerance=1e-10
        )
        assert result.passed is False
        assert result.difference == 1.0


class TestRegressionReport:
    """Test RegressionReport dataclass."""
    
    def test_report_creation(self):
        """Test creating a regression report."""
        results = [
            RegressionResult("field1", 1.0, 1.0, 0.0, 0.0, True, 1e-10)
        ]
        report = RegressionReport(
            passed=True,
            total_checks=1,
            failed_checks=0,
            results=results,
            tolerance=1e-10,
            current_file="current.json",
            baseline_file="baseline.json"
        )
        assert report.passed is True
        assert report.total_checks == 1
    
    def test_report_with_failures(self):
        """Test report with failures."""
        results = [
            RegressionResult("field1", 1.0, 1.0, 0.0, 0.0, True, 1e-10),
            RegressionResult("field2", 2.0, 1.0, 1.0, 1.0, False, 1e-10)
        ]
        report = RegressionReport(
            passed=False,
            total_checks=2,
            failed_checks=1,
            results=results,
            tolerance=1e-10,
            current_file="current.json",
            baseline_file="baseline.json"
        )
        assert report.passed is False
        assert report.failed_checks == 1


class TestRegressionCheckerInit:
    """Test RegressionChecker initialization."""
    
    def test_default_tolerance(self):
        """Test default tolerance."""
        checker = RegressionChecker()
        assert checker.tolerance == 1e-10
    
    def test_custom_tolerance(self):
        """Test custom tolerance."""
        checker = RegressionChecker(tolerance=1e-5)
        assert checker.tolerance == 1e-5


class TestLoadJson:
    """Test load_json method."""
    
    def test_load_valid_json(self, tmp_path):
        """Test loading valid JSON."""
        data = {"key": "value", "number": 42}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        checker = RegressionChecker()
        loaded = checker.load_json(json_file)
        assert loaded == data
    
    def test_load_nested_json(self, tmp_path):
        """Test loading nested JSON."""
        data = {"nested": {"key": "value", "number": 42}}
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        checker = RegressionChecker()
        loaded = checker.load_json(json_file)
        assert loaded == data


class TestExtractNumericalFields:
    """Test extract_numerical_fields method."""
    
    def test_extract_flat_fields(self):
        """Test extracting flat numerical fields."""
        checker = RegressionChecker()
        data = {"a": 1, "b": 2.5, "c": "string"}
        fields = checker.extract_numerical_fields(data)
        assert fields == {"a": 1.0, "b": 2.5}
    
    def test_extract_nested_fields(self):
        """Test extracting nested fields."""
        checker = RegressionChecker()
        data = {"outer": {"inner": 42}}
        fields = checker.extract_numerical_fields(data)
        assert fields == {"outer.inner": 42.0}
    
    def test_extract_list_fields(self):
        """Test extracting fields from lists."""
        checker = RegressionChecker()
        data = {"values": [1, 2, 3]}
        fields = checker.extract_numerical_fields(data)
        assert fields == {
            "values[0]": 1.0,
            "values[1]": 2.0,
            "values[2]": 3.0
        }
    
    def test_extract_deeply_nested(self):
        """Test deeply nested extraction."""
        checker = RegressionChecker()
        data = {
            "level1": {
                "level2": {
                    "value": 3.14
                }
            }
        }
        fields = checker.extract_numerical_fields(data)
        assert fields == {"level1.level2.value": 3.14}
    
    def test_extract_mixed_types(self):
        """Test extraction with mixed types."""
        checker = RegressionChecker()
        data = {
            "number": 42,
            "string": "text",
            "nested": {"value": 1.5},
            "list": [1, 2],
            "bool": True  # Booleans are extracted as 1.0/0.0
        }
        fields = checker.extract_numerical_fields(data)
        expected = {
            "number": 42.0,
            "nested.value": 1.5,
            "list[0]": 1.0,
            "list[1]": 2.0,
            "bool": 1.0  # True extracts as 1.0
        }
        assert fields == expected
    
    def test_empty_data(self):
        """Test with empty data."""
        checker = RegressionChecker()
        fields = checker.extract_numerical_fields({})
        assert fields == {}


class TestCompareValues:
    """Test compare_values method."""
    
    def test_equal_values(self):
        """Test comparing equal values."""
        checker = RegressionChecker(tolerance=1e-10)
        result = checker.compare_values("field", 1.0, 1.0, 1e-10)
        assert result.passed is True
        assert result.difference == 0.0
    
    def test_within_tolerance(self):
        """Test values within tolerance."""
        checker = RegressionChecker(tolerance=1e-4)
        result = checker.compare_values("field", 1.00001, 1.0, 1e-4)
        assert result.passed is True
    
    def test_exceeds_tolerance(self):
        """Test values exceeding tolerance."""
        checker = RegressionChecker(tolerance=1e-10)
        result = checker.compare_values("field", 2.0, 1.0, 1e-10)
        assert result.passed is False
        assert result.difference == 1.0
    
    def test_relative_difference(self):
        """Test relative difference calculation."""
        checker = RegressionChecker()
        result = checker.compare_values("field", 2.0, 1.0, 1e-10)
        assert result.relative_diff == 1.0  # 100% difference
    
    def test_zero_baseline(self):
        """Test with zero baseline."""
        checker = RegressionChecker()
        result = checker.compare_values("field", 1.0, 0.0, 1e-10)
        assert result.relative_diff == 1.0
    
    def test_both_zero(self):
        """Test with both values zero."""
        checker = RegressionChecker()
        result = checker.compare_values("field", 0.0, 0.0, 1e-10)
        assert result.passed is True
        assert result.relative_diff == 0.0


class TestCheckRegression:
    """Test check_regression method."""
    
    def test_identical_files(self, tmp_path):
        """Test with identical files."""
        data = {"value": 1.0}
        current = tmp_path / "current.json"
        baseline = tmp_path / "baseline.json"
        
        with open(current, 'w') as f:
            json.dump(data, f)
        with open(baseline, 'w') as f:
            json.dump(data, f)
        
        checker = RegressionChecker()
        report = checker.check_regression(current, baseline)
        
        assert report.passed is True
        assert report.total_checks == 1
        assert report.failed_checks == 0
    
    def test_different_values(self, tmp_path):
        """Test with different values."""
        current_data = {"value": 2.0}
        baseline_data = {"value": 1.0}
        
        current = tmp_path / "current.json"
        baseline = tmp_path / "baseline.json"
        
        with open(current, 'w') as f:
            json.dump(current_data, f)
        with open(baseline, 'w') as f:
            json.dump(baseline_data, f)
        
        checker = RegressionChecker(tolerance=1e-10)
        report = checker.check_regression(current, baseline)
        
        assert report.passed is False
        assert report.failed_checks == 1
    
    def test_within_tolerance(self, tmp_path):
        """Test values within tolerance."""
        current_data = {"value": 1.00001}
        baseline_data = {"value": 1.0}
        
        current = tmp_path / "current.json"
        baseline = tmp_path / "baseline.json"
        
        with open(current, 'w') as f:
            json.dump(current_data, f)
        with open(baseline, 'w') as f:
            json.dump(baseline_data, f)
        
        checker = RegressionChecker(tolerance=1e-3)
        report = checker.check_regression(current, baseline)
        
        assert report.passed is True
    
    def test_multiple_fields(self, tmp_path):
        """Test with multiple fields."""
        current_data = {"a": 1.0, "b": 2.0, "c": 3.0}
        baseline_data = {"a": 1.0, "b": 2.0, "c": 3.0}
        
        current = tmp_path / "current.json"
        baseline = tmp_path / "baseline.json"
        
        with open(current, 'w') as f:
            json.dump(current_data, f)
        with open(baseline, 'w') as f:
            json.dump(baseline_data, f)
        
        checker = RegressionChecker()
        report = checker.check_regression(current, baseline)
        
        assert report.total_checks == 3
        assert report.passed is True
    
    def test_partial_regression(self, tmp_path):
        """Test partial regression (some fields match, some don't)."""
        current_data = {"a": 1.0, "b": 2.5, "c": 3.0}
        baseline_data = {"a": 1.0, "b": 2.0, "c": 3.0}
        
        current = tmp_path / "current.json"
        baseline = tmp_path / "baseline.json"
        
        with open(current, 'w') as f:
            json.dump(current_data, f)
        with open(baseline, 'w') as f:
            json.dump(baseline_data, f)
        
        checker = RegressionChecker(tolerance=1e-10)
        report = checker.check_regression(current, baseline)
        
        assert report.total_checks == 3
        assert report.failed_checks == 1
        assert report.passed is False
    
    def test_custom_tolerance_override(self, tmp_path):
        """Test custom tolerance override."""
        current_data = {"value": 1.5}
        baseline_data = {"value": 1.0}
        
        current = tmp_path / "current.json"
        baseline = tmp_path / "baseline.json"
        
        with open(current, 'w') as f:
            json.dump(current_data, f)
        with open(baseline, 'w') as f:
            json.dump(baseline_data, f)
        
        # Default tolerance would fail
        checker = RegressionChecker(tolerance=1e-10)
        report1 = checker.check_regression(current, baseline)
        assert report1.passed is False
        
        # Override with larger tolerance
        report2 = checker.check_regression(current, baseline, tolerance=1.0)
        assert report2.passed is True


class TestFormatReport:
    """Test format_report method."""
    
    def test_format_passed_report(self):
        """Test formatting a passed report."""
        checker = RegressionChecker()
        results = [
            RegressionResult("field1", 1.0, 1.0, 0.0, 0.0, True, 1e-10)
        ]
        report = RegressionReport(
            passed=True,
            total_checks=1,
            failed_checks=0,
            results=results,
            tolerance=1e-10,
            current_file="current.json",
            baseline_file="baseline.json"
        )
        
        formatted = checker.format_report(report)
        assert "✅ PASSED" in formatted
        assert "current.json" in formatted
        assert "baseline.json" in formatted
    
    def test_format_failed_report(self):
        """Test formatting a failed report."""
        checker = RegressionChecker()
        results = [
            RegressionResult("field1", 2.0, 1.0, 1.0, 1.0, False, 1e-10)
        ]
        report = RegressionReport(
            passed=False,
            total_checks=1,
            failed_checks=1,
            results=results,
            tolerance=1e-10,
            current_file="current.json",
            baseline_file="baseline.json"
        )
        
        formatted = checker.format_report(report)
        assert "❌ FAILED" in formatted
        assert "FAILED CHECKS:" in formatted
        assert "field1" in formatted


class TestExportJson:
    """Test export_json method."""
    
    def test_export_json(self, tmp_path):
        """Test exporting report as JSON."""
        checker = RegressionChecker()
        results = [
            RegressionResult("field1", 1.0, 1.0, 0.0, 0.0, True, 1e-10)
        ]
        report = RegressionReport(
            passed=True,
            total_checks=1,
            failed_checks=0,
            results=results,
            tolerance=1e-10,
            current_file="current.json",
            baseline_file="baseline.json"
        )
        
        output = tmp_path / "report.json"
        checker.export_json(report, output)
        
        assert output.exists()
        with open(output) as f:
            data = json.load(f)
        
        assert data["passed"] is True
        assert data["total_checks"] == 1
        assert len(data["results"]) == 1


class TestExportHtml:
    """Test export_html method."""
    
    def test_export_html(self, tmp_path):
        """Test exporting report as HTML."""
        checker = RegressionChecker()
        results = [
            RegressionResult("field1", 1.0, 1.0, 0.0, 0.0, True, 1e-10)
        ]
        report = RegressionReport(
            passed=True,
            total_checks=1,
            failed_checks=0,
            results=results,
            tolerance=1e-10,
            current_file="current.json",
            baseline_file="baseline.json"
        )
        
        output = tmp_path / "report.html"
        checker.export_html(report, output)
        
        assert output.exists()
        content = output.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Regression Detection Report" in content
    
    def test_html_includes_failed_checks(self, tmp_path):
        """Test HTML includes failed checks."""
        checker = RegressionChecker()
        results = [
            RegressionResult("field1", 2.0, 1.0, 1.0, 1.0, False, 1e-10)
        ]
        report = RegressionReport(
            passed=False,
            total_checks=1,
            failed_checks=1,
            results=results,
            tolerance=1e-10,
            current_file="current.json",
            baseline_file="baseline.json"
        )
        
        output = tmp_path / "report.html"
        checker.export_html(report, output)
        
        content = output.read_text()
        assert "Failed Checks" in content or "FAILED" in content


class TestIntegration:
    """Integration tests with real-world scenarios."""
    
    def test_real_validation_data(self, tmp_path):
        """Test with realistic validation data."""
        current_data = {
            "optimization": {
                "final_rmse": 0.123456,
                "runtime_sec": 45.67,
                "n_iterations": 100
            },
            "calibration": {
                "coverage_90": 0.89,
                "sharpness": 0.34
            }
        }
        baseline_data = {
            "optimization": {
                "final_rmse": 0.123456,
                "runtime_sec": 45.68,
                "n_iterations": 100
            },
            "calibration": {
                "coverage_90": 0.89,
                "sharpness": 0.34
            }
        }
        
        current = tmp_path / "current.json"
        baseline = tmp_path / "baseline.json"
        
        with open(current, 'w') as f:
            json.dump(current_data, f)
        with open(baseline, 'w') as f:
            json.dump(baseline_data, f)
        
        checker = RegressionChecker(tolerance=0.1)
        report = checker.check_regression(current, baseline)
        
        assert report.passed is True
        assert report.total_checks == 5  # All numerical fields
    
    def test_complete_workflow(self, tmp_path):
        """Test complete workflow: check + export."""
        current_data = {"value": 1.0}
        baseline_data = {"value": 1.0}
        
        current = tmp_path / "current.json"
        baseline = tmp_path / "baseline.json"
        
        with open(current, 'w') as f:
            json.dump(current_data, f)
        with open(baseline, 'w') as f:
            json.dump(baseline_data, f)
        
        checker = RegressionChecker()
        report = checker.check_regression(current, baseline)
        
        # Export both formats
        json_out = tmp_path / "report.json"
        html_out = tmp_path / "report.html"
        
        checker.export_json(report, json_out)
        checker.export_html(report, html_out)
        
        assert json_out.exists()
        assert html_out.exists()
        
        # Verify formatted output
        formatted = checker.format_report(report)
        assert "RESULT REGRESSION DETECTION REPORT" in formatted

