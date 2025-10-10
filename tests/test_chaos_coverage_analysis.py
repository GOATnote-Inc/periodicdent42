#!/usr/bin/env python3
"""Tests for scripts/chaos_coverage_analysis.py

Comprehensive test suite for chaos coverage analysis.
"""

import json
import pytest
from pathlib import Path
from scripts.chaos_coverage_analysis import (
    Incident,
    ChaosTest,
    generate_synthetic_incidents,
    load_chaos_tests,
    map_incidents_to_tests,
    print_report
)


class TestIncident:
    """Test Incident dataclass."""
    
    def test_incident_creation(self):
        """Test creating an incident."""
        incident = Incident(
            id="INC-001",
            date="2025-01-01",
            category="network",
            description="Test incident",
            root_cause="Test root cause",
            impact="Test impact"
        )
        assert incident.id == "INC-001"
        assert incident.category == "network"
    
    def test_incident_equality(self):
        """Test incident equality."""
        inc1 = Incident("INC-001", "2025-01-01", "network", "desc", "cause", "impact")
        inc2 = Incident("INC-001", "2025-01-01", "network", "desc", "cause", "impact")
        assert inc1 == inc2


class TestChaosTest:
    """Test ChaosTest dataclass."""
    
    def test_chaos_test_creation(self):
        """Test creating a chaos test."""
        test = ChaosTest(
            name="test_example",
            failure_type="network",
            description="Test description",
            file_path="tests/chaos/test_example.py"
        )
        assert test.name == "test_example"
        assert test.failure_type == "network"
    
    def test_chaos_test_equality(self):
        """Test chaos test equality."""
        t1 = ChaosTest("test1", "network", "desc", "path")
        t2 = ChaosTest("test1", "network", "desc", "path")
        assert t1 == t2


class TestGenerateSyntheticIncidents:
    """Test generate_synthetic_incidents function."""
    
    def test_generates_incidents(self):
        """Test that synthetic incidents are generated."""
        incidents = generate_synthetic_incidents()
        assert len(incidents) > 0
        assert all(isinstance(inc, Incident) for inc in incidents)
    
    def test_incidents_have_required_fields(self):
        """Test all incidents have required fields."""
        incidents = generate_synthetic_incidents()
        for inc in incidents:
            assert inc.id
            assert inc.date
            assert inc.category
            assert inc.description
            assert inc.root_cause
            assert inc.impact
    
    def test_incidents_cover_failure_types(self):
        """Test incidents cover multiple failure types."""
        incidents = generate_synthetic_incidents()
        categories = {inc.category for inc in incidents}
        # Should have multiple categories
        assert len(categories) >= 3
        # Should include common types
        assert "network" in categories or "database" in categories
    
    def test_consistent_count(self):
        """Test generates consistent number of incidents."""
        count1 = len(generate_synthetic_incidents())
        count2 = len(generate_synthetic_incidents())
        assert count1 == count2  # Should be deterministic


class TestLoadChaosTests:
    """Test load_chaos_tests function."""
    
    def test_loads_chaos_tests(self):
        """Test that chaos tests are loaded."""
        tests = load_chaos_tests()
        assert len(tests) > 0
        assert all(isinstance(t, ChaosTest) for t in tests)
    
    def test_tests_have_required_fields(self):
        """Test all chaos tests have required fields."""
        tests = load_chaos_tests()
        for test in tests:
            assert test.name
            assert test.failure_type
            assert test.description
            assert test.file_path
    
    def test_tests_cover_failure_types(self):
        """Test chaos tests cover multiple failure types."""
        tests = load_chaos_tests()
        failure_types = {t.failure_type for t in tests}
        # Should have multiple types
        assert len(failure_types) >= 3
    
    def test_consistent_count(self):
        """Test loads consistent number of tests."""
        count1 = len(load_chaos_tests())
        count2 = len(load_chaos_tests())
        assert count1 == count2  # Should be deterministic


class TestMapIncidentsToTests:
    """Test map_incidents_to_tests function."""
    
    def test_empty_incidents_and_tests(self):
        """Test with empty lists."""
        result = map_incidents_to_tests([], [])
        assert result["total_incidents"] == 0
        assert result["mapped_incidents"] == 0
        assert result["unmapped_incidents"] == 0
        assert result["coverage_pct"] == 0
    
    def test_incidents_without_tests(self):
        """Test incidents with no matching tests."""
        incidents = [Incident("INC-001", "2025-01-01", "network", "desc", "cause", "impact")]
        result = map_incidents_to_tests(incidents, [])
        assert result["total_incidents"] == 1
        assert result["mapped_incidents"] == 0
        assert result["unmapped_incidents"] == 1
        assert result["coverage_pct"] == 0.0
    
    def test_incidents_with_matching_tests(self):
        """Test incidents with matching tests."""
        incidents = [Incident("INC-001", "2025-01-01", "network", "desc", "cause", "impact")]
        tests = [ChaosTest("test_network", "network", "desc", "path")]
        result = map_incidents_to_tests(incidents, tests)
        assert result["total_incidents"] == 1
        assert result["mapped_incidents"] == 1
        assert result["unmapped_incidents"] == 0
        assert result["coverage_pct"] == 100.0
    
    def test_partial_coverage(self):
        """Test partial test coverage."""
        incidents = [
            Incident("INC-001", "2025-01-01", "network", "desc1", "cause1", "impact1"),
            Incident("INC-002", "2025-01-02", "database", "desc2", "cause2", "impact2")
        ]
        tests = [ChaosTest("test_network", "network", "desc", "path")]
        result = map_incidents_to_tests(incidents, tests)
        assert result["total_incidents"] == 2
        assert result["mapped_incidents"] == 1
        assert result["unmapped_incidents"] == 1
        assert result["coverage_pct"] == 50.0
    
    def test_multiple_tests_per_category(self):
        """Test multiple tests covering same category."""
        incidents = [Incident("INC-001", "2025-01-01", "network", "desc", "cause", "impact")]
        tests = [
            ChaosTest("test_network_1", "network", "desc1", "path1"),
            ChaosTest("test_network_2", "network", "desc2", "path2")
        ]
        result = map_incidents_to_tests(incidents, tests)
        assert result["mapped_incidents"] == 1
        assert len(result["mapped"][0]["covered_by"]) == 2
    
    def test_test_coverage_summary(self):
        """Test test coverage summary structure."""
        incidents = generate_synthetic_incidents()
        tests = load_chaos_tests()
        result = map_incidents_to_tests(incidents, tests)
        
        assert "test_coverage" in result
        assert isinstance(result["test_coverage"], dict)
        # Should show count per category
        for category, count in result["test_coverage"].items():
            assert isinstance(count, int)
            assert count > 0
    
    def test_gap_categories_identified(self):
        """Test gap categories are identified."""
        incidents = [
            Incident("INC-001", "2025-01-01", "network", "desc1", "cause1", "impact1"),
            Incident("INC-002", "2025-01-02", "newtype", "desc2", "cause2", "impact2")
        ]
        tests = [ChaosTest("test_network", "network", "desc", "path")]
        result = map_incidents_to_tests(incidents, tests)
        
        assert "gap_categories" in result
        assert "newtype" in result["gap_categories"]
    
    def test_real_data_coverage(self):
        """Test with real synthetic data."""
        incidents = generate_synthetic_incidents()
        tests = load_chaos_tests()
        result = map_incidents_to_tests(incidents, tests)
        
        # Should have reasonable coverage
        assert result["coverage_pct"] >= 0
        assert result["total_incidents"] == len(incidents)
        assert result["mapped_incidents"] + result["unmapped_incidents"] == len(incidents)


class TestPrintReport:
    """Test print_report function."""
    
    def test_print_report_no_error(self, capsys):
        """Test that print_report doesn't raise errors."""
        analysis = {
            "total_incidents": 2,
            "mapped_incidents": 1,
            "unmapped_incidents": 1,
            "coverage_pct": 50.0,
            "mapped": [
                {
                    "incident": Incident("INC-001", "2025-01-01", "network", "desc", "cause", "impact"),
                    "covered_by": ["test_network"],
                    "status": "✅ Covered"
                }
            ],
            "unmapped": [
                {
                    "incident": Incident("INC-002", "2025-01-02", "database", "desc", "cause", "impact"),
                    "status": "⚠️  Not Covered"
                }
            ],
            "gap_categories": ["database"],
            "test_coverage": {"network": 1}
        }
        
        # Should not raise
        print_report(analysis)
        
        # Verify output contains expected elements
        captured = capsys.readouterr()
        assert "CHAOS ENGINEERING COVERAGE ANALYSIS" in captured.out
        assert "50.0%" in captured.out
    
    def test_print_complete_coverage(self, capsys):
        """Test report with 100% coverage."""
        analysis = {
            "total_incidents": 1,
            "mapped_incidents": 1,
            "unmapped_incidents": 0,
            "coverage_pct": 100.0,
            "mapped": [
                {
                    "incident": Incident("INC-001", "2025-01-01", "network", "desc", "cause", "impact"),
                    "covered_by": ["test_network"],
                    "status": "✅ Covered"
                }
            ],
            "unmapped": [],
            "gap_categories": [],
            "test_coverage": {"network": 1}
        }
        
        print_report(analysis)
        captured = capsys.readouterr()
        assert "100.0%" in captured.out
        assert "Complete Coverage" in captured.out

