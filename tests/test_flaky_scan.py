"""Tests for scripts/flaky_scan.py - Flaky test detection."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from flaky_scan import (
    parse_junit_xml,
    compute_flip_count,
    scan_flaky_tests,
)


class TestParseJunitXML:
    """Test parse_junit_xml function."""
    
    def test_parse_passing_test(self):
        """Test parsing JUnit XML with passing test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "test.xml"
            xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1">
        <testcase classname="tests.test_example" name="test_pass" time="0.001"/>
    </testsuite>
</testsuites>
""")
            
            results = parse_junit_xml(xml_file)
            
            assert len(results) == 1
            assert "tests.test_example::test_pass" in results
            assert results["tests.test_example::test_pass"] == "pass"
    
    def test_parse_failing_test(self):
        """Test parsing JUnit XML with failing test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "test.xml"
            xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="0" failures="1" skipped="0" tests="1">
        <testcase classname="tests.test_example" name="test_fail" time="0.001">
            <failure message="assert False">AssertionError</failure>
        </testcase>
    </testsuite>
</testsuites>
""")
            
            results = parse_junit_xml(xml_file)
            
            assert results["tests.test_example::test_fail"] == "fail"
    
    def test_parse_error_test(self):
        """Test parsing JUnit XML with error test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "test.xml"
            xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="1" failures="0" skipped="0" tests="1">
        <testcase classname="tests.test_example" name="test_error" time="0.001">
            <error message="Exception">RuntimeError</error>
        </testcase>
    </testsuite>
</testsuites>
""")
            
            results = parse_junit_xml(xml_file)
            
            assert results["tests.test_example::test_error"] == "error"
    
    def test_parse_skipped_test(self):
        """Test parsing JUnit XML with skipped test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "test.xml"
            xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="0" failures="0" skipped="1" tests="1">
        <testcase classname="tests.test_example" name="test_skip" time="0.000">
            <skipped message="Skipped"/>
        </testcase>
    </testsuite>
</testsuites>
""")
            
            results = parse_junit_xml(xml_file)
            
            assert results["tests.test_example::test_skip"] == "skip"
    
    def test_parse_multiple_tests(self):
        """Test parsing JUnit XML with multiple tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "test.xml"
            xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" errors="0" failures="1" skipped="1" tests="3">
        <testcase classname="tests.test_a" name="test_pass" time="0.001"/>
        <testcase classname="tests.test_a" name="test_fail" time="0.001">
            <failure message="assert False">AssertionError</failure>
        </testcase>
        <testcase classname="tests.test_b" name="test_skip" time="0.000">
            <skipped message="Skipped"/>
        </testcase>
    </testsuite>
</testsuites>
""")
            
            results = parse_junit_xml(xml_file)
            
            assert len(results) == 3
            assert results["tests.test_a::test_pass"] == "pass"
            assert results["tests.test_a::test_fail"] == "fail"
            assert results["tests.test_b::test_skip"] == "skip"
    
    def test_parse_test_without_classname(self):
        """Test parsing test without classname."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "test.xml"
            xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" tests="1">
        <testcase name="test_no_class" time="0.001"/>
    </testsuite>
</testsuites>
""")
            
            results = parse_junit_xml(xml_file)
            
            assert "test_no_class" in results
            assert results["test_no_class"] == "pass"
    
    def test_parse_malformed_xml(self):
        """Test parsing malformed XML returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "test.xml"
            xml_file.write_text("not valid xml")
            
            results = parse_junit_xml(xml_file)
            
            assert results == {}
    
    def test_parse_missing_file(self):
        """Test parsing missing file returns empty dict."""
        missing_file = Path("/nonexistent/test.xml")
        
        results = parse_junit_xml(missing_file)
        
        assert results == {}
    
    def test_parse_empty_xml(self):
        """Test parsing empty XML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "test.xml"
            xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
</testsuites>
""")
            
            results = parse_junit_xml(xml_file)
            
            assert results == {}


class TestComputeFlipCount:
    """Test compute_flip_count function."""
    
    def test_no_flips_all_pass(self):
        """Test history with no flips (all pass)."""
        history = ["pass", "pass", "pass", "pass"]
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 0
    
    def test_no_flips_all_fail(self):
        """Test history with no flips (all fail)."""
        history = ["fail", "fail", "fail"]
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 0
    
    def test_one_flip(self):
        """Test history with one flip."""
        history = ["pass", "fail", "fail", "fail"]
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 1
    
    def test_multiple_flips(self):
        """Test history with multiple flips."""
        history = ["pass", "fail", "pass", "fail", "pass"]
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 4
    
    def test_alternating_flips(self):
        """Test history with alternating pass/fail."""
        history = ["pass", "fail", "pass", "fail", "pass", "fail"]
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 5
    
    def test_single_entry(self):
        """Test history with single entry (no flips possible)."""
        history = ["pass"]
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 0
    
    def test_empty_history(self):
        """Test empty history."""
        history = []
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 0
    
    def test_skip_to_pass_counts_as_flip(self):
        """Test that skip to pass counts as flip."""
        history = ["skip", "pass", "skip"]
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 2
    
    def test_error_to_pass_counts_as_flip(self):
        """Test that error to pass counts as flip."""
        history = ["error", "pass", "error"]
        
        flip_count = compute_flip_count(history)
        
        assert flip_count == 2


class TestScanFlakyTests:
    """Test scan_flaky_tests function."""
    
    def test_scan_no_xml_files(self):
        """Test scanning directory with no XML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            
            summary, flaky_tests = scan_flaky_tests(tests_dir)
            
            assert summary["total_tests"] == 0
            assert summary["flaky_tests"] == 0
            assert summary["window"] == 0
            assert flaky_tests == []
    
    def test_scan_stable_test(self):
        """Test scanning stable test (no flips)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            
            # Create 3 XML files with same passing test
            for i in range(3):
                xml_file = tests_dir / f"test_{i:03d}.xml"
                xml_file.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" tests="1">
        <testcase classname="tests.test_stable" name="test_always_pass" time="0.001"/>
    </testsuite>
</testsuites>
""")
            
            summary, flaky_tests = scan_flaky_tests(tests_dir, window=10, flip_threshold=2)
            
            assert summary["total_tests"] == 1
            assert summary["flaky_tests"] == 0
            assert flaky_tests == []
    
    def test_scan_flaky_test(self):
        """Test scanning flaky test (multiple flips)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            
            # Create XML files with alternating pass/fail
            statuses = ["pass", "fail", "pass", "fail"]
            for i, status in enumerate(statuses):
                xml_file = tests_dir / f"test_{i:03d}.xml"
                failure = '<failure message="fail">AssertionError</failure>' if status == "fail" else ''
                xml_file.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" tests="1">
        <testcase classname="tests.test_flaky" name="test_alternates" time="0.001">{failure}</testcase>
    </testsuite>
</testsuites>
""")
            
            summary, flaky_tests = scan_flaky_tests(tests_dir, window=10, flip_threshold=2)
            
            assert summary["total_tests"] == 1
            assert summary["flaky_tests"] == 1
            assert len(flaky_tests) == 1
            
            flaky = flaky_tests[0]
            assert flaky["test"] == "tests.test_flaky::test_alternates"
            assert flaky["flips"] == 3
            assert flaky["history"] == ["pass", "fail", "pass", "fail"]
            assert flaky["runs"] == 4
    
    def test_scan_with_window_limit(self):
        """Test scanning respects window limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            
            # Create 15 XML files, but we'll only analyze last 5
            for i in range(15):
                xml_file = tests_dir / f"test_{i:03d}.xml"
                xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" tests="1">
        <testcase classname="tests.test_a" name="test_1" time="0.001"/>
    </testsuite>
</testsuites>
""")
            
            summary, flaky_tests = scan_flaky_tests(tests_dir, window=5, flip_threshold=2)
            
            assert summary["window"] == 5  # Only last 5 files analyzed
    
    def test_scan_multiple_tests_some_flaky(self):
        """Test scanning multiple tests with some flaky."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            
            # Create 5 runs with 2 tests: one stable, one flaky
            stable_status = "pass"
            flaky_statuses = ["pass", "fail", "pass", "fail", "pass"]
            
            for i, flaky_status in enumerate(flaky_statuses):
                xml_file = tests_dir / f"test_{i:03d}.xml"
                flaky_failure = '<failure message="fail">Error</failure>' if flaky_status == "fail" else ''
                xml_file.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" tests="2">
        <testcase classname="tests.test_a" name="test_stable" time="0.001"/>
        <testcase classname="tests.test_a" name="test_flaky" time="0.001">{flaky_failure}</testcase>
    </testsuite>
</testsuites>
""")
            
            summary, flaky_tests = scan_flaky_tests(tests_dir, window=10, flip_threshold=2)
            
            assert summary["total_tests"] == 2
            assert summary["flaky_tests"] == 1
            assert len(flaky_tests) == 1
            assert flaky_tests[0]["test"] == "tests.test_a::test_flaky"
    
    def test_scan_sorted_by_flip_count(self):
        """Test flaky tests sorted by flip count descending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            
            # Create one XML with multiple tests having different flip counts
            # Test A: 5 flips, Test B: 3 flips
            xml_file = tests_dir / "test_001.xml"
            xml_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" tests="2">
        <testcase classname="tests.test_a" name="test_very_flaky" time="0.001"/>
        <testcase classname="tests.test_b" name="test_somewhat_flaky" time="0.001"/>
    </testsuite>
</testsuites>
""")
            
            # Manually create history to simulate different flip counts
            # This is a bit artificial, but tests the sorting logic
            # In reality, we'd need multiple XML files to build history
            
            # For proper testing, let's create actual history across files
            for i in range(6):
                xml_file = tests_dir / f"test_{i:03d}.xml"
                # Test A alternates every run (5 flips total)
                test_a_status = "pass" if i % 2 == 0 else "fail"
                test_a_failure = '<failure message="fail">Error</failure>' if test_a_status == "fail" else ''
                
                # Test B alternates every 2 runs (2 flips total)
                test_b_status = "pass" if i < 2 or i >= 4 else "fail"
                test_b_failure = '<failure message="fail">Error</failure>' if test_b_status == "fail" else ''
                
                xml_file.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" tests="2">
        <testcase classname="tests.test_a" name="test_very_flaky" time="0.001">{test_a_failure}</testcase>
        <testcase classname="tests.test_b" name="test_somewhat_flaky" time="0.001">{test_b_failure}</testcase>
    </testsuite>
</testsuites>
""")
            
            summary, flaky_tests = scan_flaky_tests(tests_dir, window=10, flip_threshold=2)
            
            assert len(flaky_tests) == 2
            # Should be sorted by flip count descending
            assert flaky_tests[0]["flips"] >= flaky_tests[1]["flips"]
    
    def test_scan_flip_threshold(self):
        """Test flip threshold filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            
            # Create test with exactly 2 flips (should NOT be flagged with threshold=2)
            statuses = ["pass", "fail", "pass"]  # 2 flips
            for i, status in enumerate(statuses):
                xml_file = tests_dir / f"test_{i:03d}.xml"
                failure = '<failure message="fail">Error</failure>' if status == "fail" else ''
                xml_file.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest" tests="1">
        <testcase classname="tests.test_a" name="test_borderline" time="0.001">{failure}</testcase>
    </testsuite>
</testsuites>
""")
            
            # With threshold=2, this should NOT be flagged (need > 2 flips)
            summary, flaky_tests = scan_flaky_tests(tests_dir, window=10, flip_threshold=2)
            
            assert summary["flaky_tests"] == 0
            assert len(flaky_tests) == 0
            
            # With threshold=1, this SHOULD be flagged
            summary2, flaky_tests2 = scan_flaky_tests(tests_dir, window=10, flip_threshold=1)
            
            assert summary2["flaky_tests"] == 1
            assert len(flaky_tests2) == 1

