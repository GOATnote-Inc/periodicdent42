"""Pytest plugin for automatic test telemetry collection.

This plugin automatically collects test execution data for ML-powered test selection.
It runs after every test and stores results in the test_telemetry database table.

Phase 3 Week 7 Day 6: ML Test Selection Foundation

Usage:
    pytest will automatically load this plugin if it's in the tests/ directory.
    
    To disable: SKIP_TEST_TELEMETRY=1 pytest
    To debug: DEBUG_TELEMETRY=1 pytest
"""

import os
import pytest
import sys
from pathlib import Path

# Add app to path (tests/ is at repo root, app/ is sibling)
REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import test telemetry collector if available
try:
    from src.services.test_telemetry import TestCollector, TestExecution
    TELEMETRY_AVAILABLE = True
except ImportError as e:
    TELEMETRY_AVAILABLE = False
    if os.getenv("DEBUG_TELEMETRY") == "1":
        print(f"⚠️  Test telemetry not available: {e}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect test execution data after each test for ML training.
    
    This hook runs automatically after every test execution and collects:
    - Test name and file
    - Duration and pass/fail status
    - Git commit context
    - Code change features
    - Historical test metrics
    
    Data is stored in test_telemetry table for ML model training.
    """
    outcome = yield
    report = outcome.get_result()
    
    # Only collect for actual test execution (not setup/teardown)
    if report.when == "call" and TELEMETRY_AVAILABLE:
        # Skip if explicitly disabled
        if os.getenv("SKIP_TEST_TELEMETRY") == "1":
            return
        
        try:
            collector = TestCollector()
            
            # Get Git context from environment (set by CI) or local git
            commit_sha = os.getenv("GITHUB_SHA", os.getenv("CI_COMMIT_SHA", "local"))
            branch = os.getenv("GITHUB_REF_NAME", os.getenv("CI_BRANCH", "local"))
            
            # Get changed files if in CI
            changed_files = []
            diff_stats = {"lines_added": 0, "lines_deleted": 0}
            
            if commit_sha != "local":
                try:
                    changed_files = collector.get_changed_files(commit_sha)
                    diff_stats = collector.calculate_diff_stats(commit_sha)
                except Exception:
                    pass
            
            # Calculate test file path
            try:
                test_file = str(Path(item.fspath).relative_to(Path.cwd()))
            except ValueError:
                test_file = str(item.fspath)
            
            # Create execution record
            execution = TestExecution(
                test_name=item.nodeid,
                test_file=test_file,
                duration_ms=report.duration * 1000,  # Convert to ms
                passed=report.passed,
                error_message=str(report.longrepr)[:500] if report.failed else None,  # Truncate
                commit_sha=commit_sha,
                branch=branch,
                changed_files=changed_files,
                lines_added=diff_stats.get("lines_added", 0),
                lines_deleted=diff_stats.get("lines_deleted", 0),
                files_changed=len(changed_files),
                complexity_delta=0.0,  # Will be computed by collector
            )
            
            # Store in database
            collector.collect_test_result(execution)
            
            if os.getenv("DEBUG_TELEMETRY") == "1":
                pass_str = "✅ PASS" if report.passed else "❌ FAIL"
                print(f"📊 Telemetry: {item.nodeid} {pass_str} ({report.duration*1000:.1f}ms)")
            
        except Exception as e:
            # Don't fail tests if telemetry collection fails
            if os.getenv("DEBUG_TELEMETRY") == "1":
                print(f"⚠️  Test telemetry collection failed for {item.nodeid}: {e}")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "ml_test_selection: Mark tests for ML-powered test selection training"
    )
