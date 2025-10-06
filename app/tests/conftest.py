"""Pytest configuration for app package."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
APP_SRC = PROJECT_ROOT / "src"

for path in (str(PROJECT_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

if "src" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "src", APP_SRC / "__init__.py", submodule_search_locations=[str(APP_SRC)]
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.__path__ = [str(APP_SRC)]  # type: ignore[attr-defined]
    sys.modules["src"] = module

# Load test telemetry plugin if enabled
if os.getenv("SKIP_TEST_TELEMETRY") != "1":
    try:
        import pytest
        from src.services.test_telemetry import TestCollector, TestExecution
        
        TELEMETRY_AVAILABLE = True
        
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(item, call):
            """Collect test execution data after each test for ML training."""
            outcome = yield
            report = outcome.get_result()
            
            # Only collect data after the actual test call (not setup/teardown)
            if call.when != "call":
                return
            
            # Skip if telemetry collection fails (don't break tests)
            try:
                collector = TestCollector()
                
                # Extract test info
                test_name = item.nodeid
                test_file = str(Path(item.fspath).relative_to(PROJECT_ROOT))
                duration_ms = report.duration * 1000  # Convert seconds to ms
                passed = report.outcome == "passed"
                error_message = str(report.longrepr) if report.failed else None
                
                # Get git context
                commit_sha = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=2
                ).stdout.strip() or "unknown"
                
                branch = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=2
                ).stdout.strip() or "main"
                
                # Create test execution record
                execution = TestExecution(
                    test_name=test_name,
                    test_file=test_file,
                    duration_ms=duration_ms,
                    passed=passed,
                    commit_sha=commit_sha,
                    branch=branch,
                    error_message=error_message,
                )
                
                # Collect and store
                collector.collect_test_result(execution)
                
                if os.getenv("DEBUG_TELEMETRY") == "1":
                    print(f"✅ Collected telemetry for {test_name}")
                    
            except Exception as e:
                if os.getenv("DEBUG_TELEMETRY") == "1":
                    print(f"⚠️  Telemetry collection failed for {test_name}: {e}")
                    
    except ImportError as e:
        TELEMETRY_AVAILABLE = False
        if os.getenv("DEBUG_TELEMETRY") == "1":
            print(f"⚠️  Test telemetry not available: {e}")
