"""Tests for scripts/collect_ci_runs.py - CI data collection."""

import json
import tempfile
from pathlib import Path
import sys

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from collect_ci_runs import (
    generate_mock_run,
    inject_failures,
    write_run,
    load_runs,
)


def test_generate_mock_run_structure():
    """Test that mock run has correct structure."""
    run = generate_mock_run(seed=42, run_id=1)
    
    assert "run_id" in run
    assert "timestamp" in run
    assert "tests" in run
    assert "duration_sec" in run
    assert "outcome" in run
    
    assert isinstance(run["tests"], list)
    assert len(run["tests"]) > 0
    assert run["run_id"] == 1


def test_generate_mock_run_reproducibility():
    """Test that same seed produces same run."""
    run1 = generate_mock_run(seed=42, run_id=1)
    run2 = generate_mock_run(seed=42, run_id=1)
    
    assert run1 == run2


def test_inject_failures_rate():
    """Test that failure injection respects rate."""
    runs = [generate_mock_run(seed=i, run_id=i) for i in range(100)]
    
    # Inject 10% failures
    runs_with_failures = inject_failures(runs, rate=0.1, seed=42)
    
    failed = sum(1 for r in runs_with_failures if r["outcome"] == "failed")
    total = len(runs_with_failures)
    
    # Should be approximately 10% (±5%)
    assert 5 <= failed <= 15, f"Expected ~10 failures, got {failed}"


def test_inject_failures_zero_rate():
    """Test that zero rate injects no failures."""
    runs = [generate_mock_run(seed=i, run_id=i) for i in range(10)]
    runs_with_failures = inject_failures(runs, rate=0.0, seed=42)
    
    failed = sum(1 for r in runs_with_failures if r["outcome"] == "failed")
    assert failed == 0


def test_write_and_load_runs():
    """Test that runs can be written and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_runs.jsonl"
        
        # Generate and write runs
        runs = [generate_mock_run(seed=i, run_id=i) for i in range(5)]
        for run in runs:
            write_run(run, output_file)
        
        # Load runs back
        loaded_runs = load_runs(output_file)
        
        assert len(loaded_runs) == 5
        assert loaded_runs == runs


def test_load_runs_empty_file():
    """Test loading from empty file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "empty.jsonl"
        output_file.touch()
        
        loaded_runs = load_runs(output_file)
        assert loaded_runs == []


def test_test_structure():
    """Test that individual test entries have correct structure."""
    run = generate_mock_run(seed=42, run_id=1)
    
    for test in run["tests"]:
        assert "name" in test
        assert "duration_ms" in test
        assert "outcome" in test
        assert test["outcome"] in ["passed", "failed", "skipped"]


def test_mock_generation_count():
    """Test generating specific number of mock runs."""
    num_runs = 50
    runs = [generate_mock_run(seed=i, run_id=i) for i in range(num_runs)]
    
    assert len(runs) == num_runs
    assert len(set(r["run_id"] for r in runs)) == num_runs  # Unique IDs


def test_duration_positive():
    """Test that durations are positive."""
    run = generate_mock_run(seed=42, run_id=1)
    
    assert run["duration_sec"] > 0
    for test in run["tests"]:
        assert test["duration_ms"] > 0


def test_timestamp_format():
    """Test that timestamp is ISO format."""
    run = generate_mock_run(seed=42, run_id=1)
    
    # Should be ISO 8601 format
    assert "T" in run["timestamp"]
    assert "Z" in run["timestamp"]
    
    # Should be parseable
    from datetime import datetime
    datetime.fromisoformat(run["timestamp"].replace("Z", "+00:00"))

