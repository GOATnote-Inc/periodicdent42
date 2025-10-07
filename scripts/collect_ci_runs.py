#!/usr/bin/env python3
"""Collect CI run metadata for ML test selection training.

This script collects minimal CI run metadata and appends to data/ci_runs.jsonl.
It can operate in two modes:
- CI mode: Reads environment variables from GitHub Actions
- Mock mode: Generates synthetic entries for local testing

Usage:
    # In CI (automatic via environment variables)
    python scripts/collect_ci_runs.py
    
    # Local testing with mock data
    python scripts/collect_ci_runs.py --mock
    
    # Custom output location
    python scripts/collect_ci_runs.py --out data/custom_runs.jsonl --mock
"""

import argparse
import json
import os
import sys
import time
import pathlib
import random
from typing import Dict, Any


def collect_ci_metadata() -> Dict[str, Any]:
    """Collect metadata from CI environment variables.
    
    Returns:
        Dictionary with CI run metadata
    """
    return {
        "ts": int(time.time()),
        "commit": os.getenv("GITHUB_SHA", ""),
        "branch": os.getenv("GITHUB_REF_NAME", ""),
        "workflow": os.getenv("GITHUB_WORKFLOW", ""),
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "run_number": int(os.getenv("GITHUB_RUN_NUMBER", "0") or 0),
        "duration_sec": int(os.getenv("CI_DURATION_SEC", "0") or 0),
        "status": os.getenv("CI_STATUS", "unknown"),
        "tests_total": int(os.getenv("CI_TESTS_TOTAL", "0") or 0),
        "tests_failed": int(os.getenv("CI_TESTS_FAILED", "0") or 0),
        "tests_skipped": int(os.getenv("CI_TESTS_SKIPPED", "0") or 0),
        "changed_files": int(os.getenv("CI_CHANGED_FILES", "0") or 0),
        "lines_added": int(os.getenv("CI_LINES_ADDED", "0") or 0),
        "lines_deleted": int(os.getenv("CI_LINES_DELETED", "0") or 0),
    }


def generate_mock_entry() -> Dict[str, Any]:
    """Generate synthetic CI run entry for testing.
    
    Returns:
        Dictionary with mock CI run metadata
    """
    # Realistic failure rate: 5-10%
    tests_total = random.randint(50, 500)
    failure_rate = random.uniform(0.05, 0.10)
    tests_failed = int(tests_total * failure_rate) if random.random() < 0.15 else 0
    
    return {
        "ts": int(time.time()),
        "commit": f"mock-{random.randint(100000, 999999):06x}",
        "branch": random.choice([
            "main", "main", "main",  # 3x weight
            "develop",
            "feature/ml-selection",
            "fix/telemetry",
            "test/chaos"
        ]),
        "workflow": "CI & Reproducibility",
        "run_id": str(random.randint(1000000000, 9999999999)),
        "run_number": random.randint(1, 1000),
        "duration_sec": random.randint(60, 1800),
        "status": "failure" if tests_failed > 0 else "success",
        "tests_total": tests_total,
        "tests_failed": tests_failed,
        "tests_skipped": random.randint(0, 10),
        "changed_files": random.randint(1, 25),
        "lines_added": random.randint(0, 500),
        "lines_deleted": random.randint(0, 200),
    }


def append_entry(path: pathlib.Path, entry: Dict[str, Any]) -> None:
    """Append entry to JSONL file (idempotent, append-safe).
    
    Args:
        path: Path to JSONL file
        entry: Metadata dictionary to append
    """
    # Create data directory if missing
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append entry as JSON line
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect CI run metadata for ML test selection"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic entry for local testing"
    )
    parser.add_argument(
        "--out",
        default="data/ci_runs.jsonl",
        help="Output JSONL file path (default: data/ci_runs.jsonl)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print entry details"
    )
    
    args = parser.parse_args()
    
    # Generate or collect entry
    if args.mock:
        entry = generate_mock_entry()
        print(f"Generated mock CI run entry", flush=True)
    else:
        entry = collect_ci_metadata()
        print(f"Collected CI run metadata from environment", flush=True)
    
    # Append to file
    path = pathlib.Path(args.out)
    append_entry(path, entry)
    
    # Print summary
    print(f"✅ Wrote CI run entry to {path}", flush=True)
    
    if args.verbose:
        print(f"\nEntry details:", flush=True)
        for key, value in entry.items():
            print(f"  {key}: {value}", flush=True)
    
    # Count total entries
    try:
        with path.open("r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f if _.strip())
        print(f"📊 Total CI runs collected: {total_lines}", flush=True)
    except Exception:
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
