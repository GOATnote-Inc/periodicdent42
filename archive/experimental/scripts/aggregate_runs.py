#!/usr/bin/env python3
"""Aggregate CI run telemetry and produce rollup summary.

Reads evidence/runs/*.jsonl and produces:
- Terminal table with last N runs (coverage, ECE, Brier, build-hash-equal)
- evidence/summary/rollup.json with aggregated stats

Usage:
    python scripts/aggregate_runs.py
    python scripts/aggregate_runs.py --last 20 --output evidence/summary/rollup.json
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict


def load_run_records(runs_dir: pathlib.Path, last_n: int = 20) -> List[Dict[str, Any]]:
    """Load last N run records from JSONL files.
    
    Args:
        runs_dir: Directory containing run JSONL files
        last_n: Number of recent runs to load
    
    Returns:
        List of run dicts, sorted by timestamp (newest first)
    """
    records = []
    
    if not runs_dir.exists():
        return records
    
    # Read all JSONL files
    for jsonl_file in sorted(runs_dir.glob("*.jsonl")):
        try:
            with jsonl_file.open() as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            print(f"⚠️  Error reading {jsonl_file}: {e}", file=sys.stderr)
    
    # Sort by timestamp (newest first)
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    
    return records[:last_n]


def compute_rollup_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics from run records.
    
    Args:
        records: List of run dicts
    
    Returns:
        Dict with rollup statistics
    """
    if not records:
        return {
            "total_runs": 0,
            "avg_coverage": 0.0,
            "avg_ece": 0.0,
            "avg_brier": 0.0,
            "build_reproducibility_rate": 0.0,
            "pass_rate": 0.0,
        }
    
    # Extract metrics
    coverages = []
    eces = []
    briers = []
    builds_identical = []
    passed = []
    
    for record in records:
        # Coverage
        selector_metrics = record.get("selector_metrics", {})
        coverage = selector_metrics.get("coverage_pct", 0.0)
        if coverage > 0:
            coverages.append(coverage)
        
        # Calibration
        calibration = record.get("calibration", {})
        ece = calibration.get("ece", 0.0)
        brier = calibration.get("brier_score", 0.0)
        if ece > 0:
            eces.append(ece)
        if brier > 0:
            briers.append(brier)
        
        # Build reproducibility
        provenance = record.get("provenance", {})
        repro_status = provenance.get("reproducibility_status", "unknown")
        if repro_status == "bit_identical":
            builds_identical.append(True)
        elif repro_status in ["functional_identical", "diverged"]:
            builds_identical.append(False)
        
        # Pass/fail
        gates_passed = record.get("gates_passed", False)
        passed.append(gates_passed)
    
    # Compute averages
    return {
        "total_runs": len(records),
        "avg_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
        "avg_ece": sum(eces) / len(eces) if eces else 0.0,
        "avg_brier": sum(briers) / len(briers) if briers else 0.0,
        "build_reproducibility_rate": sum(builds_identical) / len(builds_identical) if builds_identical else 0.0,
        "pass_rate": sum(passed) / len(passed) if passed else 0.0,
        "time_range": {
            "first": records[-1].get("timestamp") if records else None,
            "last": records[0].get("timestamp") if records else None,
        }
    }


def print_runs_table(records: List[Dict[str, Any]]) -> None:
    """Print formatted table of recent runs.
    
    Args:
        records: List of run dicts (sorted newest first)
    """
    print("=" * 120)
    print("RECENT CI RUNS")
    print("=" * 120)
    print()
    print(f"{'Timestamp':<20s} | {'SHA':<8s} | {'Coverage':>8s} | {'ECE':>6s} | {'Brier':>6s} | {'Build':>8s} | {'Gates':<6s}")
    print("-" * 120)
    
    for record in records:
        timestamp = record.get("timestamp", "unknown")[:19]  # YYYY-MM-DDTHH:MM:SS
        git_sha = record.get("git_sha", "unknown")[:8]
        
        # Coverage
        selector_metrics = record.get("selector_metrics", {})
        coverage = selector_metrics.get("coverage_pct", 0.0)
        coverage_str = f"{coverage:.1f}%" if coverage > 0 else "N/A"
        
        # Calibration
        calibration = record.get("calibration", {})
        ece = calibration.get("ece", 0.0)
        brier = calibration.get("brier_score", 0.0)
        ece_str = f"{ece:.3f}" if ece > 0 else "N/A"
        brier_str = f"{brier:.3f}" if brier > 0 else "N/A"
        
        # Build reproducibility
        provenance = record.get("provenance", {})
        repro_status = provenance.get("reproducibility_status", "unknown")
        if repro_status == "bit_identical":
            build_str = "✅ Equal"
        elif repro_status == "functional_identical":
            build_str = "⚠️  Func"
        elif repro_status == "diverged":
            build_str = "❌ Diff"
        else:
            build_str = "? Unknown"
        
        # Gates
        gates_passed = record.get("gates_passed", False)
        gates_str = "✅ PASS" if gates_passed else "❌ FAIL"
        
        print(f"{timestamp:<20s} | {git_sha:<8s} | {coverage_str:>8s} | {ece_str:>6s} | {brier_str:>6s} | {build_str:>8s} | {gates_str:<6s}")
    
    print()


def main() -> int:
    """Aggregate CI runs and print summary.
    
    Returns:
        0 on success, 1 on error
    """
    parser = argparse.ArgumentParser(description="Aggregate CI run telemetry")
    parser.add_argument("--runs-dir", type=pathlib.Path, default="evidence/runs",
                        help="Directory with run JSONL files (default: evidence/runs)")
    parser.add_argument("--last", type=int, default=20,
                        help="Number of recent runs to show (default: 20)")
    parser.add_argument("--output", type=pathlib.Path, default="evidence/summary/rollup.json",
                        help="Output rollup JSON path (default: evidence/summary/rollup.json)")
    args = parser.parse_args()
    
    print()
    print("=" * 120)
    print("CI RUN AGGREGATION")
    print("=" * 120)
    print()
    
    # Load records
    print(f"📂 Loading runs from: {args.runs_dir}")
    records = load_run_records(args.runs_dir, args.last)
    print(f"   Loaded {len(records)} runs")
    print()
    
    if not records:
        print("⚠️  No run records found")
        print()
        print("Run evidence collection first:")
        print("  python scripts/ingest_ci_logs.py")
        print()
        return 0
    
    # Print table
    print_runs_table(records)
    
    # Compute rollup stats
    print("📊 Computing aggregate statistics...")
    rollup = compute_rollup_stats(records)
    print()
    print("Statistics:")
    print(f"  Total runs:          {rollup['total_runs']}")
    print(f"  Avg coverage:        {rollup['avg_coverage']:.2f}%")
    print(f"  Avg ECE:             {rollup['avg_ece']:.4f}")
    print(f"  Avg Brier:           {rollup['avg_brier']:.4f}")
    print(f"  Build repro rate:    {rollup['build_reproducibility_rate']:.1%}")
    print(f"  Pass rate:           {rollup['pass_rate']:.1%}")
    print()
    
    # Write rollup JSON
    print(f"💾 Writing rollup to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(rollup, f, indent=2)
    print()
    
    print("✅ Aggregation complete!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
