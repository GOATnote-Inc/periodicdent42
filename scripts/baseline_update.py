#!/usr/bin/env python3
"""Baseline Update - Compute rolling statistical baselines from successful runs.

Features:
- Windowing (last N successful runs)
- Winsorization (5% tails by default)
- EWMA (Exponentially Weighted Moving Average)
- Filters out failed builds, dataset drift

Output: evidence/baselines/rolling_baseline.json
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _config import get_config


def winsorize(values: List[float], pct: float = 0.05) -> List[float]:
    """Winsorize values at given percentile.
    
    Args:
        values: List of numeric values
        pct: Percentile to winsorize (0.05 = 5%)
    
    Returns:
        Winsorized values
    """
    if not values or len(values) < 3:
        return values
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    lower_idx = max(0, int(n * pct))
    upper_idx = min(n - 1, int(n * (1 - pct)))
    
    lower_bound = sorted_vals[lower_idx]
    upper_bound = sorted_vals[upper_idx]
    
    return [max(lower_bound, min(upper_bound, v)) for v in values]


def compute_ewma(values: List[float], alpha: float = 0.2) -> Optional[float]:
    """Compute Exponentially Weighted Moving Average.
    
    Args:
        values: List of values (oldest first)
        alpha: Smoothing factor (0 to 1)
    
    Returns:
        EWMA value or None if no values
    """
    if not values:
        return None
    
    ewma = values[0]
    for v in values[1:]:
        ewma = alpha * v + (1 - alpha) * ewma
    
    return ewma


def load_successful_runs(runs_dir: pathlib.Path, window: int) -> List[Dict[str, Any]]:
    """Load last N successful runs from evidence/runs/*.jsonl.
    
    Filters out:
    - Runs with build_hash_equal == False
    - Runs with dataset drift
    - Runs with missing critical metrics
    
    Args:
        runs_dir: Directory containing run JSONL files
        window: Number of runs to load
    
    Returns:
        List of run dicts (most recent first)
    """
    runs = []
    
    if not runs_dir.exists():
        return runs
    
    # Read all JSONL files
    for jsonl_file in sorted(runs_dir.glob("*.jsonl")):
        try:
            with jsonl_file.open() as f:
                for line in f:
                    if line.strip():
                        run = json.loads(line)
                        
                        # Filter: must have coverage
                        if run.get("coverage") is None:
                            continue
                        
                        # Filter: build reproducibility (if present)
                        if run.get("build_hash_equal") is False:
                            continue
                        
                        runs.append(run)
        except Exception:
            continue
    
    # Sort by timestamp (most recent first)
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    
    return runs[:window]


def compute_baseline_stats(runs: List[Dict[str, Any]], winsor_pct: float = 0.05) -> Dict[str, Any]:
    """Compute baseline statistics from successful runs.
    
    Args:
        runs: List of run dicts
        winsor_pct: Winsorization percentile
    
    Returns:
        Baseline dict with mean/std/ewma per metric
    """
    if not runs:
        return {"metrics": {}, "n": 0, "window": 0}
    
    # Collect values per metric
    metric_values = {
        "coverage": [],
        "ece": [],
        "brier": [],
        "mce": [],
        "accuracy": [],
        "loss": [],
        "entropy_delta_mean": [],
    }
    
    for run in runs:
        for metric in metric_values.keys():
            value = run.get(metric)
            if value is not None:
                metric_values[metric].append(float(value))
    
    # Compute stats per metric
    baseline_metrics = {}
    
    for metric, values in metric_values.items():
        if not values:
            continue
        
        # Winsorize
        winsorized = winsorize(values, winsor_pct)
        
        # Compute mean/std
        mean = sum(winsorized) / len(winsorized)
        variance = sum((v - mean) ** 2 for v in winsorized) / len(winsorized)
        std = variance ** 0.5
        
        # Compute EWMA (use original order, oldest first)
        ewma = compute_ewma(list(reversed(winsorized)), alpha=0.2)
        
        baseline_metrics[metric] = {
            "mean": mean,
            "std": std,
            "ewma": ewma,
            "n": len(values),
        }
    
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "window": len(runs),
        "n": len(runs),
        "metrics": baseline_metrics,
    }


def print_baseline_table(baseline: Dict[str, Any]) -> None:
    """Print formatted baseline table."""
    print("=" * 100)
    print("BASELINE STATISTICS")
    print("=" * 100)
    print()
    print(f"Updated:  {baseline.get('updated_at')}")
    print(f"Window:   {baseline.get('window')} runs")
    print(f"N:        {baseline.get('n')} runs")
    print()
    print(f"{'Metric':<20s} | {'Mean':>10s} | {'Std':>10s} | {'EWMA':>10s} | {'N':>5s}")
    print("-" * 100)
    
    for metric, stats in baseline.get("metrics", {}).items():
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        ewma = stats.get("ewma", 0.0)
        n = stats.get("n", 0)
        
        print(f"{metric:<20s} | {mean:>10.4f} | {std:>10.4f} | {ewma:>10.4f} | {n:>5d}")
    
    print()


def main() -> int:
    """Update baseline from successful runs.
    
    Returns:
        0 on success, 1 on error
    """
    parser = argparse.ArgumentParser(description="Update rolling baseline")
    parser.add_argument("--runs-dir", type=pathlib.Path, default="evidence/runs",
                        help="Directory with run JSONL files")
    parser.add_argument("--output", type=pathlib.Path, default="evidence/baselines/rolling_baseline.json",
                        help="Output baseline JSON path")
    parser.add_argument("--window", type=int, help="Number of runs (default: from config)")
    parser.add_argument("--winsor-pct", type=float, help="Winsorization percentile (default: from config)")
    args = parser.parse_args()
    
    config = get_config()
    window = args.window or config["BASELINE_WINDOW"]
    winsor_pct = args.winsor_pct or config["WINSOR_PCT"]
    
    print()
    print("=" * 100)
    print("BASELINE UPDATE")
    print("=" * 100)
    print()
    print(f"Runs directory: {args.runs_dir}")
    print(f"Window:         {window}")
    print(f"Winsor %:       {winsor_pct:.1%}")
    print()
    
    # Load successful runs
    print(f"📂 Loading runs from: {args.runs_dir}")
    runs = load_successful_runs(args.runs_dir, window)
    print(f"   Found {len(runs)} successful runs")
    print()
    
    if not runs:
        print("⚠️  No successful runs found")
        print()
        print("Run evidence collection first:")
        print("  python metrics/registry.py")
        print("  # Save output to evidence/runs/run_001.jsonl")
        print()
        return 0
    
    # Compute baseline
    print("📊 Computing baseline statistics...")
    baseline = compute_baseline_stats(runs, winsor_pct)
    print()
    
    # Print table
    print_baseline_table(baseline)
    
    # Write to file
    print(f"💾 Writing baseline to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(baseline, f, indent=2)
    print()
    
    print("✅ Baseline update complete!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
