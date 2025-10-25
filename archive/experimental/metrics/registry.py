#!/usr/bin/env python3
"""Metrics Registry - Unified access to all CI/evidence metrics.

Collects current run metrics from Phase-2 artifacts:
- Coverage from coverage.json
- Calibration from experiments/ledger/*.jsonl
- Epistemic from experiments/ledger/*.jsonl
- Build reproducibility from evidence/builds/*.hash
- Dataset/model provenance from ledger

Provides single source of truth for regression detection.
"""

import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional


def collect_current_run(
    base_dir: Optional[pathlib.Path] = None,
    coverage_file: Optional[pathlib.Path] = None,
    ledger_dir: Optional[pathlib.Path] = None,
    builds_dir: Optional[pathlib.Path] = None
) -> Dict[str, Any]:
    """Collect all metrics from current CI run.
    
    Args:
        base_dir: Repository root (default: cwd)
        coverage_file: Path to coverage.json (default: base_dir/coverage.json)
        ledger_dir: Path to ledger dir (default: base_dir/experiments/ledger)
        builds_dir: Path to builds dir (default: base_dir/evidence/builds)
    
    Returns:
        Dict with all collected metrics
    """
    if base_dir is None:
        base_dir = pathlib.Path.cwd()
    
    if coverage_file is None:
        coverage_file = base_dir / "coverage.json"
    
    if ledger_dir is None:
        ledger_dir = base_dir / "experiments" / "ledger"
    
    if builds_dir is None:
        builds_dir = base_dir / "evidence" / "builds"
    
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.getenv("GIT_SHA", os.getenv("GITHUB_SHA", "unknown")),
        "ci_run_id": os.getenv("CI_RUN_ID", os.getenv("GITHUB_RUN_ID", "local")),
    }
    
    # 1. Coverage
    metrics["coverage"] = _get_coverage(coverage_file)
    
    # 2. Calibration + Epistemic + Provenance from ledger
    ledger_metrics = _get_ledger_metrics(ledger_dir)
    metrics.update(ledger_metrics)
    
    # 3. Build reproducibility
    metrics["build_hash_equal"] = _check_build_reproducibility(builds_dir)
    
    return metrics


def _get_coverage(coverage_file: pathlib.Path) -> Optional[float]:
    """Extract coverage percentage from coverage.json."""
    try:
        if coverage_file.exists():
            with coverage_file.open() as f:
                data = json.load(f)
            return data.get("totals", {}).get("percent_covered")
    except Exception:
        pass
    return None


def _get_ledger_metrics(ledger_dir: pathlib.Path) -> Dict[str, Any]:
    """Extract calibration, epistemic, and provenance from latest ledger entry."""
    metrics = {
        "ece": None,
        "brier": None,
        "mce": None,
        "accuracy": None,
        "loss": None,
        "entropy_delta_mean": None,
        "dataset_id": None,
        "model_hash": None,
    }
    
    try:
        if not ledger_dir.exists():
            return metrics
        
        # Find latest ledger file
        ledger_files = sorted(ledger_dir.glob("*.jsonl"))
        if not ledger_files:
            return metrics
        
        # Read last entry
        with ledger_files[-1].open() as f:
            lines = f.readlines()
            if not lines:
                return metrics
            
            entry = json.loads(lines[-1])
        
        # Extract calibration
        calibration = entry.get("calibration", {})
        metrics["ece"] = calibration.get("ece")
        metrics["brier"] = calibration.get("brier_score")
        metrics["mce"] = calibration.get("mce")
        
        # Extract selector metrics (accuracy, loss if present)
        selector_metrics = entry.get("selector_metrics", {})
        metrics["accuracy"] = selector_metrics.get("accuracy")
        metrics["loss"] = selector_metrics.get("loss")
        
        # Extract epistemic
        uncertainty = entry.get("uncertainty", {})
        entropy_before = uncertainty.get("entropy_before", 0.0)
        entropy_after = uncertainty.get("entropy_after", 0.0)
        if entropy_before is not None and entropy_after is not None:
            metrics["entropy_delta_mean"] = abs(entropy_after - entropy_before)
        
        # Extract provenance
        metrics["dataset_id"] = entry.get("dataset_id")
        metrics["model_hash"] = entry.get("model_hash")
    
    except Exception:
        pass
    
    return metrics


def _check_build_reproducibility(builds_dir: pathlib.Path) -> Optional[bool]:
    """Check if double-build hashes are identical."""
    try:
        first_hash_file = builds_dir / "first.hash"
        second_hash_file = builds_dir / "second.hash"
        
        if first_hash_file.exists() and second_hash_file.exists():
            first_hash = first_hash_file.read_text().strip()
            second_hash = second_hash_file.read_text().strip()
            return first_hash == second_hash
    except Exception:
        pass
    return None


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print formatted metrics table."""
    print("=" * 80)
    print("CURRENT RUN METRICS")
    print("=" * 80)
    print()
    print(f"Timestamp:     {metrics.get('timestamp')}")
    print(f"Git SHA:       {metrics.get('git_sha')}")
    print(f"CI Run ID:     {metrics.get('ci_run_id')}")
    print()
    print("Metrics:")
    print(f"  Coverage:           {metrics.get('coverage'):.2%}" if metrics.get('coverage') else "  Coverage:           N/A")
    print(f"  ECE:                {metrics.get('ece'):.4f}" if metrics.get('ece') else "  ECE:                N/A")
    print(f"  Brier:              {metrics.get('brier'):.4f}" if metrics.get('brier') else "  Brier:              N/A")
    print(f"  MCE:                {metrics.get('mce'):.4f}" if metrics.get('mce') else "  MCE:                N/A")
    print(f"  Accuracy:           {metrics.get('accuracy'):.4f}" if metrics.get('accuracy') else "  Accuracy:           N/A")
    print(f"  Loss:               {metrics.get('loss'):.4f}" if metrics.get('loss') else "  Loss:               N/A")
    print(f"  Entropy Delta Mean: {metrics.get('entropy_delta_mean'):.4f}" if metrics.get('entropy_delta_mean') else "  Entropy Delta Mean: N/A")
    print(f"  Build Hash Equal:   {metrics.get('build_hash_equal')}")
    print(f"  Dataset ID:         {metrics.get('dataset_id')}")
    print(f"  Model Hash:         {metrics.get('model_hash')[:16]}..." if metrics.get('model_hash') else "  Model Hash:         N/A")
    print()


if __name__ == "__main__":
    import sys
    
    base_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path.cwd()
    metrics = collect_current_run(base_dir=base_dir)
    print_metrics(metrics)
    
    # Write to JSON for other scripts
    output_file = base_dir / "evidence" / "current_run_metrics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics written to: {output_file}")
    print()
