#!/usr/bin/env python3
"""
Extract baseline performance metrics from Nsight Compute CSV and generate
normalized hash for CI/CD regression detection.

Usage:
    python3 scripts/extract_baseline.py --csv artifacts/nsight_baseline.csv --output ci/baseline/nsight_baseline.norm.txt
    python3 scripts/extract_baseline.py --output ci/baseline/nsight_baseline.norm.txt  # uses artifacts/profile.log
"""

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, Optional


def extract_from_ncu_log(log_path: Path) -> Optional[Dict[str, float]]:
    """Extract metrics from raw ncu text output."""
    metrics = {}
    
    if not log_path.exists():
        return None
    
    content = log_path.read_text()
    
    # Parse ncu output format
    patterns = {
        "sm_active": r"sm__warps_active\.avg\.pct_of_peak_sustained_elapsed\s+([0-9.]+)",
        "tensor_core": r"sm__pipe_tensor_op_hmma_cycles_active\.avg\.pct_of_peak_sustained_elapsed\s+([0-9.]+)",
        "dram_bw": r"dram__throughput\.avg\.pct_of_peak_sustained_elapsed\s+([0-9.]+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
    
    return metrics if metrics else None


def extract_from_csv(csv_path: Path) -> Optional[Dict[str, float]]:
    """Extract metrics from Nsight Compute CSV export."""
    if not csv_path.exists():
        return None
    
    metrics = {}
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row.get("Metric Name", "")
            metric_value = row.get("Metric Value", "")
            
            if "sm__warps_active.avg.pct" in metric_name:
                metrics["sm_active"] = float(metric_value.replace("%", "").strip())
            elif "sm__pipe_tensor_op_hmma" in metric_name:
                metrics["tensor_core"] = float(metric_value.replace("%", "").strip())
            elif "dram__throughput.avg.pct" in metric_name:
                metrics["dram_bw"] = float(metric_value.replace("%", "").strip())
    
    return metrics if metrics else None


def normalize_and_hash(metrics: Dict[str, float]) -> tuple[str, str]:
    """Generate normalized text and SHA256 hash."""
    # Sort keys for determinism
    lines = []
    for key in sorted(metrics.keys()):
        lines.append(f"{key}: {metrics[key]:.2f}")
    
    normalized = "\n".join(lines) + "\n"
    hash_val = hashlib.sha256(normalized.encode()).hexdigest()
    
    return normalized, hash_val


def main():
    parser = argparse.ArgumentParser(description="Extract baseline metrics from Nsight Compute")
    parser.add_argument("--csv", type=Path, help="Path to nsight_baseline.csv")
    parser.add_argument("--log", type=Path, default=Path("artifacts/profile.log"), 
                       help="Path to ncu text output")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for normalized metrics")
    parser.add_argument("--hash-output", type=Path,
                       help="Output path for hash file (default: <output>.sha256)")
    
    args = parser.parse_args()
    
    # Try CSV first, fall back to log
    metrics = None
    if args.csv:
        metrics = extract_from_csv(args.csv)
        if metrics:
            print(f"✅ Extracted metrics from CSV: {args.csv}")
    
    if not metrics:
        metrics = extract_from_ncu_log(args.log)
        if metrics:
            print(f"✅ Extracted metrics from log: {args.log}")
    
    if not metrics:
        print("❌ ERROR: Could not extract metrics from any source", file=sys.stderr)
        print(f"  Tried: {args.csv if args.csv else 'N/A'}, {args.log}", file=sys.stderr)
        print("\nNote: GPU profiling requires permissions. On RunPod/cloud:", file=sys.stderr)
        print("  - Host must have NVreg_RestrictProfilingToAdminUsers=0", file=sys.stderr)
        print("  - Or use alternative timing methods", file=sys.stderr)
        sys.exit(1)
    
    # Normalize and hash
    normalized, hash_val = normalize_and_hash(metrics)
    
    # Write outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(normalized)
    print(f"✅ Wrote normalized metrics: {args.output}")
    
    hash_output = args.hash_output or args.output.with_suffix(".sha256")
    hash_output.write_text(f"{hash_val}  {args.output.name}\n")
    print(f"✅ Wrote hash: {hash_output}")
    
    # Print metrics
    print("\n📊 Baseline Metrics (H100):")
    print("─" * 40)
    for key, value in sorted(metrics.items()):
        key_display = key.replace("_", " ").title()
        print(f"  {key_display:20s}: {value:6.2f}%")
    print("─" * 40)
    print(f"  SHA256: {hash_val[:16]}...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

