#!/usr/bin/env python3
"""
Baseline Comparison & Regression Detection
===========================================
Compare benchmark results against baseline to detect regressions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional


def load_result(path: Path) -> Dict:
    """Load benchmark result from JSON"""
    with open(path, 'r') as f:
        return json.load(f)


def compare_results(
    baseline_path: Path,
    current_path: Path,
    regression_threshold: float = -3.0,  # -3% is regression
    output_path: Optional[Path] = None
) -> bool:
    """
    Compare current results against baseline
    
    Args:
        baseline_path: Path to baseline JSON
        current_path: Path to current JSON
        regression_threshold: Percentage threshold for regression (negative = slower)
        
    Returns:
        True if no regression, False if regression detected
    """
    
    baseline = load_result(baseline_path)
    current = load_result(current_path)
    
    # Extract key metrics
    baseline_time = baseline['metrics']['mean_time_ms']
    current_time = current['metrics']['mean_time_ms']
    
    # Calculate speedup and improvement
    speedup = baseline_time / current_time
    improvement_pct = ((baseline_time - current_time) / baseline_time) * 100
    
    # Check for regression
    is_regression = improvement_pct < regression_threshold
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"BASELINE COMPARISON")
    print(f"{'='*70}")
    
    print(f"\nKernel: {baseline['kernel_name']}")
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'Current':<15} {'Change':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    
    # Timing
    print(f"{'Mean Time (ms)':<25} {baseline_time:>13.4f} {current_time:>13.4f} "
          f"{improvement_pct:>+13.2f}%")
    
    b_std = baseline['metrics']['std_dev_ms']
    c_std = current['metrics']['std_dev_ms']
    print(f"{'Std Dev (ms)':<25} {b_std:>13.4f} {c_std:>13.4f}")
    
    # Throughput
    if 'throughput_gflops' in baseline['metrics'] and baseline['metrics']['throughput_gflops']:
        b_gflops = baseline['metrics']['throughput_gflops']
        c_gflops = current['metrics']['throughput_gflops']
        gflops_change = ((c_gflops - b_gflops) / b_gflops) * 100
        print(f"{'Throughput (GFLOPS)':<25} {b_gflops:>13.2f} {c_gflops:>13.2f} "
              f"{gflops_change:>+13.2f}%")
    
    # Bandwidth
    if 'bandwidth_gb_s' in baseline['metrics'] and baseline['metrics']['bandwidth_gb_s']:
        b_bw = baseline['metrics']['bandwidth_gb_s']
        c_bw = current['metrics']['bandwidth_gb_s']
        bw_change = ((c_bw - b_bw) / b_bw) * 100
        print(f"{'Bandwidth (GB/s)':<25} {b_bw:>13.2f} {c_bw:>13.2f} "
              f"{bw_change:>+13.2f}%")
    
    print(f"\n{'='*70}")
    print(f"Speedup:     {speedup:>6.4f}x")
    print(f"Improvement: {improvement_pct:>+6.2f}%")
    print(f"{'='*70}")
    
    # Export JSON for CI if requested
    if output_path:
        comparison_data = {
            'speedup': float(speedup),
            'improvement_pct': float(improvement_pct),
            'is_regression': is_regression,
            'baseline_time_ms': float(baseline_time),
            'current_time_ms': float(current_time),
            'threshold': float(regression_threshold)
        }
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
    
    # Verdict
    if is_regression:
        print(f"\nREGRESSION DETECTED")
        print(f"Performance decreased by {abs(improvement_pct):.2f}%")
        print(f"Threshold: {regression_threshold}%")
        print(f"\nAction required:")
        print(f"  1. Investigate recent changes")
        print(f"  2. Profile to identify bottleneck")
        print(f"  3. Revert if intentional change")
        return False
    elif improvement_pct > 10:
        print(f"\nSIGNIFICANT IMPROVEMENT")
        print(f"Performance increased by {improvement_pct:.2f}%")
        print(f"\nRecommendation:")
        print(f"  1. Verify correctness maintained")
        print(f"  2. Document optimization technique")
        print(f"  3. Update baseline with current results")
        return True
    elif improvement_pct > 0:
        print(f"\nSLIGHT IMPROVEMENT")
        print(f"Performance increased by {improvement_pct:.2f}%")
        print(f"Within normal variance range")
        return True
    else:
        print(f"\nPERFORMANCE SIMILAR")
        print(f"Change within {abs(regression_threshold)}% threshold")
        return True


def set_baseline(
    result_path: Path,
    baseline_path: Optional[Path] = None
) -> None:
    """
    Set current result as new baseline
    
    Args:
        result_path: Path to result to use as baseline
        baseline_path: Path to save baseline (default: same dir as result)
    """
    if baseline_path is None:
        baseline_path = result_path.parent / "baseline.json"
    
    result = load_result(result_path)
    
    with open(baseline_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nBaseline updated: {baseline_path}")
    print(f"Kernel: {result['kernel_name']}")
    print(f"Mean time: {result['metrics']['mean_time_ms']:.4f} ms")


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare benchmark results against baseline"
    )
    parser.add_argument(
        'current',
        type=Path,
        help='Path to current benchmark result JSON'
    )
    parser.add_argument(
        '--baseline',
        type=Path,
        help='Path to baseline JSON (default: baseline.json in same dir)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=-3.0,
        help='Regression threshold in percent (default: -3.0)'
    )
    parser.add_argument(
        '--set-baseline',
        action='store_true',
        help='Set current result as new baseline'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file for comparison results'
    )
    
    args = parser.parse_args()
    
    if args.set_baseline:
        set_baseline(args.current, args.baseline)
        return 0
    
    # Find baseline
    if args.baseline:
        baseline_path = args.baseline
    else:
        baseline_path = args.current.parent / "baseline.json"
    
    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}")
        print(f"\nCreate baseline with: {sys.argv[0]} {args.current} --set-baseline")
        return 1
    
    # Compare
    passed = compare_results(baseline_path, args.current, args.threshold, args.output)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

