#!/usr/bin/env python3
"""
CI Performance Comparison Tool

Compares candidate performance against baseline with statistical rigor.
Used in CI to enforce performance gates:
- Fail on regression > 3%
- Require non-overlapping CIs for claimed improvements
- Require medium+ effect size (Cliff's delta ≥ 0.3)

Exit Codes:
- 0: Performance maintained or improved (with statistical significance)
- 1: Regression detected (> 3% slower with non-overlapping CIs)
- 2: No significant difference (CIs overlap, or effect size too small)

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-14
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ComparisonResult:
    """Result of baseline vs candidate comparison"""
    baseline_median_ms: float
    baseline_ci_lower: float
    baseline_ci_upper: float
    candidate_median_ms: float
    candidate_ci_lower: float
    candidate_ci_upper: float
    delta_pct: float
    speedup: float
    cis_overlap: bool
    cliffs_delta: float
    cliffs_interpretation: str
    mann_whitney_p: float
    is_significant: bool
    verdict: str


def cliffs_delta(baseline: np.ndarray, candidate: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cliff's Delta effect size
    
    Args:
        baseline: Baseline latency samples
        candidate: Candidate latency samples
    
    Returns:
        (delta, interpretation)
        
    Interpretation:
        |δ| < 0.147: negligible
        0.147 ≤ |δ| < 0.33: small
        0.33 ≤ |δ| < 0.474: medium
        |δ| ≥ 0.474: large
    """
    n_baseline = len(baseline)
    n_candidate = len(candidate)
    
    # Count dominance
    more = sum(np.sum(candidate < b) for b in baseline)
    less = sum(np.sum(candidate > b) for b in baseline)
    
    delta = (more - less) / (n_baseline * n_candidate)
    
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return delta, interpretation


def compare_performance(
    baseline_file: str,
    candidate_file: str,
    regression_threshold_pct: float = -3.0,
    improvement_threshold_pct: float = 10.0
) -> ComparisonResult:
    """
    Compare candidate performance against baseline
    
    Args:
        baseline_file: Path to baseline JSON
        candidate_file: Path to candidate JSON
        regression_threshold_pct: Regression threshold (negative, e.g., -3.0 for 3% slower)
        improvement_threshold_pct: Improvement threshold (positive, e.g., 10.0 for 10% faster)
    
    Returns:
        ComparisonResult with verdict
    """
    # Load baseline
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    # Load candidate
    with open(candidate_file, 'r') as f:
        candidate_data = json.load(f)
    
    # Extract S=512 config (target shape)
    # Baseline is from baseline_comprehensive.py
    baseline_s512 = None
    if 'results' in baseline_data:
        for result in baseline_data['results']:
            if result['config']['seq'] == 512 and \
               result['config']['batch'] == 32 and \
               result['config']['heads'] == 8:
                baseline_s512 = result
                break
    
    if baseline_s512 is None:
        raise ValueError(f"Baseline S=512 config not found in {baseline_file}")
    
    # Candidate is also from baseline_comprehensive.py format
    candidate_s512 = None
    if 'results' in candidate_data:
        for result in candidate_data['results']:
            if result['config']['seq'] == 512 and \
               result['config']['batch'] == 32 and \
               result['config']['heads'] == 8:
                candidate_s512 = result
                break
    
    if candidate_s512 is None:
        raise ValueError(f"Candidate S=512 config not found in {candidate_file}")
    
    # Extract statistics
    baseline_median = baseline_s512['statistics']['median_ms']
    baseline_ci_lower = baseline_s512['statistics']['ci_95_lower']
    baseline_ci_upper = baseline_s512['statistics']['ci_95_upper']
    baseline_latencies = np.array(baseline_s512['raw_latencies'])
    
    candidate_median = candidate_s512['statistics']['median_ms']
    candidate_ci_lower = candidate_s512['statistics']['ci_95_lower']
    candidate_ci_upper = candidate_s512['statistics']['ci_95_upper']
    candidate_latencies = np.array(candidate_s512['raw_latencies'])
    
    # Compute delta and speedup
    delta_pct = ((candidate_median - baseline_median) / baseline_median) * 100
    speedup = baseline_median / candidate_median
    
    # Check CI overlap
    cis_overlap = not (candidate_ci_upper < baseline_ci_lower or candidate_ci_lower > baseline_ci_upper)
    
    # Compute Cliff's delta
    delta, delta_interp = cliffs_delta(baseline_latencies, candidate_latencies)
    
    # Mann-Whitney U test
    _, mann_whitney_p = sp_stats.mannwhitneyu(baseline_latencies, candidate_latencies, alternative='two-sided')
    
    # Determine significance
    # Significant if CIs don't overlap AND p < 0.05 AND effect size is at least small
    is_significant = not cis_overlap and mann_whitney_p < 0.05 and abs(delta) >= 0.147
    
    # Determine verdict
    if delta_pct <= regression_threshold_pct and is_significant:
        verdict = f"❌ REGRESSION: {abs(delta_pct):.1f}% slower (significant, {delta_interp} effect)"
    elif delta_pct >= improvement_threshold_pct and is_significant and abs(delta) >= 0.3:
        verdict = f"✅ IMPROVEMENT: {delta_pct:.1f}% faster (significant, {delta_interp} effect)"
    elif cis_overlap:
        verdict = f"⚠️  NO SIGNIFICANT DIFFERENCE: CIs overlap (Δ={delta_pct:+.1f}%)"
    elif abs(delta) < 0.3:
        verdict = f"⚠️  NO SIGNIFICANT DIFFERENCE: Small effect size (Δ={delta_pct:+.1f}%, δ={abs(delta):.3f})"
    else:
        verdict = f"✅ MAINTAINED: {delta_pct:+.1f}% (within noise)"
    
    return ComparisonResult(
        baseline_median_ms=baseline_median,
        baseline_ci_lower=baseline_ci_lower,
        baseline_ci_upper=baseline_ci_upper,
        candidate_median_ms=candidate_median,
        candidate_ci_lower=candidate_ci_lower,
        candidate_ci_upper=candidate_ci_upper,
        delta_pct=delta_pct,
        speedup=speedup,
        cis_overlap=cis_overlap,
        cliffs_delta=delta,
        cliffs_interpretation=delta_interp,
        mann_whitney_p=mann_whitney_p,
        is_significant=is_significant,
        verdict=verdict
    )


def print_comparison(result: ComparisonResult):
    """Print comparison results"""
    print("="*70)
    print("CI PERFORMANCE COMPARISON")
    print("="*70)
    print()
    
    print("Baseline:")
    print(f"  Median: {result.baseline_median_ms:.4f} ms")
    print(f"  95% CI: [{result.baseline_ci_lower:.4f}, {result.baseline_ci_upper:.4f}] ms")
    print()
    
    print("Candidate:")
    print(f"  Median: {result.candidate_median_ms:.4f} ms")
    print(f"  95% CI: [{result.candidate_ci_lower:.4f}, {result.candidate_ci_upper:.4f}] ms")
    print()
    
    print("Comparison:")
    print(f"  Delta: {result.delta_pct:+.1f}%")
    print(f"  Speedup: {result.speedup:.3f}×")
    print(f"  CIs Overlap: {result.cis_overlap}")
    print()
    
    print("Statistical Tests:")
    print(f"  Cliff's Delta: {result.cliffs_delta:.3f} ({result.cliffs_interpretation})")
    print(f"  Mann-Whitney p: {result.mann_whitney_p:.4f}")
    print(f"  Significant: {result.is_significant}")
    print()
    
    print("="*70)
    print(result.verdict)
    print("="*70)


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare candidate performance against baseline")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON")
    parser.add_argument("--candidate", required=True, help="Path to candidate JSON")
    parser.add_argument("--regression-threshold", type=float, default=-3.0,
                       help="Regression threshold in %% (default: -3.0)")
    parser.add_argument("--improvement-threshold", type=float, default=10.0,
                       help="Improvement threshold in %% (default: 10.0)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.baseline).exists():
        print(f"❌ Baseline file not found: {args.baseline}")
        return 1
    
    if not Path(args.candidate).exists():
        print(f"❌ Candidate file not found: {args.candidate}")
        return 1
    
    # Run comparison
    try:
        result = compare_performance(
            baseline_file=args.baseline,
            candidate_file=args.candidate,
            regression_threshold_pct=args.regression_threshold,
            improvement_threshold_pct=args.improvement_threshold
        )
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return 1
    
    # Output
    if args.json:
        print(json.dumps({
            'baseline_median_ms': result.baseline_median_ms,
            'candidate_median_ms': result.candidate_median_ms,
            'delta_pct': result.delta_pct,
            'speedup': result.speedup,
            'cis_overlap': result.cis_overlap,
            'cliffs_delta': result.cliffs_delta,
            'cliffs_interpretation': result.cliffs_interpretation,
            'mann_whitney_p': result.mann_whitney_p,
            'is_significant': result.is_significant,
            'verdict': result.verdict
        }, indent=2))
    else:
        print_comparison(result)
    
    # Exit code based on verdict
    if "REGRESSION" in result.verdict:
        return 1
    elif "IMPROVEMENT" in result.verdict:
        return 0
    elif "MAINTAINED" in result.verdict:
        return 0
    else:
        return 2  # No significant difference


if __name__ == "__main__":
    sys.exit(main())

