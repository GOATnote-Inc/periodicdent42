#!/usr/bin/env python3
"""
Compute statistics and make promotion decision for V3
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path


def bootstrap_ci(x, n=10000, alpha=0.05, seed=0):
    """Compute bootstrap 95% confidence interval"""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        sample = rng.choice(x, size=len(x), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return float(lo), float(hi), float(np.mean(x))


def hedges_g(a, b):
    """
    Compute Hedges' g effect size
    Positive g means b > a (if b is "better", make it negative)
    """
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2)/(na+nb-2))
    g = (b.mean()-a.mean())/sp if sp>0 else 0.0
    # Bias correction factor
    J = 1 - (3/(4*(na+nb)-9))
    return float(g*J)


def make_decision(csv_path, target_ms=0.255, ci_alpha=0.05, g_threshold=0.8):
    """
    Analyze benchmark results and make promotion decision
    
    Args:
        csv_path: Path to benchmark CSV
        target_ms: Target latency threshold for V3
        ci_alpha: Alpha for confidence intervals (default 0.05 = 95% CI)
        g_threshold: Minimum Hedges' g for promotion
    
    Returns:
        dict with decision and statistics
    """
    
    print(f"\n{'=' * 80}")
    print("Decision Analysis")
    print(f"{'=' * 80}\n")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract latencies
    v3_data = df.loc[df['kernel']=='v3', 'latency_ms'].to_numpy()
    
    # Check if V2 exists
    has_v2 = 'v2' in df['kernel'].values
    if has_v2:
        v2_data = df.loc[df['kernel']=='v2', 'latency_ms'].to_numpy()
    else:
        # Use SDPA as baseline if V2 not available
        v2_data = df.loc[df['kernel']=='sdpa', 'latency_ms'].to_numpy()
        print("Note: V2 not found, using SDPA as baseline\n")
    
    # Compute statistics
    v2_ci_lo, v2_ci_hi, v2_mean = bootstrap_ci(v2_data, alpha=ci_alpha)
    v3_ci_lo, v3_ci_hi, v3_mean = bootstrap_ci(v3_data, alpha=ci_alpha)
    
    # Hedges' g (negative = V3 faster than V2)
    g = hedges_g(v2_data, v3_data)
    g_v3_faster = -g  # Flip sign so positive = V3 better
    
    # Check criteria
    ci_non_overlap = not (v3_ci_lo <= v2_ci_hi and v2_ci_lo <= v3_ci_hi)
    v3_meets_target = v3_mean <= target_ms
    g_sufficient = abs(g_v3_faster) >= g_threshold
    
    # Decision
    promote = v3_meets_target and ci_non_overlap and g_sufficient
    
    result = {
        "baseline": {
            "kernel": "v2" if has_v2 else "sdpa",
            "mean_ms": v2_mean,
            "CI95": [v2_ci_lo, v2_ci_hi]
        },
        "v3": {
            "mean_ms": v3_mean,
            "CI95": [v3_ci_lo, v3_ci_hi]
        },
        "effect_size": {
            "hedges_g": g_v3_faster,
            "interpretation": "V3 faster" if g_v3_faster > 0 else "V3 slower"
        },
        "criteria": {
            "target_ms": target_ms,
            "v3_meets_target": v3_meets_target,
            "ci_non_overlap": ci_non_overlap,
            "g_sufficient": g_sufficient,
            "g_threshold": g_threshold
        },
        "decision": {
            "promote_v3": promote,
            "reason": _get_decision_reason(promote, v3_meets_target, ci_non_overlap, g_sufficient, v3_mean, target_ms, g_v3_faster)
        }
    }
    
    # Print summary
    print(f"Baseline ({result['baseline']['kernel']}):")
    print(f"  Mean: {v2_mean:.4f} ms")
    print(f"  95% CI: [{v2_ci_lo:.4f}, {v2_ci_hi:.4f}]")
    print()
    print(f"V3:")
    print(f"  Mean: {v3_mean:.4f} ms")
    print(f"  95% CI: [{v3_ci_lo:.4f}, {v3_ci_hi:.4f}]")
    print()
    print(f"Effect Size:")
    print(f"  Hedges' g: {g_v3_faster:.3f} ({result['effect_size']['interpretation']})")
    print()
    print(f"Criteria:")
    print(f"  ✓ V3 ≤ {target_ms} ms: {'✅' if v3_meets_target else '❌'} ({v3_mean:.4f} ms)")
    print(f"  ✓ CIs non-overlapping: {'✅' if ci_non_overlap else '❌'}")
    print(f"  ✓ |g| ≥ {g_threshold}: {'✅' if g_sufficient else '❌'} (|{g_v3_faster:.3f}|)")
    print()
    print(f"{'=' * 80}")
    print(f"DECISION: {'✅ PROMOTE V3' if promote else '❌ KEEP V2/BASELINE'}")
    print(f"{'=' * 80}")
    print(f"\nReason: {result['decision']['reason']}")
    
    return result


def _get_decision_reason(promote, target, ci_overlap, g_ok, v3_mean, target_ms, g):
    """Generate human-readable decision reason"""
    if promote:
        return f"V3 meets all criteria: {v3_mean:.4f} ms ≤ {target_ms} ms, CIs non-overlapping, |g|={abs(g):.3f} ≥ 0.8"
    
    reasons = []
    if not target:
        reasons.append(f"V3 mean ({v3_mean:.4f} ms) exceeds target ({target_ms} ms)")
    if not ci_overlap:
        reasons.append("CIs overlap (no statistical significance)")
    if not g_ok:
        reasons.append(f"Effect size too small (|g|={abs(g):.3f} < 0.8)")
    
    return "V3 does not meet criteria: " + "; ".join(reasons)


if __name__ == "__main__":
    # Paths
    csv_path = Path(__file__).parent.parent.parent / "artifacts" / "bench" / "bs2_s512_h8_d64.csv"
    
    if not csv_path.exists():
        print(f"Error: Benchmark results not found at {csv_path}")
        sys.exit(1)
    
    # Analyze
    result = make_decision(csv_path)
    
    # Save decision
    out_dir = Path(__file__).parent.parent.parent / "artifacts" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "decision.json"
    
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDecision saved to: {out_file}\n")
    
    # Exit with appropriate code
    sys.exit(0 if result["decision"]["promote_v3"] else 1)

