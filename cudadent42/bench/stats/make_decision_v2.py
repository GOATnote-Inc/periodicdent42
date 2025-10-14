#!/usr/bin/env python3
"""
Statistical decision for V2 promotion
Bootstrap CIs + Hedges' g effect size
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
    Positive g means a is slower (larger values = worse for latency)
    """
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2)/(na+nb-2))
    g = (a.mean()-b.mean())/sp if sp>0 else 0.0
    # Bias correction factor
    J = 1 - (3/(4*(na+nb)-9))
    return float(g*J)


def decide(csv, target_ms=0.255, n_boot=10000, seed=0):
    """
    Make V2 promotion decision
    
    Args:
        csv: Path to benchmark CSV
        target_ms: Target latency threshold (ms)
        n_boot: Bootstrap iterations
        seed: Random seed
    
    Returns:
        dict with decision and statistics
    """
    
    # Load data
    df = pd.read_csv(csv)
    
    # Extract latencies
    sdpa_data = df.loc[df['kernel']=='sdpa', 'latency_ms'].to_numpy()
    v2_data = df.loc[df['kernel']=='v2', 'latency_ms'].to_numpy()
    
    # Compute statistics
    sdpa_ci_lo, sdpa_ci_hi, sdpa_mean = bootstrap_ci(sdpa_data, n_boot, seed=seed)
    v2_ci_lo, v2_ci_hi, v2_mean = bootstrap_ci(v2_data, n_boot, seed=seed)
    
    # Hedges' g (positive = V2 faster than SDPA)
    g = hedges_g(sdpa_data, v2_data)
    
    # Check criteria
    ci_non_overlap = not (v2_ci_lo <= sdpa_ci_hi and sdpa_ci_lo <= v2_ci_hi)
    v2_meets_target = v2_mean <= target_ms
    g_sufficient = g >= 0.8  # V2 should be meaningfully faster
    
    # Speedup
    speedup = sdpa_mean / v2_mean
    
    # Decision
    promote = v2_meets_target and ci_non_overlap and g_sufficient
    
    result = {
        "sdpa": {
            "mean_ms": sdpa_mean,
            "CI95": [sdpa_ci_lo, sdpa_ci_hi],
            "std_ms": float(sdpa_data.std(ddof=1))
        },
        "v2": {
            "mean_ms": v2_mean,
            "CI95": [v2_ci_lo, v2_ci_hi],
            "std_ms": float(v2_data.std(ddof=1))
        },
        "comparison": {
            "speedup": speedup,
            "hedges_g": g,
            "interpretation": "V2 faster" if g > 0 else "V2 slower"
        },
        "criteria": {
            "target_ms": target_ms,
            "v2_meets_target": v2_meets_target,
            "ci_non_overlap": ci_non_overlap,
            "g_sufficient": g_sufficient,
            "g_threshold": 0.8
        },
        "decision": {
            "promote_v2": promote,
            "champion": "v2" if promote else "sdpa",
            "reason": _get_decision_reason(
                promote, v2_meets_target, ci_non_overlap, g_sufficient,
                v2_mean, target_ms, g, speedup
            )
        }
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("V2 PROMOTION DECISION")
    print("=" * 80)
    print(f"\nSDPA (baseline):")
    print(f"  Mean: {sdpa_mean:.4f} ms")
    print(f"  95% CI: [{sdpa_ci_lo:.4f}, {sdpa_ci_hi:.4f}]")
    print(f"\nV2 (candidate):")
    print(f"  Mean: {v2_mean:.4f} ms")
    print(f"  95% CI: [{v2_ci_lo:.4f}, {v2_ci_hi:.4f}]")
    print(f"\nComparison:")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  Hedges' g: {g:.3f} ({result['comparison']['interpretation']})")
    print(f"\nCriteria:")
    print(f"  ✓ V2 ≤ {target_ms} ms: {'✅' if v2_meets_target else '❌'} ({v2_mean:.4f} ms)")
    print(f"  ✓ CIs non-overlapping: {'✅' if ci_non_overlap else '❌'}")
    print(f"  ✓ g ≥ 0.8: {'✅' if g_sufficient else '❌'} ({g:.3f})")
    print(f"\n{'=' * 80}")
    champion_upper = result["decision"]["champion"].upper()
    decision_str = "✅ PROMOTE V2" if promote else f"❌ KEEP {champion_upper}"
    print(f"DECISION: {decision_str}")
    print(f"{'=' * 80}")
    print(f"\n{result['decision']['reason']}\n")
    
    return result


def _get_decision_reason(promote, target, ci_overlap, g_ok, v2_mean, target_ms, g, speedup):
    """Generate human-readable decision reason"""
    if promote:
        return (f"V2 meets all criteria: {v2_mean:.4f} ms ≤ {target_ms} ms, "
                f"CIs non-overlapping, g={g:.3f} ≥ 0.8, speedup={speedup:.2f}×")
    
    reasons = []
    if not target:
        reasons.append(f"V2 mean ({v2_mean:.4f} ms) exceeds target ({target_ms} ms)")
    if not ci_overlap:
        reasons.append("CIs overlap (not statistically significant)")
    if not g_ok:
        reasons.append(f"Effect size too small (g={g:.3f} < 0.8)")
    
    if not reasons:
        reasons.append("Unknown reason - check criteria")
    
    return "V2 not promoted: " + "; ".join(reasons)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Benchmark CSV file")
    parser.add_argument("--target", type=float, default=0.255, help="Target latency (ms)")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="artifacts/stats/v2_decision.json", help="Output JSON")
    
    args = parser.parse_args()
    
    result = decide(args.csv, args.target, args.n_boot, args.seed)
    
    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Decision saved to: {out_path}\n")
    
    # Exit with appropriate code
    sys.exit(0 if result["decision"]["promote_v2"] else 1)

