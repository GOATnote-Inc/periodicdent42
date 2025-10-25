#!/usr/bin/env python3
"""
Hard gate: Candidate must beat SDPA with statistical significance

Usage:
    python bench/gate.py --sdpa .ci/sdpa.json --cand .ci/cand.json
    python bench/gate.py --sdpa .ci/sdpa.json --cand .ci/cand.json --alpha 0.05 --speedup 0.95
"""

import argparse
import json
import sys
from pathlib import Path
from bench.sdpa_oracle import bootstrap_ci

def main():
    parser = argparse.ArgumentParser(description="CI gate for kernel performance")
    parser.add_argument("--sdpa", type=Path, required=True,
                       help="SDPA results JSON")
    parser.add_argument("--cand", type=Path, required=True,
                       help="Candidate results JSON")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level (default: 0.05 = 95%% CI)")
    parser.add_argument("--speedup", type=float, default=0.95,
                       help="Speedup threshold (default: 0.95 = 5%% faster)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                       help="Bootstrap samples")
    args = parser.parse_args()
    
    # Load results
    sdpa = json.loads(args.sdpa.read_text())
    cand = json.loads(args.cand.read_text())
    
    print("=" * 70)
    print("CI GATE: Candidate vs SDPA")
    print("=" * 70)
    
    # Performance check
    sdpa_median = sdpa["median_ms"]
    cand_median = cand["median_ms"]
    speedup = sdpa_median / cand_median
    
    perf_threshold = sdpa_median * args.speedup
    perf_passed = cand_median < perf_threshold
    
    print(f"\n📊 Performance:")
    print(f"   SDPA:      {sdpa_median:.4f} ms")
    print(f"   Candidate: {cand_median:.4f} ms")
    print(f"   Speedup:   {speedup:.3f}× {'✅' if speedup > 1 else '❌'}")
    print(f"   Target:    >{1.0/args.speedup:.3f}× (< {args.speedup:.2f}× SDPA)")
    print(f"   Gate:      {'✅ PASS' if perf_passed else '❌ FAIL'}")
    
    # Bootstrap CI
    sdpa_samples = sdpa["samples"]
    cand_samples = cand["samples"]
    
    median_diff, ci_lower, ci_upper = bootstrap_ci(
        cand_samples, sdpa_samples, args.n_bootstrap, args.alpha
    )
    
    ci_passed = ci_upper < 0  # Candidate must be faster with 95% confidence
    
    print(f"\n📈 Bootstrap CI ({100-args.alpha*100:.0f}%):")
    print(f"   Median Δ:  {median_diff:.4f} ms")
    print(f"   CI:        [{ci_lower:.4f}, {ci_upper:.4f}] ms")
    print(f"   CI < 0:    {ci_passed} {'✅' if ci_passed else '❌'}")
    print(f"   Gate:      {'✅ PASS (statistically faster)' if ci_passed else '❌ FAIL (not significant)'}")
    
    # Overall gate
    gate_passed = perf_passed and ci_passed
    
    print(f"\n{'='*70}")
    if gate_passed:
        print("✅ GATE PASSED: Candidate beats SDPA with statistical significance")
        print(f"{'='*70}")
        sys.exit(0)
    else:
        print("❌ GATE FAILED: Candidate does not meet performance requirements")
        print(f"{'='*70}")
        print("\nFailure reasons:")
        if not perf_passed:
            print(f"  - Performance: {cand_median:.4f} ms >= {perf_threshold:.4f} ms (threshold)")
        if not ci_passed:
            print(f"  - CI: [{ci_lower:.4f}, {ci_upper:.4f}] not entirely < 0")
        sys.exit(1)

if __name__ == "__main__":
    main()

