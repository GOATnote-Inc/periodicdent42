#!/usr/bin/env python3
"""
Compare performance baseline results (baseline vs candidate).

Usage:
    python scripts/compare_results.py <baseline.json> <candidate.json>

Outputs:
    - Console table
    - results/COMPARE.md (markdown table)
"""

import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_results.py <baseline.json> <candidate.json>")
        sys.exit(1)
    
    baseline_path = Path(sys.argv[1])
    candidate_path = Path(sys.argv[2])
    
    if not baseline_path.exists():
        print(f"❌ Baseline file not found: {baseline_path}")
        sys.exit(1)
    
    if not candidate_path.exists():
        print(f"❌ Candidate file not found: {candidate_path}")
        sys.exit(1)
    
    # Load results
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    
    # Build lookup maps
    def key(r):
        return (r["shape"], r["seed"])
    
    baseline_map = {key(r): r for r in baseline["results"]}
    candidate_map = {key(r): r for r in candidate["results"]}
    
    # Build comparison table
    lines = [
        "# FP8 WMMA — Baseline vs Stage-1 (cp.async)\n",
        "| Shape | Seed | p50 (base, μs) | p50 (new, μs) | Δ | Status |",
        "|---|---:|---:|---:|---:|---|"
    ]
    
    print("\n" + "="*80)
    print("FP8 WMMA Performance Comparison")
    print("="*80)
    print(f"Baseline:  {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print("="*80 + "\n")
    
    total_speedup = 0.0
    count = 0
    
    for k in sorted(candidate_map.keys()):
        if k not in baseline_map:
            print(f"⚠️  Shape {k} in candidate but not in baseline, skipping")
            continue
        
        base = baseline_map[k]["p50_us"]
        new = candidate_map[k]["p50_us"]
        delta = (base - new) / base * 100.0
        
        status = "✅ FASTER" if delta > 0 else "⚠️ SLOWER" if delta < -1 else "≈ SAME"
        
        lines.append(
            f"| {k[0]} | {k[1]} | {base:.2f} | {new:.2f} | {delta:+.1f}% | {status} |"
        )
        
        print(f"[{k[0]:8s}] seed={k[1]}: {base:6.2f}μs → {new:6.2f}μs  ({delta:+5.1f}%)  {status}")
        
        total_speedup += delta
        count += 1
    
    # Average speedup
    avg_speedup = total_speedup / count if count > 0 else 0.0
    lines.append("")
    lines.append(f"**Average speedup**: {avg_speedup:+.1f}%")
    
    print("\n" + "="*80)
    print(f"Average speedup: {avg_speedup:+.1f}%")
    
    if avg_speedup >= 10.0:
        print("✅ Stage-1 gate PASSED (≥10% improvement)")
    elif avg_speedup >= 5.0:
        print("⚠️  Modest improvement (5-10%)")
    elif avg_speedup >= 0:
        print("⚠️  Minimal improvement (<5%)")
    else:
        print("❌ Performance regression!")
    
    print("="*80 + "\n")
    
    # Write to file
    output_path = Path("results/COMPARE.md")
    output_path.write_text("\n".join(lines) + "\n")
    print(f"✅ Comparison saved to: {output_path}\n")

if __name__ == "__main__":
    main()

