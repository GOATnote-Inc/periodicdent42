#!/usr/bin/env python3
"""Select tests to maximize EIG under time and cost budgets.

Uses a greedy knapsack algorithm to select tests that maximize
information gain per unit cost, respecting both time and budget constraints.

Usage:
    python scripts/select_tests.py
    python scripts/select_tests.py --budget-sec 300 --budget-usd 0.05
"""

import argparse
import json
import pathlib
import sys
from typing import List, Dict, Any, Tuple


def greedy_knapsack_selection(
    tests: List[Dict[str, Any]],
    budget_sec: float,
    budget_usd: float
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Select tests maximizing EIG per cost under budgets.
    
    Greedy algorithm: Sort by EIG per cost, select while under budget.
    
    Args:
        tests: List of test dicts with eig_bits, cost_usd, duration_sec
        budget_sec: Time budget in seconds
        budget_usd: Cost budget in USD
        
    Returns:
        (selected_tests, stats) tuple
    """
    if not tests:
        return [], {"selected": 0, "total": 0, "eig_total": 0, "cost_total": 0, "time_total": 0}
    
    # Sort by EIG per dollar (efficiency metric)
    sorted_tests = sorted(tests, key=lambda t: t.get("eig_per_dollar", 0), reverse=True)
    
    selected = []
    time_used = 0.0
    cost_used = 0.0
    eig_gained = 0.0
    
    for test in sorted_tests:
        duration = test.get("duration_sec", 0)
        cost = test.get("cost_usd", 0)
        eig = test.get("eig_bits", 0)
        
        # Check if adding this test exceeds budgets
        if (time_used + duration <= budget_sec and
            cost_used + cost <= budget_usd):
            selected.append(test)
            time_used += duration
            cost_used += cost
            eig_gained += eig
    
    stats = {
        "selected": len(selected),
        "total": len(tests),
        "eig_total": eig_gained,
        "cost_total": cost_used,
        "time_total": time_used,
        "budget_sec": budget_sec,
        "budget_usd": budget_usd,
        "utilization_time": time_used / max(budget_sec, 0.001),
        "utilization_cost": cost_used / max(budget_usd, 0.001),
    }
    
    return selected, stats


def fallback_selection(tests: List[Dict[str, Any]], count: int = 10) -> List[Dict[str, Any]]:
    """Fallback: select top N by uncertainty when data sparse.
    
    Args:
        tests: List of test dicts
        count: Number of tests to select
        
    Returns:
        Selected tests
    """
    # Sort by model_uncertainty if available, else by EIG
    def key_func(t):
        if "model_uncertainty" in t and t["model_uncertainty"] is not None:
            # Prefer uncertainty near 0.5 (maximum entropy)
            p = t["model_uncertainty"]
            return abs(p - 0.5)
        return t.get("eig_bits", 0)
    
    sorted_tests = sorted(tests, key=key_func, reverse=True)
    return sorted_tests[:count]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Select tests under budget")
    parser.add_argument(
        "--input",
        default="artifact/eig_rankings.json",
        help="Input EIG rankings JSON"
    )
    parser.add_argument(
        "--output",
        default="artifact/selected_tests.json",
        help="Output selected tests JSON"
    )
    parser.add_argument(
        "--budget-sec",
        type=float,
        help="Time budget in seconds (default: 50%% of full suite)"
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        help="Cost budget in USD (default: 50%% of full suite cost)"
    )
    parser.add_argument(
        "--fallback-count",
        type=int,
        default=10,
        help="Number of tests to select in fallback mode"
    )
    
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    print("=" * 80, flush=True)
    print("🎯 Test Selection (Budget-Constrained EIG Maximization)", flush=True)
    print("=" * 80, flush=True)
    
    # Load EIG rankings
    if not input_path.exists():
        print(f"⚠️  Input file not found: {input_path}", flush=True)
        print(f"   Run scripts/score_eig.py first", flush=True)
        return 1
    
    tests = json.loads(input_path.read_text(encoding="utf-8"))
    
    if not tests:
        print("⚠️  No tests to select from", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"selected": [], "stats": {}}, indent=2))
        return 0
    
    print(f"📊 Loaded {len(tests)} scored tests", flush=True)
    
    # Compute total time and cost
    total_time = sum(t.get("duration_sec", 0) for t in tests)
    total_cost = sum(t.get("cost_usd", 0) for t in tests)
    
    # Set budgets (default: 50% of full suite)
    budget_sec = args.budget_sec if args.budget_sec else total_time * 0.5
    budget_usd = args.budget_usd if args.budget_usd else total_cost * 0.5
    
    print(f"\n💰 Budgets:", flush=True)
    print(f"   Time: {budget_sec:.1f}s / {total_time:.1f}s ({budget_sec/max(total_time,1)*100:.1f}%)", flush=True)
    print(f"   Cost: ${budget_usd:.4f} / ${total_cost:.4f} ({budget_usd/max(total_cost,0.001)*100:.1f}%)", flush=True)
    
    # Check if we have sparse data
    if len(tests) < 20:
        print(f"\n⚠️  Sparse data ({len(tests)} tests), using fallback selection", flush=True)
        selected = fallback_selection(tests, args.fallback_count)
        stats = {
            "selected": len(selected),
            "total": len(tests),
            "mode": "fallback",
            "eig_total": sum(t.get("eig_bits", 0) for t in selected),
            "cost_total": sum(t.get("cost_usd", 0) for t in selected),
            "time_total": sum(t.get("duration_sec", 0) for t in selected),
        }
    else:
        # Run greedy knapsack
        selected, stats = greedy_knapsack_selection(tests, budget_sec, budget_usd)
        stats["mode"] = "knapsack"
    
    print(f"\n✅ Selected {stats['selected']}/{stats['total']} tests", flush=True)
    print(f"   Total EIG: {stats['eig_total']:.2f} bits", flush=True)
    print(f"   Total cost: ${stats['cost_total']:.4f}", flush=True)
    print(f"   Total time: {stats['time_total']:.1f}s", flush=True)
    
    if "utilization_time" in stats:
        print(f"   Time utilization: {stats['utilization_time']*100:.1f}%", flush=True)
        print(f"   Cost utilization: {stats['utilization_cost']*100:.1f}%", flush=True)
    
    # Show selected tests
    print(f"\n📋 Selected Tests:", flush=True)
    for i, test in enumerate(selected[:10], 1):
        name = test.get("name", "unknown")
        eig = test.get("eig_bits", 0)
        cost = test.get("cost_usd", 0)
        print(f"   {i}. {name[:60]}", flush=True)
        print(f"      EIG: {eig:.4f} bits, Cost: ${cost:.4f}", flush=True)
    
    if len(selected) > 10:
        print(f"   ... and {len(selected) - 10} more", flush=True)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "selected": [t.get("name") for t in selected],
        "stats": stats,
        "tests": selected
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    
    print(f"\n✅ Saved selection to {output_path}", flush=True)
    print("=" * 80, flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
