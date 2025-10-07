#!/usr/bin/env python3
"""Score Expected Information Gain (EIG) for each test.

Computes EIG using information-theoretic methods:
- Bernoulli entropy H(p) for predicted failure probability
- ΔH = H_before - H_after when both available
- Wilson-smoothed empirical failure rate as fallback

Usage:
    python scripts/score_eig.py
    python scripts/score_eig.py --input data/ci_runs.jsonl --output artifact/eig_rankings.json
"""

import argparse
import json
import math
import pathlib
import sys
from typing import List, Dict, Any, Optional


def bernoulli_entropy(p: float) -> float:
    """Compute Bernoulli entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
    
    Args:
        p: Probability in [0, 1]
        
    Returns:
        Entropy in bits (0.0 to 1.0)
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def compute_eig(test: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    """Compute Expected Information Gain for a test.
    
    Strategy (in order of preference):
    1. If entropy_before and entropy_after exist: ΔH = H_before - H_after
    2. If model_uncertainty (predicted failure prob): H(p)
    3. Else: empirical failure rate with Wilson smoothing
    
    Args:
        test: Test metadata dict
        history: Historical test results for this test
        
    Returns:
        EIG in bits
    """
    # Method 1: Direct ΔH if available
    if "entropy_before" in test and "entropy_after" in test:
        delta_h = test["entropy_before"] - test["entropy_after"]
        return max(0.0, delta_h)
    
    # Method 2: Model uncertainty (preferred)
    if "model_uncertainty" in test and test["model_uncertainty"] is not None:
        p = test["model_uncertainty"]
        return bernoulli_entropy(p)
    
    # Method 3: Empirical failure rate with Wilson smoothing
    # Wilson score interval provides better estimates for small samples
    if history:
        failures = sum(1 for h in history if h.get("result") == "fail")
        n = len(history)
        
        # Wilson score with z=1.96 (95% confidence)
        z = 1.96
        p_hat = failures / n
        denominator = 1 + z**2 / n
        p_wilson = (p_hat + z**2 / (2*n)) / denominator
        
        return bernoulli_entropy(p_wilson)
    
    # Fallback: assume p=0.5 (maximum entropy, maximum info gain potential)
    return 1.0


def load_test_history(data_path: pathlib.Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load historical test results grouped by test name.
    
    Args:
        data_path: Path to ci_runs.jsonl
        
    Returns:
        Dict mapping test_name to list of historical results
    """
    if not data_path.exists():
        return {}
    
    history = {}
    
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                
                # Handle both Test and CIRun formats
                if "tests" in entry:
                    # CIRun format
                    for test in entry.get("tests", []):
                        name = test.get("name")
                        if name:
                            if name not in history:
                                history[name] = []
                            history[name].append(test)
                elif "name" in entry:
                    # Single Test format
                    name = entry.get("name")
                    if name:
                        if name not in history:
                            history[name] = []
                        history[name].append(entry)
            except json.JSONDecodeError:
                continue
    
    return history


def score_all_tests(
    data_path: pathlib.Path,
    runner_usd_per_hour: float = 0.60
) -> List[Dict[str, Any]]:
    """Score all tests from recent CI run.
    
    Args:
        data_path: Path to ci_runs.jsonl
        runner_usd_per_hour: Cost per hour of CI runner
        
    Returns:
        List of test dicts with EIG scores, sorted by EIG desc
    """
    # Load history
    history = load_test_history(data_path)
    
    # Load most recent run
    if not data_path.exists():
        return []
    
    lines = data_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    
    # Parse last line (most recent run)
    try:
        last_run = json.loads(lines[-1])
    except json.JSONDecodeError:
        return []
    
    # Extract tests
    if "tests" in last_run:
        tests = last_run["tests"]
    elif "name" in last_run:
        # Single test format - load all from file
        tests = []
        for line in lines:
            if line.strip():
                try:
                    tests.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        return []
    
    # Score each test
    scored_tests = []
    for test in tests:
        name = test.get("name", "unknown")
        test_history = history.get(name, [])
        
        # Compute EIG
        eig = compute_eig(test, test_history)
        
        # Compute cost if not present
        if "cost_usd" not in test or test["cost_usd"] is None:
            duration = test.get("duration_sec", 0)
            test["cost_usd"] = duration * runner_usd_per_hour / 3600.0
        
        # Add EIG to test dict
        scored_test = dict(test)
        scored_test["eig_bits"] = eig
        scored_test["eig_per_sec"] = eig / max(test.get("duration_sec", 1), 0.01)
        scored_test["eig_per_dollar"] = eig / max(test.get("cost_usd", 0.001), 0.001)
        
        scored_tests.append(scored_test)
    
    # Sort by EIG descending
    scored_tests.sort(key=lambda t: t["eig_bits"], reverse=True)
    
    return scored_tests


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Score EIG for tests")
    parser.add_argument(
        "--input",
        default="data/ci_runs.jsonl",
        help="Input CI runs JSONL file"
    )
    parser.add_argument(
        "--output",
        default="artifact/eig_rankings.json",
        help="Output EIG rankings JSON"
    )
    parser.add_argument(
        "--runner-cost",
        type=float,
        default=0.60,
        help="CI runner cost per hour (USD)"
    )
    
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    print("=" * 80, flush=True)
    print("🧠 Expected Information Gain (EIG) Scoring", flush=True)
    print("=" * 80, flush=True)
    
    # Score tests
    scored_tests = score_all_tests(input_path, args.runner_cost)
    
    if not scored_tests:
        print("⚠️  No tests found to score", flush=True)
        # Write empty output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([], indent=2))
        return 0
    
    print(f"📊 Scored {len(scored_tests)} tests", flush=True)
    
    # Compute summary statistics
    total_eig = sum(t["eig_bits"] for t in scored_tests)
    total_cost = sum(t["cost_usd"] for t in scored_tests)
    total_time = sum(t.get("duration_sec", 0) for t in scored_tests)
    
    print(f"\n📈 Summary:", flush=True)
    print(f"   Total EIG: {total_eig:.2f} bits", flush=True)
    print(f"   Total cost: ${total_cost:.4f}", flush=True)
    print(f"   Total time: {total_time:.1f}s", flush=True)
    print(f"   Bits per dollar: {total_eig/max(total_cost, 0.001):.2f}", flush=True)
    print(f"   Bits per second: {total_eig/max(total_time, 0.001):.4f}", flush=True)
    
    # Show top 10
    print(f"\n🏆 Top 10 Tests by EIG:", flush=True)
    for i, test in enumerate(scored_tests[:10], 1):
        name = test.get("name", "unknown")
        eig = test["eig_bits"]
        cost = test["cost_usd"]
        domain = test.get("domain", "generic")
        print(f"   {i}. {name[:60]}", flush=True)
        print(f"      EIG: {eig:.4f} bits, Cost: ${cost:.4f}, Domain: {domain}", flush=True)
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create compact output (just essential fields)
    output_tests = []
    for test in scored_tests:
        output_tests.append({
            "name": test.get("name"),
            "suite": test.get("suite"),
            "domain": test.get("domain"),
            "eig_bits": test["eig_bits"],
            "eig_per_dollar": test["eig_per_dollar"],
            "eig_per_sec": test["eig_per_sec"],
            "cost_usd": test["cost_usd"],
            "duration_sec": test.get("duration_sec"),
            "model_uncertainty": test.get("model_uncertainty"),
        })
    
    output_path.write_text(json.dumps(output_tests, indent=2))
    print(f"\n✅ Saved EIG rankings to {output_path}", flush=True)
    
    print("=" * 80, flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
