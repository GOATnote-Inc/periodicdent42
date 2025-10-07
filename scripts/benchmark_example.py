#!/usr/bin/env python3
"""Simple benchmark for profiling regression testing.

This script performs a simple computation that can be profiled.
"""

import argparse
import time
import json
from pathlib import Path


def fast_computation(n: int = 10000) -> float:
    """Fast matrix-like computation."""
    total = 0.0
    for i in range(n):
        total += (i ** 2) / (i + 1)
    return total


def slow_computation(n: int = 10000) -> float:
    """Intentionally slow computation (with sleep)."""
    total = 0.0
    for i in range(n):
        total += (i ** 2) / (i + 1)
        if i % 1000 == 0:
            time.sleep(0.001)  # 1ms sleep every 1000 iterations
    return total


def run_benchmark(mode: str = "fast", iterations: int = 5) -> dict:
    """Run benchmark and collect timing data."""
    compute_func = fast_computation if mode == "fast" else slow_computation
    
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = compute_func()
        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)  # Convert to ms
    
    return {
        "mode": mode,
        "iterations": iterations,
        "result": result,
        "timings_ms": timings,
        "mean_ms": sum(timings) / len(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def main():
    parser = argparse.ArgumentParser(description="Run benchmark")
    parser.add_argument("--mode", choices=["fast", "slow"], default="fast",
                       help="Computation mode")
    parser.add_argument("--output", type=Path, help="Save results to JSON")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations")
    
    args = parser.parse_args()
    
    print(f"Running benchmark (mode={args.mode})...")
    results = run_benchmark(args.mode, args.iterations)
    
    print(f"Mean time: {results['mean_ms']:.2f}ms")
    print(f"Min time: {results['min_ms']:.2f}ms")
    print(f"Max time: {results['max_ms']:.2f}ms")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
