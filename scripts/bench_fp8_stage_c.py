#!/usr/bin/env python3
"""
Benchmark FP8 Stage C WMMA Kernel vs PyTorch SDPA
==================================================

This script benchmarks the FP8 Stage C WMMA kernel against PyTorch's
scaled_dot_product_attention to validate performance claims and identify
optimization opportunities.

Usage:
    python scripts/bench_fp8_stage_c.py [--shapes SHAPES] [--iters ITERS]

Example:
    python scripts/bench_fp8_stage_c.py --shapes mission,long --iters 200
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Add cudadent42 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cudadent42"))

from bench.sdpa_fp8_stage_c_wmma import sdpa_fp8_stage_c_wmma_forward


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FP8 Stage C WMMA kernel"
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="mission,small,long",
        help="Comma-separated shape presets (mission,small,long,stress)",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-iteration timings",
    )
    return parser.parse_args()


# Shape presets
SHAPE_PRESETS = {
    "mission": (1, 8, 512, 64),  # Mission shape from evaluation
    "small": (2, 8, 512, 64),  # Small batch
    "long": (2, 8, 2048, 64),  # Long sequence
    "stress": (4, 8, 2048, 64),  # Stress test
}


def benchmark_pytorch_sdpa(
    B: int, H: int, S: int, D: int, iters: int = 100, warmup: int = 20
) -> Tuple[float, float]:
    """Benchmark PyTorch SDPA (FP16)"""
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    scale = 1.0 / math.sqrt(D)

    # Warmup
    for _ in range(warmup):
        out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=False, scale=scale
        )
        torch.cuda.synchronize()

    # Benchmark
    timings = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=False, scale=scale
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e6)  # Convert to μs

    mean_lat = sum(timings) / len(timings)
    std_lat = (
        sum((t - mean_lat) ** 2 for t in timings) / len(timings)
    ) ** 0.5

    return mean_lat, std_lat


def benchmark_fp8_stage_c(
    B: int, H: int, S: int, D: int, iters: int = 100, warmup: int = 20
) -> Tuple[float, float]:
    """Benchmark FP8 Stage C WMMA kernel"""
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    # Warmup
    for _ in range(warmup):
        out = sdpa_fp8_stage_c_wmma_forward(Q, K, V)
        torch.cuda.synchronize()

    # Benchmark
    timings = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sdpa_fp8_stage_c_wmma_forward(Q, K, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e6)  # Convert to μs

    mean_lat = sum(timings) / len(timings)
    std_lat = (
        sum((t - mean_lat) ** 2 for t in timings) / len(timings)
    ) ** 0.5

    return mean_lat, std_lat


def print_results_table(results: List[Dict]):
    """Print formatted results table"""
    print("\n" + "=" * 120)
    print("📊 BENCHMARK RESULTS: FP8 Stage C WMMA vs PyTorch SDPA")
    print("=" * 120)
    print()
    print(
        f"{'Shape':<20} {'PyTorch SDPA (μs)':<22} {'FP8 Stage C (μs)':<22} {'Speedup':<12} {'Status':<15}"
    )
    print("-" * 120)

    for r in results:
        shape_str = f"({r['B']},{r['H']},{r['S']},{r['D']})"
        pytorch_str = f"{r['pytorch_mean']:.2f} ± {r['pytorch_std']:.2f}"
        fp8_str = f"{r['fp8_mean']:.2f} ± {r['fp8_std']:.2f}"
        speedup = r["pytorch_mean"] / r["fp8_mean"]
        speedup_str = f"{speedup:.2f}×"

        # Status based on speedup
        if speedup >= 2.0:
            status = "✅ EXCELLENT"
        elif speedup >= 1.5:
            status = "✅ GOOD"
        elif speedup >= 1.1:
            status = "⚠️  MODEST"
        elif speedup >= 0.9:
            status = "⚠️  PARITY"
        else:
            status = "❌ REGRESSION"

        print(
            f"{shape_str:<20} {pytorch_str:<22} {fp8_str:<22} {speedup_str:<12} {status:<15}"
        )

    print("=" * 120)
    print()


def print_summary(results: List[Dict]):
    """Print benchmark summary"""
    avg_speedup = sum(
        r["pytorch_mean"] / r["fp8_mean"] for r in results
    ) / len(results)

    print("📋 SUMMARY")
    print("━" * 80)
    print(f"  Shapes tested:     {len(results)}")
    print(f"  Average speedup:   {avg_speedup:.2f}×")
    print()

    # Best/worst shapes
    best = max(results, key=lambda r: r["pytorch_mean"] / r["fp8_mean"])
    worst = min(results, key=lambda r: r["pytorch_mean"] / r["fp8_mean"])

    best_speedup = best["pytorch_mean"] / best["fp8_mean"]
    worst_speedup = worst["pytorch_mean"] / worst["fp8_mean"]

    print(
        f"  Best shape:        ({best['B']},{best['H']},{best['S']},{best['D']}) → {best_speedup:.2f}×"
    )
    print(
        f"  Worst shape:       ({worst['B']},{worst['H']},{worst['S']},{worst['D']}) → {worst_speedup:.2f}×"
    )
    print()

    # Verdict
    print("🎯 VERDICT")
    print("━" * 80)
    if avg_speedup >= 2.0:
        print(
            "  ✅ EXCELLENT: FP8 Stage C achieves ≥2× speedup vs PyTorch SDPA"
        )
    elif avg_speedup >= 1.5:
        print("  ✅ GOOD: FP8 Stage C achieves 1.5-2× speedup vs PyTorch SDPA")
    elif avg_speedup >= 1.1:
        print(
            "  ⚠️  MODEST: FP8 Stage C achieves modest speedup (1.1-1.5×)"
        )
        print(
            "      Recommendation: Profile with NCU to identify bottlenecks"
        )
    elif avg_speedup >= 0.9:
        print("  ⚠️  PARITY: FP8 Stage C matches PyTorch SDPA (~1×)")
        print(
            "      Recommendation: Investigate FP8 quantization overhead"
        )
    else:
        print("  ❌ REGRESSION: FP8 Stage C slower than PyTorch SDPA")
        print("      Action required: Debug kernel implementation")

    print()


def main():
    args = parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot run benchmarks.")
        return 1

    print("\n🚀 FP8 Stage C WMMA Kernel Benchmark")
    print("=" * 80)
    print(f"  Device:       {torch.cuda.get_device_name()}")
    print(
        f"  CUDA Arch:    sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    )
    print(f"  Iterations:   {args.iters} (warmup: {args.warmup})")
    print()

    # Parse shapes
    shape_names = [s.strip() for s in args.shapes.split(",")]
    shapes = []
    for name in shape_names:
        if name in SHAPE_PRESETS:
            shapes.append((name, SHAPE_PRESETS[name]))
        else:
            print(f"⚠️  Unknown shape preset: {name} (skipping)")

    if not shapes:
        print(
            "❌ No valid shapes specified. Use: mission,small,long,stress"
        )
        return 1

    print(f"📐 Testing {len(shapes)} shape(s):")
    for name, (B, H, S, D) in shapes:
        print(f"  - {name:10s}: (B={B}, H={H}, S={S}, D={D})")
    print()

    # Run benchmarks
    results = []
    for i, (name, (B, H, S, D)) in enumerate(shapes, 1):
        print(f"[{i}/{len(shapes)}] Benchmarking {name} shape...")

        try:
            # PyTorch SDPA
            print("  → PyTorch SDPA...", end=" ", flush=True)
            pytorch_mean, pytorch_std = benchmark_pytorch_sdpa(
                B, H, S, D, iters=args.iters, warmup=args.warmup
            )
            print(f"{pytorch_mean:.2f} ± {pytorch_std:.2f} μs")

            # FP8 Stage C
            print("  → FP8 Stage C...", end=" ", flush=True)
            fp8_mean, fp8_std = benchmark_fp8_stage_c(
                B, H, S, D, iters=args.iters, warmup=args.warmup
            )
            print(f"{fp8_mean:.2f} ± {fp8_std:.2f} μs")

            speedup = pytorch_mean / fp8_mean
            print(f"  → Speedup: {speedup:.2f}×")
            print()

            results.append(
                {
                    "name": name,
                    "B": B,
                    "H": H,
                    "S": S,
                    "D": D,
                    "pytorch_mean": pytorch_mean,
                    "pytorch_std": pytorch_std,
                    "fp8_mean": fp8_mean,
                    "fp8_std": fp8_std,
                }
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            print()
            continue

    # Print results
    if results:
        print_results_table(results)
        print_summary(results)
    else:
        print("❌ No successful benchmarks. Check errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

