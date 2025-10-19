#!/usr/bin/env python3
"""
Benchmark FP8 Stage C WMMA Kernel vs PyTorch SDPA (EvoEngineer Framework)
==========================================================================

This script benchmarks the FP8 Stage C WMMA kernel against PyTorch's
scaled_dot_product_attention using EvoEngineer evidence-based methodology:
  1. Compile & correctness validation (numerical parity gates)
  2. Performance timing (CUDA events, 100 iters, deterministic)
  3. Profiling-ready outputs (CSV/JSON for NCU integration)

Usage:
    python scripts/bench_fp8_stage_c.py [--shapes SHAPES] [--backend BACKEND]

Example:
    python scripts/bench_fp8_stage_c.py --shapes mission,long,wide --backend auto --iters 100
"""

import argparse
import json
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
        description="Benchmark FP8 Stage C WMMA kernel (EvoEngineer Framework)"
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="mission,small,long",
        help="Comma-separated shape presets (mission,small,long,wide,stress)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "math", "flash", "mem_efficient"],
        help="PyTorch SDPA backend to use for baseline (auto = PyTorch selects)",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs",
        help="Output directory for CSV/JSON results",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-iteration timings",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip correctness validation (for perf-only runs)",
    )
    return parser.parse_args()


# Shape presets (EvoEngineer-style: cover key dimensions)
SHAPE_PRESETS = {
    "mission": (1, 8, 512, 64),  # Mission shape from evaluation
    "small": (2, 8, 512, 64),  # Small batch
    "long": (2, 8, 2048, 64),  # Long sequence
    "wide": (2, 8, 512, 128),  # Wide head (HEAD_DIM=128)
    "stress": (4, 8, 2048, 64),  # Stress test
}


def configure_sdpa_backend(backend: str):
    """Configure PyTorch SDPA backend selection"""
    if backend == "auto":
        # Let PyTorch select best backend
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    elif backend == "math":
        # Force math (naive) backend
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    elif backend == "flash":
        # Force FlashAttention backend
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    elif backend == "mem_efficient":
        # Force memory-efficient backend
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def time_kernel_cuda_events(call, iters: int = 100, warmup: int = 20) -> Tuple[float, float, List[float]]:
    """Time a kernel using CUDA events (more accurate than wall-clock)
    
    Returns:
        mean_lat_us: Mean latency in microseconds
        std_lat_us: Standard deviation in microseconds
        all_times_us: List of all individual timings
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(warmup):
        call()
        torch.cuda.synchronize()
    
    # Timed iterations (CUDA events return ms, convert to μs)
    times_us = []
    for _ in range(iters):
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1e3)  # ms → μs
    
    mean_lat = sum(times_us) / len(times_us)
    variance = sum((t - mean_lat) ** 2 for t in times_us) / len(times_us)
    std_lat = variance ** 0.5
    
    return mean_lat, std_lat, times_us


def benchmark_pytorch_sdpa(
    B: int, H: int, S: int, D: int, iters: int = 100, warmup: int = 20
) -> Tuple[float, float, List[float]]:
    """Benchmark PyTorch SDPA (FP16) using CUDA events"""
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    scale = 1.0 / math.sqrt(D)

    def call():
        return F.scaled_dot_product_attention(
            Q, K, V, is_causal=False, scale=scale
        )

    return time_kernel_cuda_events(call, iters, warmup)


def benchmark_fp8_stage_c(
    B: int, H: int, S: int, D: int, iters: int = 100, warmup: int = 20
) -> Tuple[float, float, List[float]]:
    """Benchmark FP8 Stage C WMMA kernel using CUDA events"""
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    def call():
        return sdpa_fp8_stage_c_wmma_forward(Q, K, V)

    return time_kernel_cuda_events(call, iters, warmup)


def validate_correctness(
    B: int, H: int, S: int, D: int, atol: float = 1e-2, rtol: float = 1e-2, seed: int = 42
) -> Tuple[bool, float, float]:
    """Validate FP8 kernel correctness vs PyTorch SDPA
    
    Returns:
        passed: True if within tolerance
        max_abs_diff: Maximum absolute difference
        max_rel_diff: Maximum relative difference
    """
    # Deterministic tensors
    torch.manual_seed(seed)
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    
    # FP8 Stage C output
    try:
        out_fp8 = sdpa_fp8_stage_c_wmma_forward(Q, K, V)
    except Exception as e:
        print(f"      ❌ FP8 kernel failed: {e}")
        return False, float('inf'), float('inf')
    
    # PyTorch SDPA reference
    scale = 1.0 / math.sqrt(D)
    ref = F.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), is_causal=False, scale=scale
    ).to(torch.float16)
    
    # Compute errors
    abs_diff = (out_fp8 - ref).abs()
    max_abs_diff = abs_diff.max().item()
    
    rel_diff = abs_diff / (ref.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    
    passed = (max_abs_diff <= atol) and (max_rel_diff <= rtol)
    
    return passed, max_abs_diff, max_rel_diff


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

    # Set seed for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Configure SDPA backend
    configure_sdpa_backend(args.backend)

    # Print environment
    print("\n🚀 FP8 Stage C WMMA Kernel Benchmark (EvoEngineer Framework)")
    print("=" * 100)
    print(f"  Device:         {torch.cuda.get_device_name()}")
    major, minor = torch.cuda.get_device_capability()
    print(f"  CUDA Arch:      sm_{major}{minor}")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CUDA Version:   {torch.version.cuda}")
    print(f"  SDPA Backend:   {args.backend}")
    print(f"  Iterations:     {args.iters} (warmup: {args.warmup})")
    print(f"  Random Seed:    {args.seed}")
    print(f"  Output Dir:     {args.output_dir}")
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

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = []
    all_pass_correctness = True
    
    for i, (name, (B, H, S, D)) in enumerate(shapes, 1):
        print(f"[{i}/{len(shapes)}] Benchmarking {name} shape (B={B}, H={H}, S={S}, D={D})")

        try:
            # Step 1: Correctness validation (EvoEngineer gate)
            if not args.skip_correctness:
                print("  ✓ Correctness...", end=" ", flush=True)
                passed, max_abs, max_rel = validate_correctness(
                    B, H, S, D, seed=args.seed
                )
                if passed:
                    print(f"✅ PASS (abs={max_abs:.2e}, rel={max_rel:.2e})")
                else:
                    print(f"❌ FAIL (abs={max_abs:.2e}, rel={max_rel:.2e})")
                    print("      Skipping performance benchmark for failed kernel.")
                    all_pass_correctness = False
                    continue

            # Step 2: Performance timing (CUDA events)
            print("  ✓ PyTorch SDPA...", end=" ", flush=True)
            pytorch_mean, pytorch_std, pytorch_times = benchmark_pytorch_sdpa(
                B, H, S, D, iters=args.iters, warmup=args.warmup
            )
            print(f"{pytorch_mean:.2f} ± {pytorch_std:.2f} μs")

            print("  ✓ FP8 Stage C...", end=" ", flush=True)
            fp8_mean, fp8_std, fp8_times = benchmark_fp8_stage_c(
                B, H, S, D, iters=args.iters, warmup=args.warmup
            )
            print(f"{fp8_mean:.2f} ± {fp8_std:.2f} μs")

            speedup = pytorch_mean / fp8_mean
            print(f"  ✓ Speedup: {speedup:.2f}×")
            print()

            # Store results
            result = {
                "name": name,
                "B": B,
                "H": H,
                "S": S,
                "D": D,
                "pytorch_mean": pytorch_mean,
                "pytorch_std": pytorch_std,
                "fp8_mean": fp8_mean,
                "fp8_std": fp8_std,
                "speedup": speedup,
                "pytorch_times": pytorch_times,
                "fp8_times": fp8_times,
            }
            
            if not args.skip_correctness:
                result.update({
                    "correctness_passed": passed,
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_rel,
                })
            
            results.append(result)
            
            # Save individual result to JSON
            result_file = output_path / f"result_{name}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # Print results
    if results:
        print_results_table(results)
        print_summary(results)
        
        # Save aggregated results
        summary_file = output_path / "summary.json"
        summary_data = {
            "environment": {
                "device": torch.cuda.get_device_name(),
                "sm_arch": f"sm_{major}{minor}",
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "sdpa_backend": args.backend,
            },
            "config": {
                "iterations": args.iters,
                "warmup": args.warmup,
                "seed": args.seed,
            },
            "results": results,
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}/")
        print()
        
        # EvoEngineer-style return code: fail if correctness gate not passed
        if not args.skip_correctness and not all_pass_correctness:
            print("⚠️  WARNING: Some shapes failed correctness validation")
            return 1
        
        return 0
    else:
        print("❌ No successful benchmarks. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

