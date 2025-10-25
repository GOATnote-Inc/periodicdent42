#!/usr/bin/env python3
"""
Stage-5 Robust Benchmarking Script
===================================
- 100-run medians (p50/p90/p99), 20 warmup iterations
- Modular evaluation: compile → correctness → performance
- PyTorch SDPA baseline comparison
- JSON output for reproducibility

Aligned with EvoEngineer evaluation methodology (Sec. 4.3, 5.1)
"""

import os
import json
import time
import math
import statistics
import argparse
import sys
from pathlib import Path

import torch
torch.set_float32_matmul_precision("high")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.fp8_sdpa_stage_c_wmma.func_forward import (
    forward_ref,
    forward_kernel,
    quantize_sim_fp8_per_head,
)
from tasks.fp8_sdpa_stage_c_wmma.build import build_extension


def run_one_timed(ext, Q_q, K_q, V_q, Q_s, K_s, V_s, scale):
    """Run kernel once with CUDA event timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    O = ext.forward(Q_q, K_q, V_q, Q_s, K_s, V_s, scale)
    end.record()
    
    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end)
    return time_ms * 1000.0, O  # Convert to μs


def bench_shape(shape_name, B, H, S, D, iters, warmup, ext, tol=0.06, seed=0):
    """
    Benchmark one shape configuration.
    
    Returns:
        dict: Results including p50/p90/p99, PyTorch comparison, correctness
    """
    device = "cuda"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Generate test data (FP16)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    K = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    V = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    scale = 1.0 / math.sqrt(D)
    
    # Quantize to FP8 (sim)
    Q_q, Q_s = quantize_sim_fp8_per_head(Q)
    K_q, K_s = quantize_sim_fp8_per_head(K)
    V_q, V_s = quantize_sim_fp8_per_head(V)
    
    # ====================
    # 1. PyTorch Baseline
    # ====================
    print(f"  [{shape_name:8s}] PyTorch baseline...", end=" ", flush=True)
    torch_times = []
    for _ in range(min(warmup, 10)):
        _ = forward_ref(Q, K, V, scale)
    
    for _ in range(min(iters, 50)):  # Fewer iters for PyTorch (slower)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ref_out = forward_ref(Q, K, V, scale)
        end.record()
        torch.cuda.synchronize()
        torch_times.append(start.elapsed_time(end) * 1000.0)  # μs
    
    torch_p50 = statistics.median(torch_times)
    print(f"{torch_p50:.2f} μs")
    
    # ====================
    # 2. Our Kernel Warmup
    # ====================
    print(f"  [{shape_name:8s}] Warmup...", end=" ", flush=True)
    for _ in range(warmup):
        _ = ext.forward(Q_q, K_q, V_q, Q_s, K_s, V_s, scale)
    torch.cuda.synchronize()
    print("done")
    
    # ====================
    # 3. Correctness Check
    # ====================
    print(f"  [{shape_name:8s}] Correctness...", end=" ", flush=True)
    _, our_out = run_one_timed(ext, Q_q, K_q, V_q, Q_s, K_s, V_s, scale)
    
    diff = (our_out.float() - ref_out.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    bad_pct = (diff > tol).float().mean().item() * 100.0
    
    correctness_pass = max_err <= tol and mean_err <= 0.02 and bad_pct <= 1.0
    print(f"max_err={max_err:.4f}, mean_err={mean_err:.4f}, %bad={bad_pct:.1f}% → {'PASS' if correctness_pass else 'FAIL'}")
    
    # ====================
    # 4. Performance Timing
    # ====================
    print(f"  [{shape_name:8s}] Timing {iters} iters...", end=" ", flush=True)
    times_us = []
    for _ in range(iters):
        us, _ = run_one_timed(ext, Q_q, K_q, V_q, Q_s, K_s, V_s, scale)
        times_us.append(us)
    
    p50 = statistics.median(times_us)
    p90 = statistics.quantiles(times_us, n=10)[8]
    p99 = statistics.quantiles(times_us, n=100)[98]
    mean_us = statistics.mean(times_us)
    std_us = statistics.stdev(times_us) if len(times_us) > 1 else 0.0
    
    speedup = torch_p50 / max(p50, 1e-6)
    print(f"p50={p50:.2f} μs ({speedup:.1f}× vs PyTorch)")
    
    return {
        "shape": shape_name,
        "B": B,
        "H": H,
        "S": S,
        "D": D,
        "seed": seed,
        "p50_us": p50,
        "p90_us": p90,
        "p99_us": p99,
        "mean_us": mean_us,
        "std_us": std_us,
        "torch_p50_us": torch_p50,
        "speedup_vs_torch": speedup,
        "max_err": max_err,
        "mean_err": mean_err,
        "bad_pct": bad_pct,
        "tol": tol,
        "correctness_pass": correctness_pass,
    }


def main():
    ap = argparse.ArgumentParser(description="Stage-5 Robust SDPA Benchmark")
    ap.add_argument("--iters", type=int, default=100, help="Performance timing iterations")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    ap.add_argument("--shapes", type=str, default="small,mission,long", help="Comma-separated shape names")
    ap.add_argument("--out", type=str, default="kbench/results_stage5.json", help="Output JSON path")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    
    # ====================
    # Gate 1: Compile
    # ====================
    print("=" * 80)
    print("Gate 1: COMPILE")
    print("=" * 80)
    try:
        ext = build_extension()
        print("✅ Extension built successfully\n")
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)
    
    # ====================
    # Gate 2: Correctness & Performance
    # ====================
    print("=" * 80)
    print("Gate 2: CORRECTNESS + PERFORMANCE")
    print("=" * 80)
    
    # Shape definitions
    shape_configs = {
        "small": (1, 8, 128, 64),
        "mission": (2, 8, 512, 64),
        "long": (2, 8, 2048, 64),
    }
    
    selected_shapes = args.shapes.split(",")
    results = []
    
    for shape_name in selected_shapes:
        if shape_name not in shape_configs:
            print(f"⚠️  Unknown shape '{shape_name}', skipping")
            continue
        
        B, H, S, D = shape_configs[shape_name]
        result = bench_shape(shape_name, B, H, S, D, args.iters, args.warmup, ext, seed=args.seed)
        results.append(result)
    
    # ====================
    # Save Results
    # ====================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for r in results:
        status = "✅ PASS" if r["correctness_pass"] else "❌ FAIL"
        print(f"{r['shape']:8s}: p50={r['p50_us']:7.2f}μs  speedup={r['speedup_vs_torch']:5.1f}×  max_err={r['max_err']:.4f}  {status}")
    
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {args.out}")
    
    # Exit with failure if any correctness checks failed
    if not all(r["correctness_pass"] for r in results):
        print("\n❌ SOME CORRECTNESS CHECKS FAILED")
        sys.exit(1)
    else:
        print("\n✅ ALL GATES PASSED")


if __name__ == "__main__":
    main()

