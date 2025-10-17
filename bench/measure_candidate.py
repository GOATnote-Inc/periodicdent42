#!/usr/bin/env python3
"""
Measure candidate kernel performance with Nsight Compute metrics

Usage:
    IMPL=wmma python bench/measure_candidate.py --out .ci/cand.json
    IMPL=cublas python bench/measure_candidate.py --shape 1,8,512,64 --ncu
"""

import argparse
import json
import torch
import os
import sys
import subprocess
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.sdpa_oracle import bench_ms

def build_candidate(impl: str):
    """Build candidate kernel with IMPL selection"""
    from bench.build_phase3_variant import build_phase3_variant
    
    # Set environment for build
    os.environ['IMPL'] = impl
    
    # Build (will read IMPL from environment)
    result = build_phase3_variant()
    if result != 0:
        raise RuntimeError(f"Failed to build {impl} kernel")
    
    # Import compiled module
    import fa_phase3
    return fa_phase3

def run_ncu_metrics(impl: str, shape: tuple) -> dict:
    """Run Nsight Compute and extract metrics"""
    B, H, S, D = shape
    
    metrics = [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "dram__sectors_read.sum",
        "dram__sectors_write.sum"
    ]
    
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--metrics", ",".join(metrics),
        "--csv",
        "python", "-c",
        f"import torch, sys; sys.path.insert(0, '{Path.cwd()}'); "
        f"from bench.measure_candidate import build_candidate; "
        f"mod = build_candidate('{impl}'); "
        f"q = torch.randn({B},{H},{S},{D}, device='cuda', dtype=torch.float16); "
        f"mod.forward(q, q.clone(), q.clone(), 1.0/{D}**0.5);"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        # Parse CSV output
        lines = result.stdout.split('\n')
        ncu_metrics = {}
        for line in lines:
            for metric in metrics:
                if metric in line:
                    parts = line.split(',')
                    # Extract value (NCU CSV format varies)
                    value = parts[-1].strip() if parts else "0"
                    try:
                        ncu_metrics[metric.split('.')[0]] = float(value)
                    except ValueError:
                        ncu_metrics[metric.split('.')[0]] = 0.0
        return ncu_metrics
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Warning: NCU profiling failed: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Measure candidate kernel")
    parser.add_argument("--impl", default=None,
                       help="Implementation (custom_v3/cublas/wmma/etc, or read from IMPL env)")
    parser.add_argument("--shape", default="1,8,512,64", 
                       help="Tensor shape B,H,S,D")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                       help="Data type")
    parser.add_argument("--iters", type=int, default=100,
                       help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=20,
                       help="Warmup iterations")
    parser.add_argument("--ncu", action="store_true",
                       help="Run Nsight Compute profiling")
    parser.add_argument("--out", type=Path, default=None,
                       help="Output JSON file")
    args = parser.parse_args()
    
    # Get IMPL from argument or environment
    impl = args.impl or os.environ.get('IMPL', 'custom_v3')
    
    # Parse shape
    B, H, S, D = map(int, args.shape.split(','))
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    print(f"Measuring candidate kernel:")
    print(f"  IMPL: {impl}")
    print(f"  Shape: B={B}, H={H}, S={S}, D={D}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Iterations: {args.iters} (warmup: {args.warmup})")
    
    # Build kernel
    print(f"\n📦 Building {impl} kernel...")
    module = build_candidate(impl)
    
    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    scale = 1.0 / (D ** 0.5)
    
    # Benchmark
    median_ms, samples = bench_ms(
        lambda: module.forward(q, k, v, scale),
        iters=args.iters,
        warmup=args.warmup
    )
    
    # NCU metrics (optional)
    ncu_metrics = {}
    if args.ncu:
        print(f"\n🔍 Running Nsight Compute...")
        ncu_metrics = run_ncu_metrics(impl, (B, H, S, D))
    
    # Results
    results = {
        "impl": impl,
        "shape": [B, H, S, D],
        "dtype": args.dtype,
        "median_ms": median_ms,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": samples,
        "iters": args.iters,
        "warmup": args.warmup,
        "ncu_metrics": ncu_metrics
    }
    
    print(f"\nResults:")
    print(f"  Median: {median_ms:.4f} ms")
    print(f"  Min:    {min(samples):.4f} ms")
    print(f"  Max:    {max(samples):.4f} ms")
    
    if ncu_metrics:
        print(f"\nNsight Compute Metrics:")
        for k, v in ncu_metrics.items():
            print(f"  {k}: {v}")
    
    # Save
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\n✅ Saved to {args.out}")
    
    return results

if __name__ == "__main__":
    main()

