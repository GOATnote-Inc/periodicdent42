#!/usr/bin/env python3
"""
Measure PyTorch SDPA baseline performance

Usage:
    python bench/measure_sdpa.py --out .ci/sdpa.json
    python bench/measure_sdpa.py --backend flash --shape 1,8,512,64
"""

import argparse
import json
import torch
from pathlib import Path
from bench.sdpa_oracle import sdpa_ref, bench_ms

def main():
    parser = argparse.ArgumentParser(description="Measure SDPA baseline")
    parser.add_argument("--backend", default="flash", choices=["flash", "math"],
                       help="SDPA backend (flash or math)")
    parser.add_argument("--shape", default="1,8,512,64", 
                       help="Tensor shape B,H,S,D")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                       help="Data type")
    parser.add_argument("--iters", type=int, default=100,
                       help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=20,
                       help="Warmup iterations")
    parser.add_argument("--out", type=Path, default=None,
                       help="Output JSON file")
    args = parser.parse_args()
    
    # Parse shape
    B, H, S, D = map(int, args.shape.split(','))
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    print(f"Measuring SDPA baseline:")
    print(f"  Backend: {args.backend}")
    print(f"  Shape: B={B}, H={H}, S={S}, D={D}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Iterations: {args.iters} (warmup: {args.warmup})")
    
    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    scale = 1.0 / (D ** 0.5)
    
    # Benchmark
    backends = {
        "enable_flash": args.backend == "flash",
        "enable_math": args.backend == "math",
        "enable_mem_efficient": False
    }
    
    median_ms, samples = bench_ms(
        lambda: sdpa_ref(q, k, v, scale, backends),
        iters=args.iters,
        warmup=args.warmup
    )
    
    # Results
    results = {
        "backend": args.backend,
        "shape": [B, H, S, D],
        "dtype": args.dtype,
        "median_ms": median_ms,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": samples,
        "iters": args.iters,
        "warmup": args.warmup
    }
    
    print(f"\nResults:")
    print(f"  Median: {median_ms:.4f} ms")
    print(f"  Min:    {min(samples):.4f} ms")
    print(f"  Max:    {max(samples):.4f} ms")
    
    # Save
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\n✅ Saved to {args.out}")
    
    return results

if __name__ == "__main__":
    main()

