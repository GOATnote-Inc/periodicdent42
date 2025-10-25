"""
Test Phase D.1: Minimal Custom Kernel

Compare against:
1. PyTorch SDPA (25.94 μs baseline)
2. Phase C best (26.00 μs PyTorch backends)

Goal: Establish baseline (expected: 100-200 μs)
This is the STARTING POINT for D.2-D.5 optimizations!
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Build kernel
from bench.build_d1 import build_d1

def test_phase_d1():
    print("=" * 70)
    print("Phase D.1: Minimal Custom Kernel Test")
    print("=" * 70)
    print()
    print("GOAL: Pure CUDA baseline (no PyTorch backends)")
    print("EXPECTED: 100-200 μs (will optimize in D.2-D.5)")
    print("TARGET (D.5): < 5 μs (5× faster than SDPA)")
    print()
    
    # Configuration
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    print(f"Configuration: B={B}, H={H}, S={S}, D={D}")
    print()
    
    # Build kernel
    print("Step 1: Building D.1 kernel...")
    module = build_d1()
    print()
    
    # Generate test data
    print("Step 2: Generate test data...")
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    print("  ✅ Data generated")
    print()
    
    # Reference: PyTorch SDPA
    print("Step 3: Compute reference (PyTorch SDPA)...")
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True
    ):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sdpa_time_us = (t1 - t0) * 1e6
    
    print(f"  ✅ SDPA: {sdpa_time_us:.2f} μs (production baseline)")
    print()
    
    # Test D.1 kernel
    print("Step 4: Test Phase D.1 minimal kernel...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    O_d1 = module.forward(Q, K, V, scale)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    d1_time_us = (t1 - t0) * 1e6
    
    print(f"  ✅ D.1: {d1_time_us:.2f} μs")
    print()
    
    # Correctness
    print("Step 5: Verify correctness...")
    diff = (O_ref - O_d1).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Tolerance: 2e-3")
    
    if max_diff < 2e-3:
        print("  ✅ Correctness: PASSED")
        correctness = True
    else:
        print(f"  ❌ Correctness: FAILED (max_diff={max_diff:.6f})")
        correctness = False
    print()
    
    # Benchmark
    if correctness:
        print("Step 6: Benchmark (100 iterations)...")
        warmup = 10
        iters = 100
        
        # Warmup
        for _ in range(warmup):
            _ = module.forward(Q, K, V, scale)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = module.forward(Q, K, V, scale)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        d1_bench_us = (t1 - t0) * 1e6 / iters
        
        print(f"  ✅ D.1 (benchmarked): {d1_bench_us:.2f} μs")
        print()
        
        # Analysis
        print("=" * 70)
        print("PHASE D.1 RESULTS")
        print("=" * 70)
        print()
        print(f"PyTorch SDPA:      {sdpa_time_us:.2f} μs (production)")
        print(f"Phase D.1:         {d1_bench_us:.2f} μs (our baseline)")
        print()
        
        ratio = d1_bench_us / sdpa_time_us
        
        if d1_bench_us < 200:
            print(f"✅ Performance: {d1_bench_us:.2f} μs (within expected 100-200 μs)")
        else:
            print(f"⚠️  Performance: {d1_bench_us:.2f} μs (slower than expected)")
        
        print()
        print(f"vs SDPA: {ratio:.2f}× slower (expected: 4-8× slower)")
        print()
        print("ROADMAP TO < 5 μs:")
        print(f"  D.1 (current):  {d1_bench_us:.2f} μs → baseline ✅")
        print(f"  D.2 (memory):   target < 50 μs")
        print(f"  D.3 (TC):       target < 20 μs")
        print(f"  D.4 (fusion):   target < 10 μs")
        print(f"  D.5 (extreme):  target < 5 μs (5× faster than SDPA) 🎯")
        print()
        
        speedup_needed = d1_bench_us / 5.0
        print(f"Speedup needed: {speedup_needed:.1f}× from D.1 → D.5")
        print()
        
        return correctness and (d1_bench_us < 500)
    else:
        print("⏭️  Skipping benchmark (correctness failed)")
        return False

if __name__ == "__main__":
    success = test_phase_d1()
    
    print("=" * 70)
    if success:
        print("✅ Phase D.1: SUCCESS - Baseline established!")
        print("   Next: Phase D.2 (memory optimization)")
    else:
        print("❌ Phase D.1: FAILED - Debug needed")
    print("=" * 70)
    
    sys.exit(0 if success else 1)

