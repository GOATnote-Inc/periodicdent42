"""
Test script for Phase C.1: WMMA Q@K^T kernel

Tests:
1. Correctness vs Phase B (78 μs baseline)
2. Performance measurement
3. NCU profiling readiness
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Build kernel
from bench.build_wmma_qkt import build_wmma_qkt

def test_wmma_qkt():
    print("=" * 70)
    print("Phase C.1 Test: WMMA Q@K^T Kernel")
    print("=" * 70)
    print()
    
    # Configuration
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    print(f"Configuration:")
    print(f"  Batch: {B}")
    print(f"  Heads: {H}")
    print(f"  Seq Length: {S}")
    print(f"  Head Dim: {D}")
    print(f"  Scale: {scale:.6f}")
    print()
    
    # Build kernel
    print("Step 1: Building WMMA Q@K^T kernel...")
    module = build_wmma_qkt()
    print("  ✅ Kernel built")
    print()
    
    # Generate test data
    print("Step 2: Generate test data...")
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    print("  ✅ Q, K, V generated")
    print()
    
    # Reference: Phase B (cuBLAS hybrid)
    print("Step 3: Compute Phase B reference (cuBLAS)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        # Manual path (cuBLAS + softmax + matmul)
        S_ref = torch.matmul(Q, K.transpose(-2, -1)) * scale
        P_ref = torch.softmax(S_ref, dim=-1)
        O_ref = torch.matmul(P_ref, V)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    phase_b_time_us = (t1 - t0) * 1e6
    
    print(f"  ✅ Phase B: {phase_b_time_us:.2f} μs")
    print()
    
    # Test WMMA kernel
    print("Step 4: Test WMMA Q@K^T kernel...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    O_wmma = module.forward(Q, K, V, scale)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    wmma_time_us = (t1 - t0) * 1e6
    
    print(f"  ✅ WMMA: {wmma_time_us:.2f} μs")
    print()
    
    # Correctness check
    print("Step 5: Verify correctness...")
    diff = (O_ref - O_wmma).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Tolerance: 2e-3")
    
    if max_diff < 2e-3:
        print("  ✅ Correctness: PASSED")
    else:
        print(f"  ❌ Correctness: FAILED (max_diff={max_diff:.6f})")
    print()
    
    # Benchmark
    print("Step 6: Benchmark (100 iterations)...")
    warmup = 10
    iters = 100
    
    # Warmup
    for _ in range(warmup):
        O = module.forward(Q, K, V, scale)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for _ in range(iters):
        O = module.forward(Q, K, V, scale)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    wmma_bench_us = (t1 - t0) * 1e6 / iters
    
    print(f"  ✅ WMMA (benchmarked): {wmma_bench_us:.2f} μs")
    print()
    
    # Compare with targets
    print("Step 7: Compare with targets...")
    sdpa_time_us = 40.0  # SDPA Flash baseline
    phase_b_baseline = 78.0  # Phase B baseline
    phase_c1_target = 55.0  # Phase C.1 target
    
    print(f"  Phase B baseline: {phase_b_baseline:.2f} μs")
    print(f"  Phase C.1 target: {phase_c1_target:.2f} μs")
    print(f"  WMMA achieved: {wmma_bench_us:.2f} μs")
    print(f"  SDPA Flash: {sdpa_time_us:.2f} μs")
    print()
    
    speedup_vs_phase_b = phase_b_baseline / wmma_bench_us
    gap_to_sdpa = wmma_bench_us / sdpa_time_us
    
    print(f"  Speedup vs Phase B: {speedup_vs_phase_b:.2f}×")
    print(f"  Gap to SDPA: {gap_to_sdpa:.2f}× slower")
    print()
    
    # Final verdict
    print("=" * 70)
    success = True
    
    if max_diff < 2e-3:
        print("✅ CORRECTNESS: PASSED")
    else:
        print(f"❌ CORRECTNESS: FAILED (max_diff={max_diff:.6f})")
        success = False
    
    if wmma_bench_us < phase_c1_target:
        print(f"✅ PERFORMANCE: EXCEEDED TARGET ({wmma_bench_us:.2f} < {phase_c1_target:.2f} μs)")
    elif wmma_bench_us < phase_b_baseline:
        print(f"✅ PERFORMANCE: IMPROVED ({wmma_bench_us:.2f} < {phase_b_baseline:.2f} μs)")
    else:
        print(f"⚠️  PERFORMANCE: REGRESSION ({wmma_bench_us:.2f} >= {phase_b_baseline:.2f} μs)")
        success = False
    
    print("=" * 70)
    print()
    
    return success

if __name__ == "__main__":
    success = test_wmma_qkt()
    sys.exit(0 if success else 1)

