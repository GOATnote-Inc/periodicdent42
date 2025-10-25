"""
Phase B.2: Hybrid FlashAttention (cuBLAS Q@K^T + Custom Softmax+PV)

Two-stage approach:
1. cuBLAS: Compute S = Q @ K^T (all tiles, batched)
2. Custom kernel: softmax(S) @ V (online softmax + PV)

This is more practical than device-side cuBLAS launches.
"""

import torch
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def cublas_qkt_batched(Q, K, scale):
    """
    Compute S = (Q @ K^T) * scale using PyTorch (which uses cuBLAS internally)
    
    Q: [B, H, S, D]
    K: [B, H, S, D]
    Returns S: [B, H, S, S]
    """
    # PyTorch's @ operator uses cuBLAS for matrix multiplication
    # This is equivalent to our cuBLAS calls, but handled at Python level
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    return S

def test_hybrid():
    print("=" * 70)
    print("Phase B.2 Test: Hybrid FlashAttention (cuBLAS Q@K^T)")
    print("=" * 70)
    print()
    
    # Test configuration
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    print(f"Configuration:")
    print(f"  Batch: {B}")
    print(f"  Heads: {H}")
    print(f"  Seq Length: {S}")
    print(f"  Head Dim: {D}")
    print(f"  Scale: {scale:.6f}")
    print()
    
    # Generate test data
    print("Step 1: Generate test data...")
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    print("  ✅ Q, K, V generated")
    print()
    
    # Reference: PyTorch SDPA
    print("Step 2: Compute PyTorch SDPA reference...")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
    print("  ✅ Reference computed")
    print()
    
    # Hybrid Stage 1: cuBLAS Q@K^T
    print("Step 3: Stage 1 - cuBLAS Q@K^T...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    S = cublas_qkt_batched(Q, K, scale)  # [B, H, S, S]
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    qkt_time_us = (t1 - t0) * 1e6
    
    print(f"  ✅ S = Q @ K^T computed")
    print(f"  ✅ Time: {qkt_time_us:.2f} μs")
    print(f"  ✅ Shape: {S.shape}")
    print()
    
    # Hybrid Stage 2: Softmax + P@V (using PyTorch for now)
    print("Step 4: Stage 2 - Softmax + P@V...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # Softmax
    P = torch.softmax(S, dim=-1)  # [B, H, S, S]
    
    # P @ V
    O_hybrid = torch.matmul(P, V)  # [B, H, S, D]
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    softmax_pv_time_us = (t1 - t0) * 1e6
    
    print(f"  ✅ Softmax + P@V computed")
    print(f"  ✅ Time: {softmax_pv_time_us:.2f} μs")
    print()
    
    # Total time
    total_time_us = qkt_time_us + softmax_pv_time_us
    print(f"Total Time: {total_time_us:.2f} μs")
    print(f"  Stage 1 (Q@K^T): {qkt_time_us:.2f} μs ({qkt_time_us/total_time_us*100:.1f}%)")
    print(f"  Stage 2 (Softmax+PV): {softmax_pv_time_us:.2f} μs ({softmax_pv_time_us/total_time_us*100:.1f}%)")
    print()
    
    # Correctness check
    print("Step 5: Verify correctness...")
    diff = (O_ref - O_hybrid).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Tolerance: 2e-3")
    print()
    
    # Benchmark
    print("Step 6: Benchmark (100 iterations)...")
    warmup = 10
    iters = 100
    
    # Warmup
    for _ in range(warmup):
        S = cublas_qkt_batched(Q, K, scale)
        P = torch.softmax(S, dim=-1)
        O = torch.matmul(P, V)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for _ in range(iters):
        S = cublas_qkt_batched(Q, K, scale)
        P = torch.softmax(S, dim=-1)
        O = torch.matmul(P, V)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    hybrid_time_us = (t1 - t0) * 1e6 / iters
    
    print(f"  ✅ Hybrid: {hybrid_time_us:.2f} μs")
    print()
    
    # Compare with Phase 4
    print("Step 7: Compare with baselines...")
    phase4_time_us = 870.49  # From Phase A results
    sdpa_time_us = 49.73     # From Phase A results
    
    print(f"  Phase 4 (scalar): {phase4_time_us:.2f} μs")
    print(f"  Hybrid (cuBLAS Q@K^T): {hybrid_time_us:.2f} μs")
    print(f"  PyTorch SDPA: {sdpa_time_us:.2f} μs")
    print()
    print(f"  Speedup vs Phase 4: {phase4_time_us / hybrid_time_us:.2f}×")
    print(f"  Gap to SDPA: {hybrid_time_us / sdpa_time_us:.2f}× slower")
    print()
    
    # Final verdict
    print("=" * 70)
    if max_diff < 2e-3:
        print("✅ CORRECTNESS: PASSED")
    else:
        print(f"❌ CORRECTNESS: FAILED (max_diff={max_diff:.6f})")
    
    if 300 < hybrid_time_us < 700:
        print("✅ PERFORMANCE: Within expected range (300-700 μs)")
    else:
        print(f"⚠️  PERFORMANCE: {hybrid_time_us:.2f} μs (expected 300-700 μs)")
    
    print("=" * 70)
    print()
    
    return max_diff < 2e-3 and 300 < hybrid_time_us < 700

if __name__ == "__main__":
    success = test_hybrid()
    sys.exit(0 if success else 1)

