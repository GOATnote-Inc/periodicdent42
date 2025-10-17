#!/usr/bin/env python3
"""
Standalone KernelBench-style evaluation for Phase 4 kernel

Implements:
- fast_0: correctness rate (100 random tests)
- fast_1: % faster than PyTorch SDPA
- Speedup metric: t_baseline / t_solution
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
from pathlib import Path

# Add periodicdent42 to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import Phase 4 kernel
from bench.build_phase3_variant import build_phase3_variant

# Configuration
BATCH_SIZE = 1
N_HEADS = 8
SEQ_LEN = 512
HEAD_DIM = 64
N_CORRECTNESS = 100
N_TRIALS = 100
ATOL = 1e-3
RTOL = 1e-3

def measure_time(fn, *args, n_trials=100, warmup=10):
    """Measure average runtime over n_trials with warmup"""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)  # Convert to μs
    
    return sum(times) / len(times)

def pytorch_sdpa(Q, K, V):
    """PyTorch SDPA reference implementation"""
    scale = 1.0 / (HEAD_DIM ** 0.5)
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, scale=scale)

def main():
    print("=" * 70)
    print("KernelBench-style Evaluation: Phase 4 vs PyTorch SDPA")
    print("=" * 70)
    print(f"Problem: FlashAttention (B={BATCH_SIZE}, H={N_HEADS}, S={SEQ_LEN}, D={HEAD_DIM})")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Correctness tests: {N_CORRECTNESS}")
    print(f"Timing trials: {N_TRIALS}")
    print()

    # Build Phase 4 kernel
    print("📦 Building Phase 4 kernel...")
    fa_phase4 = build_phase3_variant(
        BLOCK_M=32,
        NUM_WARPS=8,
        VEC_WIDTH=4,
        SYNC_POLICY=2,
        REDUCE="warp"
    )
    scale = 1.0 / (HEAD_DIM ** 0.5)
    print("   ✅ Phase 4 loaded (M=32, W=8, VEC=4)")
    print()

    # ========== CORRECTNESS TESTING ==========
    print("🔍 CORRECTNESS TESTING")
    print("-" * 70)
    
    correct_count = 0
    max_diff = 0.0
    
    for i in range(N_CORRECTNESS):
        # Generate random inputs
        Q = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
        K = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
        V = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
        
        # Reference output
        with torch.no_grad():
            ref_out = pytorch_sdpa(Q, K, V)
        
        # Phase 4 output
        with torch.no_grad():
            phase4_out = fa_phase4.forward(Q, K, V, scale)
        
        # Check correctness
        diff = (ref_out - phase4_out).abs().max().item()
        max_diff = max(max_diff, diff)
        
        is_correct = torch.allclose(ref_out, phase4_out, atol=ATOL, rtol=RTOL)
        if is_correct:
            correct_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{N_CORRECTNESS} tests, {correct_count} correct, max_diff={max_diff:.6f}")
    
    fast_0 = correct_count / N_CORRECTNESS
    print()
    print(f"📊 Correctness Results:")
    print(f"   Passed: {correct_count}/{N_CORRECTNESS}")
    print(f"   Max diff: {max_diff:.6f}")
    print(f"   fast_0: {fast_0*100:.1f}%")
    print()

    # ========== PERFORMANCE TESTING ==========
    print("⏱️  PERFORMANCE TESTING")
    print("-" * 70)
    
    # Generate fixed inputs for timing
    Q = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    K = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    V = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    
    # Measure PyTorch SDPA
    print("   Measuring PyTorch SDPA...")
    baseline_time = measure_time(lambda: pytorch_sdpa(Q, K, V), n_trials=N_TRIALS)
    print(f"   ✅ PyTorch SDPA: {baseline_time:.2f} μs")
    
    # Measure Phase 4
    print("   Measuring Phase 4 kernel...")
    phase4_time = measure_time(lambda: fa_phase4.forward(Q, K, V, scale), n_trials=N_TRIALS)
    print(f"   ✅ Phase 4: {phase4_time:.2f} μs")
    
    # Calculate speedup
    speedup = baseline_time / phase4_time
    fast_1 = 1.0 if speedup > 1.0 else 0.0
    fast_2 = 1.0 if speedup > 2.0 else 0.0
    
    print()
    print(f"📊 Performance Results:")
    print(f"   Baseline (PyTorch SDPA): {baseline_time:.2f} μs")
    print(f"   Solution (Phase 4): {phase4_time:.2f} μs")
    print(f"   Speedup: {speedup:.3f}×")
    print(f"   fast_1 (faster than PyTorch): {fast_1*100:.1f}%")
    print(f"   fast_2 (2× faster): {fast_2*100:.1f}%")
    print()

    # ========== SUMMARY ==========
    print("=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"✅ fast_0 (correctness): {fast_0*100:.1f}%")
    print(f"{'✅' if fast_1 > 0 else '❌'} fast_1 (faster): {fast_1*100:.1f}%")
    print(f"{'✅' if fast_2 > 0 else '❌'} fast_2 (2× faster): {fast_2*100:.1f}%")
    print(f"📈 Speedup: {speedup:.3f}× ({'FASTER' if speedup > 1 else 'SLOWER'})")
    print()
    
    if speedup < 1.0:
        gap = 1.0 / speedup
        print(f"⚠️  Phase 4 is {gap:.1f}× SLOWER than PyTorch SDPA")
        print(f"    Target: {phase4_time / gap:.1f} μs to match PyTorch")
        print(f"    Gap: {phase4_time - baseline_time:.1f} μs to close")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

