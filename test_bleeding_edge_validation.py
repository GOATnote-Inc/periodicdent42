#!/usr/bin/env python3
"""
Validation Suite for Bleeding Edge Attention Kernel
Checks: NaN/Inf, Determinism, Accuracy vs PyTorch SDPA
"""

import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import os
from pathlib import Path
import time

# Load compiled kernel
LIB_PATH = Path("build/bin/attention_bleeding_edge")
if not LIB_PATH.exists():
    print(f"❌ Kernel not found: {LIB_PATH}")
    print("   Build first: ./kernel_dev_pipeline.sh --stage=build")
    exit(1)

# Load shared library
lib = ctypes.CDLL(str(LIB_PATH))

# Function signature: launch_attention_bleeding_edge_64(Q, K, V, O, B, H, S, D, scale, is_causal, stream)
lib.launch_attention_bleeding_edge_64.argtypes = [
    ctypes.c_void_p,  # Q
    ctypes.c_void_p,  # K
    ctypes.c_void_p,  # V
    ctypes.c_void_p,  # O
    ctypes.c_int,     # B
    ctypes.c_int,     # H
    ctypes.c_int,     # S
    ctypes.c_int,     # D
    ctypes.c_float,   # scale
    ctypes.c_bool,    # is_causal
    ctypes.c_void_p,  # stream
]

def attention_bleeding_edge(Q, K, V, scale=None, is_causal=False):
    """
    Wrapper for bleeding edge kernel
    
    Args:
        Q, K, V: [B, H, S, D] tensors (FP16, CUDA)
        scale: softmax scale (default: 1/sqrt(D))
        is_causal: apply causal masking
    
    Returns:
        O: [B, H, S, D] output tensor
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype == torch.float16
    
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert D == 64 or D == 128, "Only D=64,128 supported"
    
    if scale is None:
        scale = 1.0 / np.sqrt(D)
    
    O = torch.empty_like(Q)
    
    # Call kernel
    func = lib.launch_attention_bleeding_edge_64 if D == 64 else lib.launch_attention_bleeding_edge_128
    func(
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        O.data_ptr(),
        B, H, S, D,
        scale,
        is_causal,
        None  # default stream
    )
    
    torch.cuda.synchronize()
    return O


def check_nan_inf(tensor, name="tensor"):
    """Check for NaN/Inf values"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
        return False
    return True


def test_correctness(B=2, H=8, S=512, D=64, num_runs=10):
    """Test correctness vs PyTorch SDPA"""
    print(f"\n{'='*60}")
    print(f"TEST 1: CORRECTNESS (vs PyTorch SDPA)")
    print(f"{'='*60}")
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    print(f"Runs:   {num_runs}")
    print()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Generate random inputs
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    
    scale = 1.0 / np.sqrt(D)
    
    # Ground truth (PyTorch)
    with torch.no_grad():
        O_ref = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    # Check reference for NaN/Inf
    if not check_nan_inf(O_ref, "PyTorch SDPA output"):
        print("❌ Reference implementation has NaN/Inf!")
        return False
    
    # Our kernel (multiple runs)
    errors = []
    
    for run in range(num_runs):
        O_ours = attention_bleeding_edge(Q, K, V, scale=scale)
        
        # Check for NaN/Inf
        if not check_nan_inf(O_ours, f"Run {run+1}"):
            print(f"❌ Run {run+1}: NaN/Inf detected")
            return False
        
        # Compute error
        diff = (O_ref - O_ours).abs()
        max_error = diff.max().item()
        avg_error = diff.mean().item()
        
        errors.append({'max': max_error, 'avg': avg_error})
        
        if run < 3:  # Print first 3 runs
            print(f"  Run {run+1}: max_err={max_error:.6f}, avg_err={avg_error:.6f}")
    
    # Summary
    max_errors = [e['max'] for e in errors]
    avg_errors = [e['avg'] for e in errors]
    
    print(f"\nSummary ({num_runs} runs):")
    print(f"  Max error: {np.mean(max_errors):.6f} ± {np.std(max_errors):.6f}")
    print(f"  Avg error: {np.mean(avg_errors):.6f} ± {np.std(avg_errors):.6f}")
    
    # Acceptance criteria
    threshold_max = 2e-3  # FP16 precision limit
    threshold_avg = 1e-3
    
    max_err_mean = np.mean(max_errors)
    avg_err_mean = np.mean(avg_errors)
    
    pass_max = max_err_mean < threshold_max
    pass_avg = avg_err_mean < threshold_avg
    
    print(f"\nAcceptance:")
    print(f"  Max error < {threshold_max}: {'✅ PASS' if pass_max else '❌ FAIL'}")
    print(f"  Avg error < {threshold_avg}: {'✅ PASS' if pass_avg else '❌ FAIL'}")
    
    return pass_max and pass_avg


def test_determinism(B=2, H=8, S=512, D=64, num_runs=100):
    """Test determinism (same input → same output)"""
    print(f"\n{'='*60}")
    print(f"TEST 2: DETERMINISM")
    print(f"{'='*60}")
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    print(f"Runs:   {num_runs}")
    print()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Fixed input
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    
    # First run (reference)
    O_ref = attention_bleeding_edge(Q, K, V)
    
    # Subsequent runs
    mismatches = 0
    max_diff = 0.0
    
    for run in range(1, num_runs):
        O_test = attention_bleeding_edge(Q, K, V)
        
        # Bit-exact comparison
        diff = (O_ref - O_test).abs().max().item()
        
        if diff > 0:
            mismatches += 1
            max_diff = max(max_diff, diff)
    
    print(f"Results:")
    print(f"  Mismatches:     {mismatches} / {num_runs-1}")
    print(f"  Max difference: {max_diff:.10f}")
    
    # Accept small differences due to scheduling variance
    threshold = 1e-6
    
    if max_diff < threshold:
        print(f"\n✅ PASS: Deterministic (max_diff < {threshold})")
        return True
    else:
        print(f"\n❌ FAIL: Non-deterministic (max_diff = {max_diff} ≥ {threshold})")
        return False


def test_performance(B=2, H=8, S=512, D=64, num_runs=100):
    """Benchmark performance"""
    print(f"\n{'='*60}")
    print(f"TEST 3: PERFORMANCE")
    print(f"{'='*60}")
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    print(f"Runs:   {num_runs} (warmup: 10)")
    print()
    
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = attention_bleeding_edge(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        O = attention_bleeding_edge(Q, K, V)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    
    # Compute TFLOPS
    # FlashAttention: 4*B*H*S^2*D FLOPs (2× for Q@K^T, 2× for P@V)
    flops = 4 * B * H * S * S * D
    tflops = (flops / 1e12) / (np.mean(times) / 1000)
    
    print(f"Latency:")
    print(f"  Mean:  {np.mean(times):.3f} ms")
    print(f"  Std:   {np.std(times):.3f} ms")
    print(f"  P50:   {np.percentile(times, 50):.3f} ms")
    print(f"  P95:   {np.percentile(times, 95):.3f} ms")
    print(f"  P99:   {np.percentile(times, 99):.3f} ms")
    print(f"\nThroughput:")
    print(f"  TFLOPS: {tflops:.2f}")
    
    # Compare to PyTorch
    print(f"\nBaseline (PyTorch SDPA):")
    
    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    times_pytorch = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_pytorch.append((end - start) * 1000)
    
    times_pytorch = np.array(times_pytorch)
    tflops_pytorch = (flops / 1e12) / (np.mean(times_pytorch) / 1000)
    
    print(f"  Latency: {np.mean(times_pytorch):.3f} ms")
    print(f"  TFLOPS:  {tflops_pytorch:.2f}")
    
    speedup = np.mean(times_pytorch) / np.mean(times)
    tflops_improvement = tflops / tflops_pytorch
    
    print(f"\nSpeedup:")
    print(f"  Latency: {speedup:.2f}× faster")
    print(f"  TFLOPS:  {tflops_improvement:.2f}× higher")
    
    # Target: >50 TFLOPS, >15× vs PyTorch
    pass_tflops = tflops > 50
    pass_speedup = tflops_improvement > 15
    
    print(f"\nTargets:")
    print(f"  TFLOPS > 50:   {'✅ PASS' if pass_tflops else '⚠️  CLOSE' if tflops > 40 else '❌ FAIL'}")
    print(f"  Speedup > 15×: {'✅ PASS' if pass_speedup else '⚠️  CLOSE' if tflops_improvement > 10 else '❌ FAIL'}")
    
    return pass_tflops and pass_speedup


def main():
    print("="*60)
    print("BLEEDING EDGE ATTENTION KERNEL - VALIDATION SUITE")
    print("="*60)
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    
    # Run tests
    results = {}
    
    try:
        results['correctness'] = test_correctness(B=2, H=8, S=512, D=64, num_runs=10)
    except Exception as e:
        print(f"\n❌ Correctness test failed: {e}")
        results['correctness'] = False
    
    try:
        results['determinism'] = test_determinism(B=2, H=8, S=512, D=64, num_runs=100)
    except Exception as e:
        print(f"\n❌ Determinism test failed: {e}")
        results['determinism'] = False
    
    try:
        results['performance'] = test_performance(B=2, H=8, S=512, D=64, num_runs=100)
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        results['performance'] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.capitalize():15} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎉 SUCCESS: All tests passed!")
        print(f"   Kernel is ready for production deployment.")
        return 0
    else:
        print(f"\n⚠️  WARNING: Some tests failed.")
        print(f"   Review errors above before deployment.")
        return 1


if __name__ == "__main__":
    exit(main())
