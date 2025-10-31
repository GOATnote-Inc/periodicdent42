#!/usr/bin/env python3
"""
Gate 7 Correctness Validation
Tests TMA kernel vs Gate 6 baseline and PyTorch SDPA
"""

import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import os
import sys
from pathlib import Path

# Configuration
KERNEL_PATH = Path("build/bin/attention_gate7")
GATE6_PATH = Path("build/bin/attention_bleeding_edge")  # From previous work

# Color output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_pass(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_fail(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warn(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

# Load kernel
if not KERNEL_PATH.exists():
    print_fail(f"Kernel not found: {KERNEL_PATH}")
    print_info("Build first: ./build_gate7.sh")
    sys.exit(1)

lib = ctypes.CDLL(str(KERNEL_PATH))

# Function signatures
lib.launch_attention_tma_wgmma_64.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_bool, ctypes.c_void_p
]

lib.launch_attention_tma_wgmma_128.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_bool, ctypes.c_void_p
]

def attention_gate7(Q, K, V, scale=None, is_causal=False):
    """Gate 7 TMA kernel wrapper"""
    assert Q.is_cuda and Q.dtype == torch.float16
    B, H, S, D = Q.shape
    
    if scale is None:
        scale = 1.0 / np.sqrt(D)
    
    O = torch.empty_like(Q)
    
    func = lib.launch_attention_tma_wgmma_64 if D == 64 else lib.launch_attention_tma_wgmma_128
    func(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        B, H, S, D, scale, is_causal, None)
    
    torch.cuda.synchronize()
    return O

def test_correctness_vs_pytorch(B=2, H=8, S=512, D=64):
    """Test vs PyTorch SDPA"""
    print(f"\n{'='*60}")
    print(f"TEST 1: CORRECTNESS VS PYTORCH SDPA")
    print(f"{'='*60}")
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Generate inputs
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    
    scale = 1.0 / np.sqrt(D)
    
    # Reference (PyTorch)
    with torch.no_grad():
        O_ref = F.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    # Gate 7
    O_gate7 = attention_gate7(Q, K, V, scale=scale)
    
    # Check for NaN/Inf
    has_nan = torch.isnan(O_gate7).any().item()
    has_inf = torch.isinf(O_gate7).any().item()
    
    if has_nan or has_inf:
        print_fail(f"Output contains NaN={has_nan}, Inf={has_inf}")
        return False
    
    # Compute errors
    diff = (O_ref - O_gate7).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    
    print(f"\nError Metrics:")
    print(f"  Max error:  {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  RMSE:       {rmse:.6f}")
    
    # Thresholds
    threshold_max = 2e-3  # FP16 limit
    threshold_mean = 1e-3
    threshold_rmse = 1e-3
    
    pass_max = max_error < threshold_max
    pass_mean = mean_error < threshold_mean
    pass_rmse = rmse < threshold_rmse
    
    print(f"\nValidation:")
    if pass_max:
        print_pass(f"Max error < {threshold_max}")
    else:
        print_fail(f"Max error {max_error:.6f} ≥ {threshold_max}")
    
    if pass_mean:
        print_pass(f"Mean error < {threshold_mean}")
    else:
        print_fail(f"Mean error {mean_error:.6f} ≥ {threshold_mean}")
    
    if pass_rmse:
        print_pass(f"RMSE < {threshold_rmse}")
    else:
        print_fail(f"RMSE {rmse:.6f} ≥ {threshold_rmse}")
    
    return pass_max and pass_mean and pass_rmse

def test_determinism(B=2, H=8, S=512, D=64, runs=10):
    """Test determinism"""
    print(f"\n{'='*60}")
    print(f"TEST 2: DETERMINISM")
    print(f"{'='*60}")
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    print(f"Runs:   {runs}")
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    
    # First run
    O_ref = attention_gate7(Q, K, V)
    
    # Subsequent runs
    max_diff = 0.0
    mismatches = 0
    
    for run in range(1, runs):
        O_test = attention_gate7(Q, K, V)
        diff = (O_ref - O_test).abs().max().item()
        
        if diff > 1e-7:
            mismatches += 1
            max_diff = max(max_diff, diff)
    
    print(f"\nResults:")
    print(f"  Mismatches:     {mismatches} / {runs-1}")
    print(f"  Max difference: {max_diff:.10f}")
    
    if max_diff < 1e-7:
        print_pass(f"Deterministic (max_diff < 1e-7)")
        return True
    else:
        print_warn(f"Non-deterministic (max_diff = {max_diff})")
        return mismatches == 0

def test_causal_masking(B=1, H=4, S=256, D=64):
    """Test causal masking"""
    print(f"\n{'='*60}")
    print(f"TEST 3: CAUSAL MASKING")
    print(f"{'='*60}")
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    
    # Reference (PyTorch with causal)
    with torch.no_grad():
        O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    
    # Gate 7 with causal
    O_gate7 = attention_gate7(Q, K, V, is_causal=True)
    
    # Check
    diff = (O_ref - O_gate7).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    
    print(f"\nCausal Mask Error:")
    print(f"  Max error:  {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    
    if max_error < 2e-3:
        print_pass("Causal masking correct")
        return True
    else:
        print_fail(f"Causal masking error: {max_error:.6f}")
        return False

def test_multiple_configs():
    """Test various configurations"""
    print(f"\n{'='*60}")
    print(f"TEST 4: MULTIPLE CONFIGURATIONS")
    print(f"{'='*60}")
    
    configs = [
        (1, 1, 128, 64),   # Minimal
        (2, 8, 512, 64),   # Standard
        (4, 16, 1024, 64), # Large
        (2, 8, 512, 128),  # D=128
    ]
    
    results = []
    
    for B, H, S, D in configs:
        print(f"\nTesting B={B}, H={H}, S={S}, D={D}...", end=" ")
        
        try:
            Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
            K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
            V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
            
            O_ref = F.scaled_dot_product_attention(Q, K, V)
            O_gate7 = attention_gate7(Q, K, V)
            
            error = (O_ref - O_gate7).abs().max().item()
            
            if error < 2e-3:
                print_pass(f"PASS (error={error:.6f})")
                results.append(True)
            else:
                print_fail(f"FAIL (error={error:.6f})")
                results.append(False)
        except Exception as e:
            print_fail(f"ERROR: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nSummary: {passed}/{total} configurations passed")
    
    return all(results)

def main():
    print("="*60)
    print("GATE 7 CORRECTNESS VALIDATION")
    print("="*60)
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    
    results = {}
    
    try:
        results['pytorch'] = test_correctness_vs_pytorch(B=2, H=8, S=512, D=64)
    except Exception as e:
        print_fail(f"PyTorch test failed: {e}")
        results['pytorch'] = False
    
    try:
        results['determinism'] = test_determinism(B=2, H=8, S=512, D=64, runs=10)
    except Exception as e:
        print_fail(f"Determinism test failed: {e}")
        results['determinism'] = False
    
    try:
        results['causal'] = test_causal_masking(B=1, H=4, S=256, D=64)
    except Exception as e:
        print_fail(f"Causal test failed: {e}")
        results['causal'] = False
    
    try:
        results['configs'] = test_multiple_configs()
    except Exception as e:
        print_fail(f"Config test failed: {e}")
        results['configs'] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20} {status}")
    
    all_passed = all(results.values())
    
    print(f"\n{'='*60}")
    if all_passed:
        print_pass("ALL TESTS PASSED - Gate 7 Ready for Benchmarking")
        return 0
    else:
        print_fail("SOME TESTS FAILED - Review errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
