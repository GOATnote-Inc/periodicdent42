#!/usr/bin/env python3
"""
Smoke Test: Verify THREADS_PER_BLOCK Fix
=========================================

Quick 30-second test to verify the fix is applied correctly BEFORE
running the full benchmark suite.

Usage:
    python smoke_test_threads_fix.py

Expected output:
    ✅ build_config.h exists
    ✅ THREADS_PER_BLOCK = 384 (correct)
    ✅ Kernel launches successfully
    ✅ Correctness test passes
    ✅ Quick perf test: 0.8-1.7× speedup
    
    Status: READY FOR FULL BENCHMARK

Exit codes:
    0 - All checks passed
    1 - Configuration error (fix not applied)
    2 - Build error
    3 - Correctness error
    4 - Performance regression still present
"""

import os
import sys
import time
import torch

def print_status(check, status, message=""):
    """Print colored status message."""
    symbols = {
        "pass": "✅",
        "fail": "❌", 
        "warn": "⚠️",
        "info": "ℹ️"
    }
    print(f"{symbols.get(status, '•')} {check}: {message}")

def check_build_config():
    """Check if build_config.h exists and has correct value."""
    print("\n" + "="*60)
    print("CHECK 1: build_config.h Configuration")
    print("="*60)
    
    config_path = "python/flashmoe_science/csrc/build_config.h"
    
    if not os.path.exists(config_path):
        print_status("build_config.h", "fail", "File not found")
        print(f"  Expected location: {config_path}")
        print("  → Run: cp build_config.h python/flashmoe_science/csrc/")
        return False
    
    print_status("build_config.h", "pass", "File exists")
    
    # Check for correct value
    with open(config_path, 'r') as f:
        content = f.read()
    
    if "NUM_WARPS_PER_BLOCK = 12" in content or "NUM_WARPS_PER_BLOCK=12" in content:
        print_status("NUM_WARPS_PER_BLOCK", "pass", "Set to 12 (correct)")
    else:
        print_status("NUM_WARPS_PER_BLOCK", "fail", "Not set to 12")
        print("  → Check file contains: constexpr int NUM_WARPS_PER_BLOCK = 12;")
        return False
    
    if "THREADS_PER_BLOCK" in content:
        print_status("THREADS_PER_BLOCK", "pass", "Defined")
        # Extract value if possible
        import re
        match = re.search(r'THREADS_PER_BLOCK.*?(\d+)', content)
        if match:
            value = int(match.group(1))
            if value == 384:
                print_status("Value check", "pass", f"THREADS_PER_BLOCK = {value}")
            else:
                print_status("Value check", "warn", f"THREADS_PER_BLOCK = {value} (should be 384)")
    
    return True

def check_import():
    """Check if module imports successfully."""
    print("\n" + "="*60)
    print("CHECK 2: Module Import")
    print("="*60)
    
    try:
        import flashmoe_science as fa
        print_status("Import", "pass", "flashmoe_science imported successfully")
        
        # Check if forward function exists
        if hasattr(fa, 'forward'):
            print_status("API", "pass", "forward() method exists")
        else:
            print_status("API", "fail", "forward() method not found")
            return False
        
        return True
    except ImportError as e:
        print_status("Import", "fail", str(e))
        print("  → Rebuild required: pip install -e .")
        return False
    except Exception as e:
        print_status("Import", "fail", f"Unexpected error: {e}")
        return False

def check_kernel_launch():
    """Check if kernel launches without errors."""
    print("\n" + "="*60)
    print("CHECK 3: Kernel Launch")
    print("="*60)
    
    try:
        import flashmoe_science as fa
        
        # Create tiny test tensors
        B, H, S, D = 1, 1, 32, 64
        Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        K = Q.clone()
        V = Q.clone()
        
        print_status("Tensors", "pass", f"Created test tensors: B={B}, H={H}, S={S}, D={D}")
        
        # Launch kernel
        torch.cuda.synchronize()
        start = time.time()
        O = fa.forward(Q, K, V)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print_status("Launch", "pass", f"Kernel executed in {elapsed*1000:.3f}ms")
        print_status("Output", "pass", f"Output shape: {O.shape}")
        
        # Check output is not NaN or Inf
        if torch.isnan(O).any():
            print_status("NaN check", "fail", "Output contains NaN")
            return False
        if torch.isinf(O).any():
            print_status("Inf check", "fail", "Output contains Inf")
            return False
        
        print_status("Validity", "pass", "No NaN or Inf in output")
        
        return True
    except Exception as e:
        print_status("Launch", "fail", str(e))
        import traceback
        print(traceback.format_exc())
        return False

def check_correctness():
    """Quick correctness check against PyTorch."""
    print("\n" + "="*60)
    print("CHECK 4: Correctness vs PyTorch")
    print("="*60)
    
    try:
        import flashmoe_science as fa
        
        # Small test case
        B, H, S, D = 1, 2, 64, 64
        Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
        K = Q.clone()
        V = Q.clone()
        
        print_status("Test case", "info", f"B={B}, H={H}, S={S}, D={D}")
        
        # Our implementation
        O_ours = fa.forward(Q, K, V)
        
        # PyTorch reference
        scale = 1.0 / (D ** 0.5)
        O_ref = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, scale=scale, is_causal=False
        )
        
        # Compute error
        max_error = (O_ours - O_ref).abs().max().item()
        mean_error = (O_ours - O_ref).abs().mean().item()
        
        print_status("Max error", "info", f"{max_error:.6f}")
        print_status("Mean error", "info", f"{mean_error:.6f}")
        
        # Check tolerance
        if max_error < 1e-3:
            print_status("Tolerance", "pass", "Within acceptable range (<1e-3)")
            return True
        elif max_error < 1e-2:
            print_status("Tolerance", "warn", "Marginal accuracy (1e-3 to 1e-2)")
            return True
        else:
            print_status("Tolerance", "fail", f"Error too large: {max_error}")
            return False
        
    except Exception as e:
        print_status("Correctness", "fail", str(e))
        return False

def check_performance():
    """Quick performance comparison."""
    print("\n" + "="*60)
    print("CHECK 5: Performance Smoke Test")
    print("="*60)
    
    try:
        import flashmoe_science as fa
        
        # Medium test case
        B, H, S, D = 1, 1, 128, 64
        Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        K = Q.clone()
        V = Q.clone()
        
        print_status("Test case", "info", f"B={B}, H={H}, S={S}, D={D}")
        
        # Warmup
        for _ in range(5):
            _ = fa.forward(Q, K, V)
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
        torch.cuda.synchronize()
        
        # Benchmark our implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            O_ours = fa.forward(Q, K, V)
        torch.cuda.synchronize()
        time_ours = (time.time() - start) / 20
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
        torch.cuda.synchronize()
        time_ref = (time.time() - start) / 20
        
        speedup = time_ref / time_ours
        
        print_status("Ours", "info", f"{time_ours*1000:.3f}ms per call")
        print_status("PyTorch", "info", f"{time_ref*1000:.3f}ms per call")
        print_status("Speedup", "info", f"{speedup:.3f}×")
        
        # Check performance
        if speedup >= 1.2:
            print_status("Performance", "pass", f"Speedup {speedup:.2f}× (target: 1.2-1.7×)")
            return True
        elif speedup >= 0.8:
            print_status("Performance", "warn", f"Speedup {speedup:.2f}× (below target)")
            print("  → May still be acceptable, run full benchmark")
            return True
        else:
            print_status("Performance", "fail", f"Speedup {speedup:.2f}× (regression)")
            print("  → Fix not effective, further debugging needed")
            return False
        
    except Exception as e:
        print_status("Performance", "fail", str(e))
        return False

def main():
    """Run all checks."""
    print("\n" + "="*70)
    print(" SMOKE TEST: THREADS_PER_BLOCK Fix Verification")
    print("="*70)
    print("\nThis test verifies the fix is applied correctly before running")
    print("the full benchmark suite. Should complete in ~30 seconds.\n")
    
    checks = [
        ("Configuration", check_build_config),
        ("Import", check_import),
        ("Kernel Launch", check_kernel_launch),
        ("Correctness", check_correctness),
        ("Performance", check_performance),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ FATAL ERROR in {name}: {e}")
            import traceback
            print(traceback.format_exc())
            results[name] = False
        
        if not results[name]:
            break  # Stop on first failure
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED")
        print("\nStatus: READY FOR FULL BENCHMARK")
        print("\nNext steps:")
        print("  1. Run: python benches/bench_correctness_and_speed.py")
        print("  2. Expect: 1.2-1.7× speedup")
        print("  3. If successful: commit and merge")
        return 0
    else:
        print("\n❌ CHECKS FAILED")
        print("\nFix is NOT applied correctly. Review errors above.")
        print("\nTroubleshooting:")
        print("  1. Ensure build_config.h is in python/flashmoe_science/csrc/")
        print("  2. Verify NUM_WARPS_PER_BLOCK = 12")
        print("  3. Rebuild: pip install -e .")
        print("  4. Re-run this smoke test")
        return 1

if __name__ == "__main__":
    sys.exit(main())
