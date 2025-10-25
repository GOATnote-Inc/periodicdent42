#!/usr/bin/env python3
"""Test script for FlashCore Fused FP32 P kernel."""

import torch
import sys
from pathlib import Path

# Add flashcore to path
sys.path.insert(0, str(Path(__file__).parent))

from build_fp32p import build_fp32p

def test_flashcore_fp32p():
    """Test the FP32 P kernel."""
    
    print("=" * 80)
    print("FlashCore Fused FP32 P Kernel - Test")
    print("=" * 80)
    print()
    
    # Build kernel
    print("Building kernel...")
    ext = build_fp32p()
    print("✅ Build successful")
    print()
    
    # Test configuration
    device = torch.device('cuda')
    dtype = torch.float16
    
    test_shapes = [
        ("mission", 1, 8, 512, 64),
        ("short", 1, 8, 256, 64),
        ("long", 1, 8, 1024, 64),
    ]
    
    results = {"name": "FlashCore FP32 P", "shapes": {}}
    
    for name, B, H, S, D in test_shapes:
        print(f"Testing shape: {name} (B={B}, H={H}, S={S}, D={D})")
        
        # Create inputs
        Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        K = torch.randn(B, H, S, D, device=device, dtype=dtype)
        V = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Allocate output
        O_kernel = torch.zeros(B, H, S, D, device=device, dtype=dtype)
        
        # Run kernel
        ext.forward(Q, K, V, O_kernel)
        
        # PyTorch reference
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V)
        
        # Check correctness
        max_err = (O_ref - O_kernel).abs().max().item()
        mean_err = (O_ref - O_kernel).abs().mean().item()
        
        print(f"  Correctness:")
        print(f"    max_err:  {max_err:.6f}")
        print(f"    mean_err: {mean_err:.6f}")
        
        # Performance (10 warmup + 100 test)
        torch.cuda.synchronize()
        for _ in range(10):
            ext.forward(Q, K, V, O_kernel)
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            ext.forward(Q, K, V, O_kernel)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        latency_us = (elapsed_ms / 100) * 1000
        
        print(f"  Performance:")
        print(f"    p50: {latency_us:.2f} μs")
        
        # Check pass/fail
        threshold = 0.10  # Relaxed threshold for FP32 P
        passed = max_err < threshold
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}: max_err {'<' if passed else '>='} {threshold}")
        print()
        
        results["shapes"][name] = {
            "max_err": max_err,
            "mean_err": mean_err,
            "latency_us": latency_us,
            "passed": passed
        }
    
    # Summary
    print("=" * 80)
    print("FlashCore FP32 P - Test Summary")
    print("=" * 80)
    print()
    
    all_passed = all(r["passed"] for r in results["shapes"].values())
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print()
        print("Detailed Results:")
        for shape_name, shape_results in results["shapes"].items():
            print(f"  {shape_name:10s}: max_err={shape_results['max_err']:.6f}, "
                  f"latency={shape_results['latency_us']:.2f} μs")
    else:
        print("⚠️ SOME TESTS FAILED - DEBUG REQUIRED")
        print()
        failed = [name for name, r in results["shapes"].items() if not r["passed"]]
        print(f"Failed shapes: {', '.join(failed)}")
        print()
        print("Detailed Results:")
        for shape_name, shape_results in results["shapes"].items():
            status = "PASSED" if shape_results["passed"] else "FAILED"
            print(f"  {shape_name:10s}: {status} (max_err={shape_results['max_err']:.6f})")
    
    return results

if __name__ == "__main__":
    results = test_flashcore_fp32p()

