#!/usr/bin/env python3
"""
Test Phase 5 Q@K^T WMMA integration
Tests both scalar fallback (USE_WMMA=0) and Tensor Core path (USE_WMMA=1)
"""
import torch
import os
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

def test_phase5(use_wmma: int):
    """Test Phase 5 kernel with USE_WMMA flag"""
    
    # Set build flags
    os.environ["USE_WMMA"] = str(use_wmma)
    os.environ["SYNC_POLICY"] = "2"  # Use Phase 4 light-barrier path
    
    # Build
    print(f"\n{'='*70}")
    print(f" Testing Phase 5 with USE_WMMA={use_wmma}".center(70))
    print(f"{'='*70}\n")
    
    sys.path.insert(0, str(repo_root / "bench"))
    from build_phase5_variant import build_phase5_variant
    
    if build_phase5_variant() != 0:
        print(f"❌ Build failed for USE_WMMA={use_wmma}")
        return False, 0.0
    
    # Import compiled module
    import fa_phase5
    
    # Test config
    B, H, S, D = 1, 8, 512, 64
    device = "cuda"
    dtype = torch.float16
    
    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    scale = 1.0 / (D ** 0.5)
    
    # Compute reference (PyTorch SDPA)
    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            scale=scale,
            is_causal=False
        )
    
    # Compute Phase 5
    with torch.no_grad():
        out = fa_phase5.forward(q, k, v, scale)
    
    # Check correctness
    max_diff = (out - ref).abs().max().item()
    mean_diff = (out - ref).abs().mean().item()
    
    passed = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    
    print(f"\n{'='*70}")
    print(f"Correctness Results (USE_WMMA={use_wmma}):".center(70))
    print(f"{'='*70}")
    print(f"  max_diff:  {max_diff:.6f}")
    print(f"  mean_diff: {mean_diff:.6f}")
    print(f"  Status:    {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"{'='*70}\n")
    
    if not passed:
        print(f"❌ Correctness test FAILED for USE_WMMA={use_wmma}")
        return False
    
    # Benchmark
    torch.cuda.synchronize()
    import time
    
    warmup = 10
    iters = 100
    
    # Warmup
    for _ in range(warmup):
        _ = fa_phase5.forward(q, k, v, scale)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = fa_phase5.forward(q, k, v, scale)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time_us = (elapsed / iters) * 1e6
    
    print(f"{'='*70}")
    print(f"Performance Results (USE_WMMA={use_wmma}):".center(70))
    print(f"{'='*70}")
    print(f"  Time:      {avg_time_us:.2f} μs")
    print(f"  Iters:     {iters}")
    print(f"{'='*70}\n")
    
    return True, avg_time_us

def main():
    print("\n" + "="*70)
    print(" Phase 5 Q@K^T WMMA Integration Test ".center(70))
    print("="*70 + "\n")
    
    # Test scalar fallback (USE_WMMA=0)
    print("Step 1: Test scalar fallback (USE_WMMA=0)")
    print("  Expected: Should match Phase 4 performance (~1028 μs)")
    success0, time0 = test_phase5(use_wmma=0)
    
    if not success0:
        print("\n❌ Scalar fallback test FAILED")
        return 1
    
    # Test WMMA path (USE_WMMA=1)
    print("\nStep 2: Test WMMA path (USE_WMMA=1)")
    print("  Expected: Should be faster than scalar (~500→100 μs for Q@K^T)")
    print("  Expected: Total time ~600-700 μs (5× speedup on Q@K^T)")
    success1, time1 = test_phase5(use_wmma=1)
    
    if not success1:
        print("\n❌ WMMA path test FAILED")
        return 1
    
    # Compare
    print("\n" + "="*70)
    print(" Phase 5 Q@K^T Results Summary ".center(70))
    print("="*70)
    print(f"  Scalar (USE_WMMA=0):  {time0:.2f} μs")
    print(f"  WMMA (USE_WMMA=1):    {time1:.2f} μs")
    
    if time1 < time0:
        speedup = time0 / time1
        improvement_us = time0 - time1
        print(f"  Speedup:              {speedup:.2f}×")
        print(f"  Improvement:          {improvement_us:.2f} μs faster")
        print(f"  Status:               ✅ WMMA is FASTER")
    else:
        print(f"  Status:               ⚠️ WMMA is SLOWER (may need tuning)")
    
    print("="*70 + "\n")
    
    print("✅ Phase 5 Step 2 (Q@K^T WMMA) complete!\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())

