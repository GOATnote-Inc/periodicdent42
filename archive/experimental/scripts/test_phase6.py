#!/usr/bin/env python3
"""
Test Phase 6 (Aggressive Scalar Optimization) - Correctness & Performance
"""
import torch
import sys
import os
import time
from pathlib import Path

def setup_env():
    """Setup Python path"""
    repo_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(repo_root / "bench"))

def test_phase6():
    """Test Phase 6 kernel"""
    
    print("=" * 70)
    print("Phase 6: Aggressive Scalar Optimization")
    print("Target: 1,028 → 500-600 μs (2× speedup)")
    print("=" * 70)
    print()
    
    # Setup
    setup_env()
    from build_phase6 import build_phase6
    
    # Build
    print("Building Phase 6 kernel...")
    if build_phase6() != 0:
        print("❌ Build failed")
        return False, 0.0
    
    import fa_phase6
    
    # Test configuration
    batch_size = 1
    num_heads = 8
    seq_len = 512
    head_dim = 64
    
    device = torch.device('cuda:0')
    dtype = torch.float16
    
    print(f"\n📊 Test Configuration:")
    print(f"   B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    print()
    
    # Generate test data
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # PyTorch SDPA baseline
    print("Running PyTorch SDPA (baseline)...")
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            output_ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=softmax_scale
            )
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(100):
            output_ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=softmax_scale
            )
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        ref_time_us = (end_time - start_time) * 1e6 / 100
        print(f"   Time: {ref_time_us:.2f} μs")
    
    # Phase 6 kernel
    print("\nRunning Phase 6 kernel...")
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            output_custom = fa_phase6.forward(q, k, v, softmax_scale)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(100):
            output_custom = fa_phase6.forward(q, k, v, softmax_scale)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        custom_time_us = (end_time - start_time) * 1e6 / 100
        print(f"   Time: {custom_time_us:.2f} μs")
    
    # Correctness check
    print("\n🔍 Correctness Check:")
    max_diff = (output_custom - output_ref).abs().max().item()
    mean_diff = (output_custom - output_ref).abs().mean().item()
    passed = torch.allclose(output_custom, output_ref, atol=1e-3, rtol=1e-3)
    
    print(f"   Max diff: {max_diff:.6f}")
    print(f"   Mean diff: {mean_diff:.6f}")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("📊 RESULTS:")
    print("=" * 70)
    print(f"PyTorch SDPA:  {ref_time_us:8.2f} μs")
    print(f"Phase 6:       {custom_time_us:8.2f} μs")
    print(f"Speedup:       {ref_time_us / custom_time_us:8.2f}×")
    
    if passed:
        print(f"\n✅ Phase 6 PASSED (correctness maintained)")
    else:
        print(f"\n❌ Phase 6 FAILED (max_diff={max_diff:.6f} > 0.001)")
    
    print("=" * 70)
    
    return passed, custom_time_us

def main():
    try:
        success, time_us = test_phase6()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

