#!/usr/bin/env python3
"""
Compare Phase 4 vs Phase 6 performance
"""
import torch
import sys
import os
import time
from pathlib import Path

def setup_env():
    repo_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(repo_root / "bench"))

def test_kernel(kernel_name, build_fn):
    """Test a kernel and return performance"""
    setup_env()
    
    # Build
    try:
        if build_fn() != 0:
            return False, 0.0, 0.0, 0.0
    except Exception as e:
        print(f"Build error: {e}")
        return False, 0.0, 0.0, 0.0
    
    # Import
    try:
        module = __import__(kernel_name)
    except Exception as e:
        print(f"Import error: {e}")
        return False, 0.0, 0.0, 0.0
    
    # Test config
    batch_size = 1
    num_heads = 8
    seq_len = 512
    head_dim = 64
    device = torch.device('cuda:0')
    dtype = torch.float16
    
    # Generate data
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # PyTorch baseline
    with torch.no_grad():
        for _ in range(10):
            output_ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=softmax_scale
            )
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            output_ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=softmax_scale
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        sdpa_time = (end - start) * 1e6 / 100
    
    # Custom kernel
    with torch.no_grad():
        for _ in range(10):
            output_custom = module.forward(q, k, v, softmax_scale)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            output_custom = module.forward(q, k, v, softmax_scale)
        torch.cuda.synchronize()
        end = time.perf_counter()
        custom_time = (end - start) * 1e6 / 100
    
    # Correctness
    max_diff = (output_custom - output_ref).abs().max().item()
    passed = torch.allclose(output_custom, output_ref, atol=1e-3, rtol=1e-3)
    
    return passed, custom_time, sdpa_time, max_diff

def main():
    print("=" * 70)
    print("PHASE COMPARISON: Phase 4 vs Phase 6")
    print("=" * 70)
    print()
    
    # Test Phase 4
    print("Testing Phase 4...")
    setup_env()
    from build_phase3_variant import build_phase3_variant
    result4 = test_kernel("fa_phase3", build_phase3_variant)
    if result4[0] is False:
        print(f"❌ Phase 4: FAILED to build/import")
        return
    passed4, time4, sdpa4, diff4 = result4
    
    if passed4:
        print(f"✅ Phase 4: {time4:.2f} μs (correctness OK, max_diff={diff4:.6f})")
    else:
        print(f"❌ Phase 4: Correctness FAILED (max_diff={diff4:.6f})")
        return
    
    # Test Phase 6
    print("\nTesting Phase 6...")
    from build_phase6 import build_phase6
    result6 = test_kernel("fa_phase6", build_phase6)
    if result6[0] is False:
        print(f"❌ Phase 6: FAILED to build/import")
        return
    passed6, time6, sdpa6, diff6 = result6
    
    if passed6:
        print(f"✅ Phase 6: {time6:.2f} μs (correctness OK, max_diff={diff6:.6f})")
    else:
        print(f"❌ Phase 6: Correctness FAILED (max_diff={diff6:.6f})")
        return
    
    # Comparison
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"PyTorch SDPA:  {sdpa4:8.2f} μs (baseline)")
    print(f"Phase 4:       {time4:8.2f} μs ({sdpa4/time4:.2f}× vs SDPA)")
    print(f"Phase 6:       {time6:8.2f} μs ({sdpa6/time6:.2f}× vs SDPA)")
    print()
    
    if time6 < time4:
        improvement = time4 / time6
        print(f"✅ Phase 6 is {improvement:.2f}× FASTER than Phase 4")
    else:
        regression = time6 / time4
        print(f"❌ Phase 6 is {regression:.2f}× SLOWER than Phase 4")
    
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

