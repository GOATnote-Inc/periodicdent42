#!/usr/bin/env python3
"""
I7 Deterministic Test
=====================
Tests correctness with deterministic flags enabled (per EXCELLENCE_AUDIT.md §4)
"""

import torch
import torch.nn.functional as F

# Enable deterministic mode (EXCELLENCE_AUDIT.md §4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True

B, H, S, D = 4, 16, 1024, 64

torch.manual_seed(42)
Q = torch.randn(B*H, S, D, dtype=torch.float16, device='cuda')
K = torch.randn(B*H, S, D, dtype=torch.float16, device='cuda')
V = torch.randn(B*H, S, D, dtype=torch.float16, device='cuda')

# Reference
Q_ref = Q.reshape(B, H, S, D)
K_ref = K.reshape(B, H, S, D)
V_ref = V.reshape(B, H, S, D)
out_ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=True).reshape(B*H, S, D)

def bench(name, fn, warmup=10, runs=100):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    
    ms = start.elapsed_time(end) / runs
    us_per_head = (ms * 1000) / H
    return out, ms, us_per_head

print("="*80)
print("I7 Deterministic Test (EXCELLENCE_AUDIT.md §1-4)")
print("="*80)
print()

# PyTorch baseline
_, pt_ms, pt_us = bench("PyTorch", lambda: F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=True).reshape(B*H, S, D))
print(f"PyTorch SDPA: {pt_ms:.3f} ms ({pt_us:.2f} μs/head)")

# I7
try:
    import dhp_i7_kernel
    
    # Correctness test
    print()
    print("Correctness Test:")
    out_i7 = dhp_i7_kernel.forward(Q, K, V, S, S)
    max_diff = torch.abs(out_i7 - out_ref).max().item()
    mean_diff = torch.abs(out_i7 - out_ref).mean().item()
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    passed = max_diff < 2e-3
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if passed:
        # Deterministic reproducibility test
        print()
        print("Deterministic Reproducibility Test:")
        out_i7_run2 = dhp_i7_kernel.forward(Q, K, V, S, S)
        bitwise_match = torch.equal(out_i7, out_i7_run2)
        print(f"  Bitwise reproducibility: {'✅ PASS' if bitwise_match else '❌ FAIL'}")
        
        # Performance
        print()
        print("Performance Test:")
        _, i7_ms, i7_us = bench("I7", lambda: dhp_i7_kernel.forward(Q, K, V, S, S))
        print(f"  I7:           {i7_ms:.3f} ms ({i7_us:.2f} μs/head)")
        print(f"  Slowdown:     {i7_ms/pt_ms:.1f}x")
        
        if not bitwise_match:
            print()
            print("⚠️  Warning: Non-deterministic despite deterministic flags")
            print("   Likely cause: FP16 associativity in reductions")
    else:
        print()
        print("❌ Correctness failed - skipping other tests")
        
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

