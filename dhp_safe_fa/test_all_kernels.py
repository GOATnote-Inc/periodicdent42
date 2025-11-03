#!/usr/bin/env python3
"""
Comprehensive DHP Kernel Test
==============================
Tests I4, I5, I7, I8 for correctness and performance
"""

import torch
import torch.nn.functional as F

# Enable deterministic mode
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
print("DHP Kernel Progression Test")
print("="*80)
print()

# PyTorch baseline
_, pt_ms, pt_us = bench("PyTorch", lambda: F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=True).reshape(B*H, S, D))
print(f"PyTorch SDPA (baseline): {pt_ms:.3f} ms ({pt_us:.2f} μs/head)")
print()

# Precompute scores for I4/I5
scores = torch.matmul(Q, K.transpose(-2, -1)) * 0.125

kernels = [
    ("I4", "dhp_i4_kernel", lambda k: lambda: k.forward(scores, V, S, S)),
    ("I5", "dhp_i5_kernel", lambda k: lambda: k.forward(scores, V, S, S)),
    ("I7", "dhp_i7_kernel", lambda k: lambda: k.forward(Q, K, V, S, S)),
    ("I8", "dhp_i8_kernel", lambda k: lambda: k.forward(Q, K, V, S, S)),
    ("I9", "dhp_i9_kernel", lambda k: lambda: k.forward(Q, K, V, S, S)),
]

results = []
for name, module_name, fn_builder in kernels:
    try:
        module = __import__(module_name)
        fn = fn_builder(module)
        
        # Correctness
        out = fn()
        max_diff = torch.abs(out - out_ref).max().item()
        mean_diff = torch.abs(out - out_ref).mean().item()
        correct = max_diff < 2e-3
        
        # Performance
        _, ms, us = bench(name, fn)
        
        # Reproducibility
        out2 = fn()
        reproducible = torch.equal(out, out2)
        
        results.append({
            'name': name,
            'ms': ms,
            'us': us,
            'slowdown': ms/pt_ms,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'correct': correct,
            'reproducible': reproducible
        })
        
        status = '✅' if correct else '❌'
        repro = '✅' if reproducible else '⚠️'
        print(f"{name:3} {status} {ms:6.3f}ms ({us:6.2f}μs/head) | {ms/pt_ms:4.1f}x | diff={max_diff:.6f} | repro={repro}")
        
    except Exception as e:
        print(f"{name:3} ❌ Failed: {e}")

print()
print("="*80)
print("Summary")
print("="*80)
print()

if results:
    best = min(results, key=lambda x: x['ms'])
    print(f"Best Performance: {best['name']} @ {best['us']:.2f} μs/head ({best['slowdown']:.1f}x slower than PyTorch)")
    
    correct = [r for r in results if r['correct']]
    if correct:
        best_correct = min(correct, key=lambda x: x['ms'])
        print(f"Best Correct:     {best_correct['name']} @ {best_correct['us']:.2f} μs/head ({best_correct['slowdown']:.1f}x slower)")
    
    repro = [r for r in results if r['reproducible']]
    if repro:
        print(f"Reproducible:     {', '.join([r['name'] for r in repro])}")

