#!/usr/bin/env python3
import torch
import torch.nn.functional as F

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

print("Testing I4, I5, I7 (WMMA)...")
print()

# PyTorch
_, pt_ms, pt_us = bench("PyTorch", lambda: F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=True).reshape(B*H, S, D))
print(f"PyTorch SDPA: {pt_ms:.3f} ms ({pt_us:.2f} μs/head)")

# I4
try:
    import dhp_i4_kernel
    _, i4_ms, i4_us = bench("I4", lambda: dhp_i4_kernel.forward(torch.matmul(Q, K.transpose(-2, -1)) * 0.125, V, S, S))
    print(f"I4:           {i4_ms:.3f} ms ({i4_us:.2f} μs/head) - {i4_ms/pt_ms:.1f}x slower")
except: print("I4: Not available")

# I5
try:
    import dhp_i5_kernel
    _, i5_ms, i5_us = bench("I5", lambda: dhp_i5_kernel.forward(torch.matmul(Q, K.transpose(-2, -1)) * 0.125, V, S, S))
    print(f"I5:           {i5_ms:.3f} ms ({i5_us:.2f} μs/head) - {i5_ms/pt_ms:.1f}x slower")
except: print("I5: Not available")

# I7
try:
    import dhp_i7_kernel
    out_i7, i7_ms, i7_us = bench("I7", lambda: dhp_i7_kernel.forward(Q, K, V, S, S))
    print(f"I7 (WMMA):    {i7_ms:.3f} ms ({i7_us:.2f} μs/head) - {i7_ms/pt_ms:.1f}x slower")
    
    max_diff = torch.abs(out_i7 - out_ref).max().item()
    print(f"I7 max_diff:  {max_diff:.6f} {'✅' if max_diff < 2e-3 else '❌'}")
except Exception as e:
    print(f"I7: Failed - {e}")

