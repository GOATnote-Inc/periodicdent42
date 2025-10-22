#!/usr/bin/env python3
"""Test FlashCore WMMA kernels for correctness and performance."""

import torch
import torch.nn.functional as F
from build_wmma import build_extension


def test_qkt_correctness():
    """Test QK^T kernel correctness."""
    print("\n=== QK^T Correctness Test ===")
    
    B, H, S, D = 1, 8, 512, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    scale = 1.0 / (D ** 0.5)
    
    # Reference (PyTorch)
    ref = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Build and run kernel
    module = build_extension(verbose=False)
    out = module.qkt(q, k, scale)
    
    # Compare
    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()
    
    print(f"Shape: Q {list(q.shape)}, K {list(k.shape)}")
    print(f"Output shape: {list(out.shape)}")
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    
    if max_err < 0.05:
        print("✅ QK^T: PASS")
        return True
    else:
        print(f"❌ QK^T: FAIL (error {max_err:.6f} >= 0.05)")
        return False


def test_pv_correctness():
    """Test P·V kernel correctness."""
    print("\n=== P·V Correctness Test ===")
    
    B, H, S, D = 1, 8, 512, 64
    p = torch.randn(B, H, S, S, device="cuda", dtype=torch.float16)
    p = F.softmax(p.float(), dim=-1).to(torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    
    # Reference (PyTorch)
    ref = torch.matmul(p, v)
    
    # Build and run kernel
    module = build_extension(verbose=False)
    out = module.pv(p, v)
    
    # Compare
    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()
    
    print(f"Shape: P {list(p.shape)}, V {list(v.shape)}")
    print(f"Output shape: {list(out.shape)}")
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    
    if max_err < 0.05:
        print("✅ P·V: PASS")
        return True
    else:
        print(f"❌ P·V: FAIL (error {max_err:.6f} >= 0.05)")
        return False


def benchmark_kernels():
    """Benchmark both kernels."""
    print("\n=== Performance Benchmark ===")
    
    B, H, S, D = 1, 8, 512, 64
    iters = 200
    warmup = 20
    
    # Prepare inputs
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    scale = 1.0 / (D ** 0.5)
    
    p = torch.randn(B, H, S, S, device="cuda", dtype=torch.float16)
    p = F.softmax(p.float(), dim=-1).to(torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    
    module = build_extension(verbose=False)
    
    # Benchmark QK^T
    for _ in range(warmup):
        module.qkt(q, k, scale)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        module.qkt(q, k, scale)
    end.record()
    torch.cuda.synchronize()
    qkt_us = (start.elapsed_time(end) / iters) * 1000.0
    
    # Benchmark P·V
    for _ in range(warmup):
        module.pv(p, v)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(iters):
        module.pv(p, v)
    end.record()
    torch.cuda.synchronize()
    pv_us = (start.elapsed_time(end) / iters) * 1000.0
    
    print(f"Configuration: B={B}, H={H}, S={S}, D={D}")
    print(f"[QK^T] Average latency: {qkt_us:.2f} µs")
    print(f"[P·V]  Average latency: {pv_us:.2f} µs")
    print(f"[Total] Estimated: {qkt_us + pv_us:.2f} µs")
    
    # Benchmark PyTorch SDPA for comparison
    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(iters):
        F.scaled_dot_product_attention(q, k, v, scale=scale)
    end.record()
    torch.cuda.synchronize()
    sdpa_us = (start.elapsed_time(end) / iters) * 1000.0
    
    print(f"\n[PyTorch SDPA] {sdpa_us:.2f} µs")
    print(f"[Speedup vs SDPA] {sdpa_us / (qkt_us + pv_us):.2f}×")
    
    if (qkt_us + pv_us) < 40.0:
        print("\n🎉 TARGET ACHIEVED: <40 µs! 🎉")
    elif (qkt_us + pv_us) < sdpa_us:
        print(f"\n✅ Faster than PyTorch SDPA by {sdpa_us / (qkt_us + pv_us):.2f}×")
    else:
        gap = (qkt_us + pv_us) / sdpa_us
        print(f"\n📊 Current: {gap:.2f}× slower than SDPA (target: <40 µs)")


if __name__ == "__main__":
    print("=" * 60)
    print("FlashCore WMMA Kernel Test Suite")
    print("=" * 60)
    
    qkt_pass = test_qkt_correctness()
    pv_pass = test_pv_correctness()
    
    if qkt_pass and pv_pass:
        print("\n" + "=" * 60)
        print("✅ All correctness tests PASSED")
        print("=" * 60)
        benchmark_kernels()
    else:
        print("\n" + "=" * 60)
        print("❌ Correctness tests FAILED - fix errors before benchmarking")
        print("=" * 60)

