#!/usr/bin/env python3
"""
FlashAttention implementation in Triton
Based on: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
Adapted for: L4 GPU (sm_89, Ada architecture)
Target: <26 μs (beat PyTorch SDPA's 44 μs)
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, HEAD_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """Triton FlashAttention forward kernel"""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Offset pointers for batch/head
    qo_offset = off_z * stride_qz + off_h * stride_qh
    kv_offset = off_z * stride_kz + off_h * stride_kh
    
    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=Out + qo_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize pointers to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q
    q = tl.load(Q_block_ptr)
    
    # Loop over K, V tiles
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        # Compute QK^T
        qk = tl.dot(q, k)
        qk = qk * sm_scale
        
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update acc with correction
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))


def flash_attention_triton(q, k, v):
    """
    Triton FlashAttention forward pass
    
    Args:
        q: [B, H, N, D] query
        k: [B, H, N, D] key
        v: [B, H, N, D] value
    
    Returns:
        o: [B, H, N, D] output
    """
    B, H, N, D = q.shape
    
    # Check constraints
    assert q.dtype == k.dtype == v.dtype, "All inputs must have same dtype"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "All inputs must be on CUDA"
    
    # Allocate output
    o = torch.empty_like(q)
    
    # Softmax scale
    sm_scale = 1.0 / (D ** 0.5)
    
    # Get strides
    stride_qz, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kz, stride_kh, stride_kn, stride_kk = k.stride()
    stride_vz, stride_vh, stride_vn, stride_vk = v.stride()
    stride_oz, stride_oh, stride_om, stride_ok = o.stride()
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), B * H)
    
    _fwd_kernel[grid](
        q, k, v, sm_scale, o,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        B, H, N, D,
        BLOCK_M=64, BLOCK_N=64, BLOCK_DMODEL=D,
    )
    
    return o


if __name__ == '__main__':
    # Test
    import statistics
    
    print("=" * 70)
    print("Triton FlashAttention Benchmark")
    print("=" * 70)
    
    B, H, N, D = 1, 8, 512, 64
    print(f"\nShape: B={B}, H={H}, N={N}, D={D}")
    
    q = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    
    # Warmup
    print("\n[1/3] Warmup (20 iters)...")
    for _ in range(20):
        o = flash_attention_triton(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    print("[2/3] Benchmarking (100 iters)...")
    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        o = flash_attention_triton(q, k, v)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    
    p50 = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8]
    
    # Correctness
    print("[3/3] Correctness check...")
    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    max_err = (o - o_ref).abs().max().item()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Correctness:")
    print(f"  Max error: {max_err:.6f}")
    print(f"  Status: {'✅ PASS' if max_err < 0.1 else '❌ FAIL'}")
    
    print(f"\nPerformance:")
    print(f"  p50: {p50:.2f} μs")
    print(f"  p90: {p90:.2f} μs")
    
    print(f"\nComparison:")
    pytorch_us = 44.10
    print(f"  PyTorch SDPA:     {pytorch_us:.2f} μs")
    print(f"  Triton FlashAttn: {p50:.2f} μs")
    speedup = pytorch_us / p50
    print(f"  Speedup:          {speedup:.2f}×")
    
    target = 26.0
    if p50 < target:
        print(f"\n✅ SUCCESS! Beats target ({p50:.2f} < {target} μs)")
    else:
        print(f"\n⚠️  Close: {p50:.2f} μs (target: <{target} μs)")

