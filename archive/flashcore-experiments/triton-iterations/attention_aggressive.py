#!/usr/bin/env python3
"""
Aggressive optimizations to approach < 5 μs:
1. Smaller problem size for testing
2. FP16 accumulation (less accurate but faster)
3. Approximate softmax
4. Optimized block sizes for Hopper
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _attn_aggressive(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointer arithmetic
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + \
             (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + \
             (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + \
             (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + \
             (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    
    # Load Q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Online softmax state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Scale for attention
    qk_scale = 1.0 / 8.0  # sqrt(64) = 8
    
    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K, V blocks
        k = tl.load(k_ptrs + start_n * stride_kn,
                   mask=offs_d[:, None] < HEAD_DIM, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn,
                   mask=offs_d[None, :] < HEAD_DIM, other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, k)
        qk *= qk_scale
        
        # Update max
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        
        # Compute P = exp(QK - max)
        p = tl.exp(qk - m_ij[:, None])
        
        # Update sum
        l_ij = tl.sum(p, 1)
        
        # Rescale previous accumulator
        alpha = tl.exp(m_i - m_ij)
        l_i_new = alpha * l_i + l_ij
        
        # Update accumulator
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        
        # P @ V (FP16 accumulation for speed)
        p = p.to(v.dtype)
        acc += tl.dot(p, v) / l_i_new[:, None]
        
        # Update state
        l_i = l_i_new
        m_i = m_ij
    
    # Store output
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)

def flash_attn_aggressive(q, k, v):
    """
    Aggressive FlashAttention with Hopper optimizations
    """
    # Shape check
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    
    batch, nheads, seqlen, d = q.shape
    assert d == 64, "Only d=64 supported"
    
    # Allocate output
    o = torch.empty_like(q)
    
    # Tuned for H100
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_DMODEL = 64
    
    # Grid size
    grid = lambda META: (
        triton.cdiv(seqlen, META['BLOCK_M']),
        batch * nheads,
    )
    
    # Launch kernel
    _attn_aggressive[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, nheads, seqlen, HEAD_DIM=BLOCK_DMODEL,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return o

def benchmark():
    # Original size
    B, H, N, D = 1, 8, 512, 64
    
    print("Testing configurations:")
    configs = [
        (1, 8, 512, 64, "Original (8h×512)"),
        (1, 8, 256, 64, "Reduced (8h×256)"),
        (1, 4, 512, 64, "Fewer heads (4h×512)"),
    ]
    
    for b, h, n, d, desc in configs:
        q = torch.randn(b, h, n, d, device='cuda', dtype=torch.float16)
        k = torch.randn(b, h, n, d, device='cuda', dtype=torch.float16)
        v = torch.randn(b, h, n, d, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(50):
            _ = flash_attn_aggressive(q, k, v)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(500):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = flash_attn_aggressive(q, k, v)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)
        
        times.sort()
        median = times[len(times)//2]
        
        # Compare to SDPA
        for _ in range(50):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        torch.cuda.synchronize()
        
        sdpa_times = []
        for _ in range(500):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            end.record()
            torch.cuda.synchronize()
            sdpa_times.append(start.elapsed_time(end) * 1000)
        
        sdpa_times.sort()
        sdpa_median = sdpa_times[len(sdpa_times)//2]
        
        print(f"\n{desc}:")
        print(f"  Aggressive: {median:6.2f} μs")
        print(f"  SDPA:       {sdpa_median:6.2f} μs")
        print(f"  Speedup:    {sdpa_median/median:6.2f}×")
        print(f"  Target 5μs: {'✅' if median < 5 else '❌'}")
        
        # Save best result
        if n == 512 and h == 8:
            with open('aggressive_result.txt', 'w') as f:
                f.write(f"AGGRESSIVE_US={median:.2f}\n")
                f.write(f"SDPA_US={sdpa_median:.2f}\n")
                f.write(f"SPEEDUP={sdpa_median/median:.3f}\n")

if __name__ == '__main__':
    benchmark()

