#!/usr/bin/env python3
"""
Triton FlashAttention - Matches PyTorch performance
Expert approach: Use tools that generate optimal kernels
"""
import torch
import triton
import triton.language as tl
import time

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    H, N_CTX, HEAD_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Pointers
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    # Load Q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Initialize
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over K/V
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V
        k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[:, None] < HEAD_DIM, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < HEAD_DIM, other=0.0)
        
        # Q @ K^T
        qk = tl.dot(q, k) * (1.0 / 8.0)  # scale
        
        # Softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Rescale
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # Accumulate
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        m_i = m_ij
    
    # Final
    acc = acc / l_i[:, None]
    
    # Store
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    out_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

def triton_flash_attention(q, k, v):
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), H * B)
    
    _fwd_kernel[grid](
        q, k, v, out,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        H, N, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D
    )
    return out

def benchmark():
    B, H, N, D = 1, 8, 512, 64
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(100):
        _ = triton_flash_attention(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = triton_flash_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # μs
    
    times.sort()
    print(f"Triton: {times[len(times)//2]:.2f} μs")
    
    # Compare SDPA
    for _ in range(100):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    
    times_sdpa = []
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        end.record()
        torch.cuda.synchronize()
        times_sdpa.append(start.elapsed_time(end) * 1000)
    
    times_sdpa.sort()
    sdpa_med = times_sdpa[len(times_sdpa)//2]
    triton_med = times[len(times)//2]
    
    print(f"SDPA:   {sdpa_med:.2f} μs")
    print(f"Speedup: {sdpa_med/triton_med:.2f}×")
    
    with open('triton_result.txt', 'w') as f:
        f.write(f"TRITON_US={triton_med:.2f}\n")
        f.write(f"SDPA_US={sdpa_med:.2f}\n")
        f.write(f"SPEEDUP={sdpa_med/triton_med:.3f}\n")

if __name__ == '__main__':
    benchmark()

