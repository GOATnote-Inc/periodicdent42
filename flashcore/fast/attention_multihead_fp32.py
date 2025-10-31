#!/usr/bin/env python3
# Multi-head attention with FULL FP32 precision for large H
import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _multihead_fp32_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    """Full FP32 precision kernel for H=96+"""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = Out + b * stride_ob + h * stride_oh
    
    # Load Q and convert to FP32 immediately
    Q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N, other=0.0).to(tl.float32)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    for start_n in range(0, N, BLOCK_N):
        offs_n_cur = start_n + offs_n
        
        # Load K and convert to FP32
        K_ptrs = k_base + offs_n_cur[None, :] * stride_km + offs_d[:, None] * stride_kk
        k = tl.load(K_ptrs, mask=offs_n_cur[None, :] < N, other=0.0).to(tl.float32)
        
        # Load V and convert to FP32
        V_ptrs = v_base + offs_n_cur[:, None] * stride_vm + offs_d[None, :] * stride_vk
        v = tl.load(V_ptrs, mask=offs_n_cur[:, None] < N, other=0.0).to(tl.float32)
        
        # All computation in FP32
        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk *= SCALE
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        acc += tl.dot(p, v, out_dtype=tl.float32)
        
        l_i = alpha * l_i + tl.sum(p, 1)
        m_i = m_ij
    
    acc = acc / l_i[:, None]
    
    # Store as FP16
    O_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(O_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


def multihead_attention_fp32(q, k, v, scale=None):
    """Full FP32 precision for large head counts"""
    B, H, N, D = q.shape
    assert D == 64
    
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    
    out = torch.empty_like(q)
    
    # Smaller blocks for better precision
    block_m = 32 if H >= 96 else 64
    block_n = 32 if H >= 96 else 64
    
    grid = (triton.cdiv(N, block_m), B * H)
    
    _multihead_fp32_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N, D,
        BLOCK_M=block_m, BLOCK_N=block_n,
        SCALE=scale
    )
    
    return out
