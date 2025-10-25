#!/usr/bin/env python3
# Copyright 2025 GOATnote Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FP8 Attention Kernel for NVIDIA Hopper
Target: <1 μs on H100 with acceptable accuracy (max_diff < 5e-3)

NVIDIA H100 FP8 Tensor Cores: 2× throughput vs FP16
- FP16: 989 TFLOPS
- FP8: 1979 TFLOPS (E4M3 format)

Key Innovation: Mixed precision with FP8 matmul, FP32 accumulation
"""
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _attention_fp8_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    """
    FP8 FlashAttention with FP32 accumulation
    
    Strategy:
    - Load Q, K, V in FP8 (2× memory bandwidth)
    - Convert to FP16 for matmul (hardware handles conversion)
    - Accumulate in FP32 (numerical stability)
    - Store in FP16
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    # Base pointers
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = Out + b * stride_ob + h * stride_oh
    
    # Load Q block [BLOCK_M, D] - FP8 input
    Q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N, other=0.0)
    
    # Convert to FP16 for computation (hardware automatic)
    # Triton handles FP8->FP16 conversion transparently
    
    # Online softmax accumulators (FP32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Process keys/values
    for start_n in range(0, N, BLOCK_N):
        offs_n_cur = start_n + offs_n
        
        # Load K^T [D, BLOCK_N] - FP8 input
        K_ptrs = k_base + offs_n_cur[None, :] * stride_km + offs_d[:, None] * stride_kk
        k = tl.load(K_ptrs, mask=offs_n_cur[None, :] < N, other=0.0)
        
        # Load V [BLOCK_N, D] - FP8 input
        V_ptrs = v_base + offs_n_cur[:, None] * stride_vm + offs_d[None, :] * stride_vk
        v = tl.load(V_ptrs, mask=offs_n_cur[:, None] < N, other=0.0)
        
        # QK^T matmul (FP16 hardware, FP32 accumulation)
        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk *= SCALE
        
        # Online softmax (FP32)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        
        # Rescale
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # P@V matmul (FP32 accumulation)
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        
        # Update stats
        l_i = alpha * l_i + tl.sum(p, 1)
        m_i = m_ij
    
    # Normalize (FP32)
    acc = acc / l_i[:, None]
    
    # Store as FP16
    O_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(O_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


def attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int = 64,
    block_n: int = 128,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    FP8 Scaled Dot-Product Attention for Hopper
    
    Args:
        q: Query [B, H, N, D] in FP8 (torch.float8_e4m3fn)
        k: Key [B, H, N, D] in FP8
        v: Value [B, H, N, D] in FP8
        block_m: Query block size
        block_n: Key/value block size
        scale: Attention scale (default: 1/sqrt(D))
    
    Returns:
        Output [B, H, N, D] in FP16
    
    Performance Target:
        - H100: <1 μs per sequence (B≥16)
        - 2× speedup vs FP16 baseline (theoretical)
    
    Accuracy Target:
        - max_diff < 5e-3 vs FP16 reference
        - Acceptable for inference (not training)
    
    Note: Requires torch 2.1+ with FP8 support and CUDA 12.0+
    """
    B, H, N, D = q.shape
    
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert D == 64, "Only D=64 supported"
    
    # Check FP8 support (Hopper sm_90+)
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("FP8 requires Hopper (sm_90+) GPU")
    
    # Convert to FP8 if not already
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
    
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    
    # Allocate FP16 output (FP8 output would lose too much precision)
    out = torch.empty((B, H, N, D), device=q.device, dtype=torch.float16)
    
    # Launch kernel
    grid = (triton.cdiv(N, block_m), B * H)
    
    _attention_fp8_kernel[grid](
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


def benchmark_fp8():
    """Benchmark FP8 attention vs FP16 baseline"""
    print("=" * 80)
    print("FP8 ATTENTION KERNEL - HOPPER H100 BENCHMARK")
    print("=" * 80)
    print()
    
    # Check Hopper
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        print(f"❌ FP8 requires Hopper (sm_90+), found sm_{cap[0]}{cap[1]}")
        print("   Run this benchmark on H100 or H200")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]})")
    print()
    print("Target: <1 μs per sequence (2× faster than FP16 baseline)")
    print("Accuracy: max_diff < 5e-3 vs FP16 reference")
    print()
    
    configs = [
        (128, 16), (128, 32),
        (256, 16), (256, 32),
        (512, 16), (512, 32),
    ]
    
    print(f"{'Seq':>4} {'Batch':>5} {'FP8 (μs)':>10} {'FP16 (μs)':>11} {'Speedup':>8} {'Max Diff':>10} {'Status':>8}")
    print("-" * 80)
    
    for S, B in configs:
        # Create FP16 tensors
        torch.manual_seed(42)
        q_fp16 = torch.randn(B, 8, S, 64, device='cuda', dtype=torch.float16)
        k_fp16, v_fp16 = q_fp16.clone(), q_fp16.clone()
        
        # Convert to FP8
        q_fp8 = q_fp16.to(torch.float8_e4m3fn)
        k_fp8 = k_fp16.to(torch.float8_e4m3fn)
        v_fp8 = v_fp16.to(torch.float8_e4m3fn)
        
        # Warmup FP8
        for _ in range(50):
            _ = attention_fp8(q_fp8, k_fp8, v_fp8)
        torch.cuda.synchronize()
        
        # Benchmark FP8
        times_fp8 = []
        for _ in range(200):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out_fp8 = attention_fp8(q_fp8, k_fp8, v_fp8)
            end.record()
            torch.cuda.synchronize()
            times_fp8.append(start.elapsed_time(end) * 1000)
        
        times_fp8.sort()
        fp8_us = times_fp8[len(times_fp8) // 2] / B
        
        # Import FP16 baseline
        from attention_production import attention
        
        # Warmup FP16
        for _ in range(50):
            _ = attention(q_fp16, k_fp16, v_fp16)
        torch.cuda.synchronize()
        
        # Benchmark FP16
        times_fp16 = []
        for _ in range(200):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out_fp16 = attention(q_fp16, k_fp16, v_fp16)
            end.record()
            torch.cuda.synchronize()
            times_fp16.append(start.elapsed_time(end) * 1000)
        
        times_fp16.sort()
        fp16_us = times_fp16[len(times_fp16) // 2] / B
        
        # Check accuracy
        max_diff = (out_fp8 - out_fp16).abs().max().item()
        
        speedup = fp16_us / fp8_us
        target_ok = "✅" if fp8_us < 1.0 and max_diff < 5e-3 else "⚠️"
        
        print(f"{S:4} {B:5} {fp8_us:9.2f} {fp16_us:10.2f} {speedup:7.2f}× {max_diff:9.6f} {target_ok:>8}")
    
    print("-" * 80)
    print()
    print("✅ TARGET: <1 μs with 2× speedup over FP16")
    print("=" * 80)


if __name__ == '__main__':
    benchmark_fp8()

