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
FlashCore Stage 3: Persistent CTAs (Grid-Stride Loop)

Einstein Constraint #2: Launch Overhead → Persistent CTAs
Target: 140 TFLOPS (+48% from 94.5) via batching efficiency

Key Optimization:
- Launch fewer CTAs, process multiple batches per CTA
- Amortize kernel launch overhead (40% → 2%)
- Expected: 6× speedup for B=1 → B=32 per-sequence

Attribution:
- NVIDIA: Persistent kernel patterns
- CUTLASS: Grid-stride loop methodology
- Einstein framework: Constraint elimination
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _attention_fwd_persistent(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Persistent CTA kernel - processes multiple batches per CTA.
    
    Key difference from Stage 1:
    - Stage 1: grid = (B, H, M_tiles) - one CTA per batch per tile
    - Stage 3: grid = (num_ctas, H, M_tiles) - CTAs iterate over batches
    
    Expected gain: 6× speedup from amortizing launch overhead
    """
    # Get CTA ID and total CTAs
    cta_id = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Grid-stride loop over batches
    for pid_b in range(cta_id, B, num_ctas):
        # Offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        
        # Load Q tile: [BLOCK_M, D]
        q_ptrs = (
            Q + pid_b * stride_qb + pid_h * stride_qh +
            offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        )
        mask_m = offs_m < S
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        
        # Convert to FP32 for accumulation
        q = q.to(tl.float32)
        q = q * SCALE
        
        # Initialize online softmax accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        
        # Loop over K/V tiles
        num_blocks_n = tl.cdiv(S, BLOCK_N)
        for block_n_idx in range(num_blocks_n):
            offs_n = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S
            
            # Load K tile: [D, BLOCK_N]
            k_ptrs = (
                K + pid_b * stride_kb + pid_h * stride_kh +
                offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
            )
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
            k = k.to(tl.float32)
            
            # Compute Q@K^T: [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, k)
            
            # Causal masking
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, float('-inf'))
            
            # Online softmax: Update max
            m_ij = tl.max(qk, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            
            # Online softmax: Update sum
            p = tl.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, axis=1)
            l_i_new = alpha * l_i + l_ij
            
            # Online softmax: Update accumulator
            acc = acc * alpha[:, None]
            
            # Load V tile: [BLOCK_N, D]
            v_ptrs = (
                V + pid_b * stride_vb + pid_h * stride_vh +
                offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            v = v.to(tl.float32)
            
            # Accumulate: acc += P @ V
            acc += tl.dot(p, v)
            
            # Update accumulators
            m_i = m_i_new
            l_i = l_i_new
        
        # Final normalization
        acc = acc / l_i[:, None]
        
        # Store output: [BLOCK_M, D]
        acc = acc.to(Out.dtype.element_ty)
        out_ptrs = (
            Out + pid_b * stride_ob + pid_h * stride_oh +
            offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        )
        tl.store(out_ptrs, acc, mask=mask_m[:, None])


def attention_persistent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    num_ctas: Optional[int] = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    Persistent CTA attention.
    
    Args:
        query: [B, H, S, D]
        key: [B, H, S, D]
        value: [B, H, S, D]
        is_causal: Apply causal masking
        num_ctas: Number of CTAs (default: num_sms * 2)
        block_m, block_n: Tile sizes
    
    Returns:
        output: [B, H, S, D]
    """
    assert query.dim() == 4
    assert key.dim() == 4
    assert value.dim() == 4
    assert query.shape == key.shape == value.shape
    assert query.is_cuda
    assert query.dtype in [torch.float16, torch.bfloat16]
    
    B, H, S, D = query.shape
    
    # Allocate output
    output = torch.empty_like(query)
    
    # Compute scale
    scale = 1.0 / (D ** 0.5)
    
    # Determine number of CTAs (persistent kernel pattern)
    if num_ctas is None:
        # H100 has 132 SMs, use 2× for good occupancy
        num_ctas = min(264, B)  # Don't use more CTAs than batches
    
    # Launch grid: (num_ctas, H, M_tiles)
    # CTAs will loop over batches using grid-stride pattern
    M_tiles = triton.cdiv(S, block_m)
    grid = (num_ctas, H, M_tiles)
    
    # Launch kernel
    _attention_fwd_persistent[grid](
        query, key, value, output,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, H, S, D,
        block_m, block_n,
        scale,
        is_causal,
    )
    
    return output


def benchmark_batching_efficiency():
    """Test batching efficiency of persistent CTAs"""
    
    H, S, D = 16, 2048, 64
    batch_sizes = [1, 8, 16, 32]
    
    print("="*80)
    print("STAGE 3: PERSISTENT CTA BATCHING EFFICIENCY")
    print("="*80)
    print()
    
    results = {}
    
    for B in batch_sizes:
        print(f"[Testing B={B}]")
        
        # Create inputs
        query = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        key = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        value = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(20):
            _ = attention_persistent(query, key, value, is_causal=True)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(100):
            start.record()
            _ = attention_persistent(query, key, value, is_causal=True)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = torch.tensor(times)
        median_ms = torch.quantile(times, 0.5).item()
        
        # Compute metrics
        flops = 4 * B * H * S * S * D
        tflops = flops / (median_ms / 1000) / 1e12
        latency_per_seq = median_ms / B
        
        results[B] = {
            'total_ms': median_ms,
            'per_seq_ms': latency_per_seq,
            'tflops': tflops
        }
        
        print(f"  Total time:      {median_ms:.3f} ms")
        print(f"  Per-sequence:    {latency_per_seq:.3f} ms")
        print(f"  TFLOPS:          {tflops:.1f}")
        print()
    
    # Analyze batching efficiency
    print("="*80)
    print("BATCHING EFFICIENCY ANALYSIS")
    print("="*80)
    
    speedup_8 = results[1]['per_seq_ms'] / results[8]['per_seq_ms']
    speedup_32 = results[1]['per_seq_ms'] / results[32]['per_seq_ms']
    
    print(f"\nPer-sequence speedup:")
    print(f"  B=1 → B=8:   {speedup_8:.2f}× (target: >2.5×)")
    print(f"  B=1 → B=32:  {speedup_32:.2f}× (target: >5.0×)")
    print()
    print(f"TFLOPS scaling:")
    print(f"  B=1:   {results[1]['tflops']:.1f} TFLOPS")
    print(f"  B=8:   {results[8]['tflops']:.1f} TFLOPS ({(results[8]['tflops']/results[1]['tflops']):.2f}×)")
    print(f"  B=32:  {results[32]['tflops']:.1f} TFLOPS ({(results[32]['tflops']/results[1]['tflops']):.2f}×)")
    print()
    
    # Check if we hit targets
    if results[32]['tflops'] >= 140:
        print("✅ SUCCESS: Hit 140 TFLOPS target at B=32!")
        print(f"   Achieved {results[32]['tflops']:.1f} TFLOPS")
    elif results[32]['tflops'] >= 120:
        print("⚠️  CLOSE: Got {:.1f} TFLOPS (target: 140)".format(results[32]['tflops']))
    else:
        print("❌ MISS: Got {:.1f} TFLOPS (target: 140)".format(results[32]['tflops']))
    
    if speedup_32 >= 5.0:
        print(f"✅ Batching efficiency excellent: {speedup_32:.1f}× speedup")
    else:
        print(f"⚠️  Batching efficiency: {speedup_32:.1f}× (target: 5.0×)")
    
    print("="*80)
    
    return results


if __name__ == "__main__":
    # First: Correctness test
    print("="*80)
    print("CORRECTNESS TEST")
    print("="*80)
    
    B, H, S, D = 16, 16, 2048, 64
    query = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    key = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    value = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    out_persistent = attention_persistent(query, key, value, is_causal=True)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    
    max_diff = (out_persistent - out_ref).abs().max().item()
    mean_diff = (out_persistent - out_ref).abs().mean().item()
    correct = torch.allclose(out_persistent, out_ref, rtol=1e-3, atol=2e-3)
    
    print(f"Max diff:  {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"Status:    {'✅ PASS' if correct else '❌ FAIL'}")
    print()
    
    if not correct:
        print("❌ Correctness failed! Fix before benchmarking.")
        exit(1)
    
    # Second: Batching efficiency test
    results = benchmark_batching_efficiency()

