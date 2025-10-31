"""
BlackwellSparseK: High-Performance Sparse Attention Kernel
Optimized for NVIDIA H100 (sm_90a) and B200 (sm_100)

Architecture: FlashAttention-2 + WMMA Tensor Cores + CuTe DSL
Author: BlackwellSparseK Core Team
License: MIT with Ethical Use Clause
Citations: SparseK (arXiv:2406.16747), NVIDIA CUTLASS, Meta xFormers, vLLM (Berkeley)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# Try to import compiled CUDA extension (will be built via setup.py)
try:
    from blackwell_sparsek_cuda import (
        attention_forward_cuda,
        attention_backward_cuda
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "BlackwellSparseK CUDA kernels not compiled. "
        "Falling back to PyTorch SDPA. "
        "Run 'pip install -e .' to build CUDA extension.",
        RuntimeWarning
    )


class SparseKAttentionFunction(torch.autograd.Function):
    """
    Custom autograd function for SparseK attention.
    
    Forward: Q@K^T -> softmax -> P@V using FlashAttention-2 tiling
    Backward: Gradients computed via recomputation (memory-efficient)
    
    Optimizations:
    - WMMA Tensor Cores (16x16x16 tiles, FP16)
    - CuTe TMA async copy (Hopper/Blackwell)
    - Online softmax (no materialized attention matrix)
    - Shared memory tiling (Br=32, Bc=64)
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # [B, H, S, D]
        k: torch.Tensor,  # [B, H, S, D]
        v: torch.Tensor,  # [B, H, S, D]
        causal: bool = False,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass: Compute attention output.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            causal: Apply causal masking (autoregressive)
            scale: Attention scale (default: 1/sqrt(head_dim))
        
        Returns:
            out: Attention output [batch, heads, seq_len, head_dim]
        """
        B, H, S, D = q.shape
        
        # Validate inputs
        assert q.is_cuda and k.is_cuda and v.is_cuda, "All inputs must be on CUDA"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only FP16/BF16 supported"
        assert k.shape == v.shape == (B, H, S, D), "Q, K, V must have same shape"
        
        # Default scale: 1/sqrt(D)
        if scale is None:
            scale = 1.0 / (D ** 0.5)
        
        # Route to CUDA kernel if available
        if CUDA_AVAILABLE:
            out = attention_forward_cuda(q, k, v, scale, causal)
            
            # Save for backward
            ctx.save_for_backward(q, k, v, out)
            ctx.scale = scale
            ctx.causal = causal
            
            return out
        else:
            # Fallback to PyTorch SDPA (baseline)
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=scale
            )
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass: Compute gradients for Q, K, V.
        
        Uses recomputation strategy (memory-efficient):
        - Recompute attention weights from Q, K
        - Compute dV = P^T @ dO
        - Compute dP = dO @ V^T
        - Compute dS = dP * P (elementwise)
        - Compute dQ = dS @ K, dK = dS^T @ Q
        """
        q, k, v, out = ctx.saved_tensors
        scale = ctx.scale
        causal = ctx.causal
        
        if CUDA_AVAILABLE:
            grad_q, grad_k, grad_v = attention_backward_cuda(
                grad_out, q, k, v, out, scale, causal
            )
            return grad_q, grad_k, grad_v, None, None
        else:
            # Fallback to PyTorch autograd
            # (This path won't be optimal, but ensures correctness)
            q_req = q.requires_grad_(True)
            k_req = k.requires_grad_(True)
            v_req = v.requires_grad_(True)
            
            out_ref = F.scaled_dot_product_attention(
                q_req, k_req, v_req,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=scale
            )
            
            grad_q, grad_k, grad_v = torch.autograd.grad(
                out_ref, [q_req, k_req, v_req],
                grad_out,
                retain_graph=False
            )
            
            return grad_q, grad_k, grad_v, None, None


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    BlackwellSparseK attention forward pass.
    
    **Performance Targets** (H100, H=96, S=512, D=64, FP16):
    - Tier 1 (Match): ≤3.820 μs/head (parity with PyTorch SDPA)
    - Tier 2 (Exceed): <3.0 μs/head (25% faster, competitive with FA3)
    - Tier 3 (Push): <2.0 μs/head (50% faster, state-of-the-art)
    
    **Optimizations**:
    - FlashAttention-2 tiling (Br=32, Bc=64)
    - WMMA Tensor Cores (16x16x16, FP16)
    - CuTe TMA async (Hopper/Blackwell)
    - Warp specialization (producer/consumer)
    - Online softmax (FP32 accumulators)
    
    **Citations**:
    - SparseK: Sun et al., arXiv:2406.16747
    - FlashAttention: Dao et al., arXiv:2205.14135, arXiv:2307.08691
    - CUTLASS: NVIDIA, BSD 3-Clause License
    
    Args:
        q: Query [B, H, S, D], FP16/BF16, CUDA
        k: Key [B, H, S, D], FP16/BF16, CUDA
        v: Value [B, H, S, D], FP16/BF16, CUDA
        causal: Causal masking (autoregressive)
        scale: Attention scale (default: 1/sqrt(D))
    
    Returns:
        out: Attention output [B, H, S, D]
    
    Example:
        >>> import torch
        >>> q = torch.randn(16, 96, 512, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(16, 96, 512, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(16, 96, 512, 64, device='cuda', dtype=torch.float16)
        >>> out = attention_forward(q, k, v, causal=True)
        >>> print(out.shape)  # [16, 96, 512, 64]
    """
    return SparseKAttentionFunction.apply(q, k, v, causal, scale)


def attention_forward_with_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Attention forward with performance statistics.
    
    Returns:
        out: Attention output [B, H, S, D]
        stats: Dict with timing and memory info
    """
    # Warm up
    _ = attention_forward(q, k, v, causal, scale)
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    out = attention_forward(q, k, v, causal, scale)
    end.record()
    torch.cuda.synchronize()
    
    latency_ms = start.elapsed_time(end)
    B, H, S, D = q.shape
    latency_per_head_us = (latency_ms * 1000) / H
    
    # Memory usage
    memory_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    memory_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    
    stats = {
        'latency_ms': latency_ms,
        'latency_per_head_us': latency_per_head_us,
        'memory_allocated_mb': memory_allocated_mb,
        'memory_reserved_mb': memory_reserved_mb,
        'batch_size': B,
        'num_heads': H,
        'seq_len': S,
        'head_dim': D,
        'cuda_available': CUDA_AVAILABLE
    }
    
    return out, stats


# Export public API
__all__ = [
    'attention_forward',
    'attention_forward_with_stats',
    'SparseKAttentionFunction',
    'CUDA_AVAILABLE'
]

