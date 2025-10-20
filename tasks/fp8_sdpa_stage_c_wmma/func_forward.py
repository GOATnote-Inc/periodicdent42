#!/usr/bin/env python3
"""
Forward functions for FP8 SDPA Stage-C WMMA kernel validation.

Provides:
  - forward_ref: PyTorch SDPA reference (FP16)
  - forward_kernel: Our CUDA kernel (FP8 quantized)
  - quantize_sim_fp8_per_head: FP8 simulation quantizer
"""

import torch
import math
from typing import Tuple

def quantize_sim_fp8_per_head(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate FP8 E4M3 quantization per head (symmetric, zero-centered).
    
    Args:
        tensor: [B, H, S, D] FP16 tensor
    
    Returns:
        quantized: [B, H, S, D] uint8 tensor (128 = zero, [1, 255] = [-448, 448])
        scales: [H] FP32 per-head scales
    """
    assert tensor.dim() == 4, "Expected [B, H, S, D] tensor"
    
    # FP8 E4M3 range: [-448, 448]
    fp8_max = 448.0
    
    # Per-head absolute max: [B, H, S, D] → [1, H, 1, 1]
    abs_max = tensor.abs().to(torch.float32).amax(dim=(0, 2, 3), keepdim=True)
    
    # Compute scales: if abs_max > 1e-6, use abs_max/fp8_max, else 1.0
    scales = torch.where(
        abs_max > 1e-6,
        abs_max / fp8_max,
        torch.ones_like(abs_max)
    ).to(torch.float32)  # [1, H, 1, 1]
    
    # Normalize and map to [0, 255]
    # normalized ∈ [-1, 1] → [0, 255] with 128 = 0
    denom = (scales * fp8_max).to(torch.float32)
    normalized = (tensor.to(torch.float32) / denom).clamp(-1.0, 1.0)
    quantized = ((normalized + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    
    # Return quantized tensor and per-head scales [H]
    # scales shape: [1, H, 1, 1] → squeeze to [H]
    return quantized, scales.squeeze()  # [H]

def forward_ref(
    Q: torch.Tensor,  # [B, H, S, D] FP16
    K: torch.Tensor,  # [B, H, S, D] FP16
    V: torch.Tensor,  # [B, H, S, D] FP16
    scale: float,
) -> torch.Tensor:
    """
    PyTorch SDPA reference implementation.
    
    Args:
        Q, K, V: [B, H, S, D] FP16 tensors
        scale: 1.0 / sqrt(D)
    
    Returns:
        O: [B, H, S, D] FP16 output
    """
    with torch.inference_mode():
        O = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
    return O

def forward_kernel(
    Q_q: torch.Tensor,  # [B, H, S, D] uint8
    K_q: torch.Tensor,  # [B, H, S, D] uint8
    V_q: torch.Tensor,  # [B, H, S, D] uint8
    Q_s: torch.Tensor,  # [H] FP32 scales
    K_s: torch.Tensor,  # [H] FP32 scales
    V_s: torch.Tensor,  # [H] FP32 scales
    scale: float,
    ext,  # Loaded CUDA extension
) -> torch.Tensor:
    """
    FP8 SDPA kernel forward pass.
    
    Args:
        Q_q, K_q, V_q: [B, H, S, D] uint8 quantized tensors
        Q_s, K_s, V_s: [H] FP32 per-head scales
        scale: 1.0 / sqrt(D)
        ext: Loaded torch.utils.cpp_extension module
    
    Returns:
        O: [B, H, S, D] FP16 output
    """
    return ext.forward(Q_q, K_q, V_q, Q_s, K_s, V_s, scale)

def validate_correctness(
    ref: torch.Tensor,
    out: torch.Tensor,
    atol: float = 0.05,
    rtol: float = 0.05,
    pct_bad_max: float = 1.0,
) -> dict:
    """
    Validate kernel output against reference.
    
    Gates (all must pass):
      1. max_abs_err ≤ atol (0.05)
      2. mean_abs_err ≤ 0.01
      3. %(|err| > atol) ≤ pct_bad_max (1.0%)
    
    Args:
        ref: Reference output [B, H, S, D]
        out: Kernel output [B, H, S, D]
        atol: Absolute tolerance threshold
        rtol: Relative tolerance (for reporting, not gated)
        pct_bad_max: Max percentage of elements with |err| > atol
    
    Returns:
        dict with metrics and 'pass' boolean
    """
    abs_diff = (out - ref).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    
    # Percentage of elements with |err| > atol
    bad_mask = abs_diff > atol
    pct_bad = bad_mask.float().mean().item() * 100.0
    
    # Relative error (for info only)
    rel_diff = abs_diff / (ref.abs() + 1e-6)
    max_rel_err = rel_diff.max().item()
    
    # Gates
    gate_1 = max_abs_err <= atol
    gate_2 = mean_abs_err <= 0.01
    gate_3 = pct_bad <= pct_bad_max
    
    passed = gate_1 and gate_2 and gate_3
    
    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "max_rel_err": max_rel_err,
        "pct_bad": pct_bad,
        "gates": {
            "max_abs_err_pass": gate_1,
            "mean_abs_err_pass": gate_2,
            "pct_bad_pass": gate_3,
        },
        "pass": passed,
    }

