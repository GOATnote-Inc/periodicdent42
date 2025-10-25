"""FP8 Stage C (WMMA) FlashAttention forward helper.

This module provides a high-level Python wrapper around the simulated FP8
FlashAttention kernel implemented in
``cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu``. It handles

* Per-head symmetric quantisation into simulated FP8 (`uint8`) values.
* Lazy JIT compilation of the CUDA extension with the correct architecture
  flags for the active GPU.
* Convenience API that accepts FP16 tensors and returns FP16 output suitable
  for parity checks against PyTorch's SDPA reference implementation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load


_KERNEL_DIR = Path(__file__).parent / "kernels"


def _get_arch_flag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"--generate-code=arch=compute_{major}{minor},code=sm_{major}{minor}"


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to build FP8 WMMA kernel")

    arch_flag = _get_arch_flag()

    return load(
        name="sdpa_fp8_stage_c_wmma",
        sources=[
            str(_KERNEL_DIR / "sdpa_fp8_stage_c_wmma.cu"),
            str(_KERNEL_DIR / "sdpa_fp8_stage_c_wmma_bindings.cpp"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-lineinfo",
            "-Xptxas",
            "-v",
            "-std=c++17",
            arch_flag,
        ],
        verbose=False,
    )


def quantize_sim_fp8_per_head(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise an FP16 tensor to simulated FP8 using symmetric mapping.

    Args:
        tensor: Input tensor with shape ``[B, H, S, D]`` on CUDA and dtype
            ``torch.float16``.

    Returns:
        A tuple ``(encoded, scales)`` where ``encoded`` is a ``torch.uint8``
        tensor of the same shape representing simulated FP8 values and
        ``scales`` is a length-``H`` vector of per-head scaling factors in
        ``torch.float32``.
    """

    if tensor.device.type != "cuda":
        raise ValueError("quantize_sim_fp8_per_head expects CUDA tensors")
    if tensor.dtype != torch.float16:
        raise ValueError("quantize_sim_fp8_per_head expects FP16 input")

    if tensor.dim() != 4:
        raise ValueError("Expected tensor shape [B, H, S, D]")

    fp8_max = 448.0

    abs_max = tensor.abs().to(torch.float32).amax(dim=(0, 2, 3), keepdim=True)
    
    # PRIORITY 1 FIX: For zero/near-zero tensors, use scale=1.0 (not 1.0/448.0)
    # This ensures quantization maps zeros to midpoint (128) with correct scale
    scales = torch.where(
        abs_max > 1e-6,
        abs_max / fp8_max,
        torch.ones_like(abs_max)
    ).to(torch.float32)
    
    # Compute denominator for quantization: scale * fp8_max
    # For zero tensors: scale=1.0 → denom=448.0 → input/denom=0 → encoded=128 ✓
    denom = (scales * fp8_max).to(torch.float32)

    # Quantize: map [-denom, +denom] → [0, 255] with midpoint at 128
    encoded = torch.round(
        tensor.to(torch.float32) / denom * 127.0 + 128.0
    )
    encoded = torch.clamp(encoded, 0.0, 255.0)

    # For zero/near-zero heads, explicitly set encoded to midpoint (128)
    # This redundantly ensures correct handling even if numerical precision fails
    zero_mask = abs_max <= 1e-6
    if zero_mask.any():
        encoded = torch.where(
            zero_mask,
            torch.full_like(encoded, 128.0),
            encoded
        )

    encoded_uint8 = encoded.to(torch.uint8)
    scales_vector = scales.view(-1).to(torch.float32)

    return encoded_uint8, scales_vector


def sdpa_fp8_stage_c_wmma_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run the Stage C FP8 WMMA kernel.

    Args:
        Q, K, V: Input tensors shaped ``[B, H, S, D]`` (FP16, CUDA).
        softmax_scale: Optional override for the attention scale factor. The
            default uses ``1/sqrt(D)``.
        is_causal: Causal masking is not currently supported and will raise
            ``NotImplementedError`` if ``True``.

    Returns:
        The FP16 attention output tensor.
    """

    if is_causal:
        raise NotImplementedError("Causal masking is not implemented for Stage C")

    for name, tensor in {"Q": Q, "K": K, "V": V}.items():
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must reside on CUDA device")
        if tensor.dtype != torch.float16:
            raise ValueError(f"{name} must be torch.float16")
        if tensor.dim() != 4:
            raise ValueError(f"{name} must have shape [B, H, S, D]")

    if not (Q.shape == K.shape == V.shape):
        raise ValueError("Q, K, V must share shape [B, H, S, D]")

    B, H, S, D = Q.shape
    if D != 64:
        raise ValueError("Stage C kernel currently supports HEAD_DIM=64 only")

    module = _load_extension()

    Q_enc, Q_scale = quantize_sim_fp8_per_head(Q.contiguous())
    K_enc, K_scale = quantize_sim_fp8_per_head(K.contiguous())
    V_enc, V_scale = quantize_sim_fp8_per_head(V.contiguous())

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(D))

    # Launch kernel via bindings.
    return module.forward(
        Q_enc,
        K_enc,
        V_enc,
        Q_scale,
        K_scale,
        V_scale,
        softmax_scale,
    )


__all__ = [
    "sdpa_fp8_stage_c_wmma_forward",
    "quantize_sim_fp8_per_head",
]

