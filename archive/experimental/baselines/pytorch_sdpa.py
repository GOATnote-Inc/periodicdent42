"""
PyTorch SDPA Baseline Implementations

Provides wrappers for all PyTorch SDPA backends:
- FLASH_ATTENTION
- CUDNN_ATTENTION
- EFFICIENT_ATTENTION (xFormers)
- MATH (reference)
"""

import torch
from contextlib import nullcontext

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAS_SDPA = True
except ImportError:
    HAS_SDPA = False

from .registry import register

def _sdpa_call(q, k, v, attn_mask=None, dropout_p=0.0, scale=None, causal=False, backend=None):
    """Internal SDPA caller with optional backend selection"""
    if not HAS_SDPA:
        raise RuntimeError("torch.nn.attention not available (PyTorch >= 2.0 required)")
    
    ctx = sdpa_kernel(backend) if backend else nullcontext()
    with ctx:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=dropout_p, 
            scale=scale, 
            is_causal=causal
        )

if HAS_SDPA:
    @register("pytorch_sdpa_flash")
    def sdpa_flash(q, k, v, **kw):
        """PyTorch SDPA with FLASH_ATTENTION backend"""
        return _sdpa_call(q, k, v, backend=SDPBackend.FLASH_ATTENTION, **kw)

    @register("pytorch_sdpa_cudnn")
    def sdpa_cudnn(q, k, v, **kw):
        """PyTorch SDPA with CUDNN_ATTENTION backend"""
        return _sdpa_call(q, k, v, backend=SDPBackend.CUDNN_ATTENTION, **kw)

    @register("pytorch_sdpa_efficient")
    def sdpa_efficient(q, k, v, **kw):
        """PyTorch SDPA with EFFICIENT_ATTENTION backend (xFormers)"""
        return _sdpa_call(q, k, v, backend=SDPBackend.EFFICIENT_ATTENTION, **kw)

    @register("pytorch_sdpa_math")
    def sdpa_math(q, k, v, **kw):
        """PyTorch SDPA with MATH backend (reference)"""
        return _sdpa_call(q, k, v, backend=SDPBackend.MATH, **kw)
else:
    print("⚠️  PyTorch SDPA not available (PyTorch < 2.0)")

