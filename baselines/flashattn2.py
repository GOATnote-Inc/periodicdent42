"""
FlashAttention-2 Baseline Implementation

Wraps official flash-attn package for drop-in usage.
"""

import torch
from .registry import register

try:
    from flash_attn import flash_attn_func
    HAS_FA2 = True
except ImportError:
    HAS_FA2 = False
    print("⚠️  flash-attn not installed, flashattn2 baseline unavailable")

if HAS_FA2:
    @register("flashattn2")
    def fa2(q, k, v, attn_mask=None, dropout_p=0.0, scale=None, causal=False, **_):
        """
        FlashAttention-2 wrapper
        
        Note: flash_attn_func expects [B, S, H, D] layout and doesn't support arbitrary masks.
        We reshape from PyTorch's [B, H, S, D] to flash-attn's expected layout.
        """
        # Input: [B, H, S, D] (PyTorch convention)
        # FA-2 expects: [B, S, H, D]
        B, H, S, D = q.shape
        
        # Transpose to [B, S, H, D]
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        
        # Call FA-2
        out_fa = flash_attn_func(
            q_fa, k_fa, v_fa, 
            dropout_p=dropout_p, 
            softmax_scale=scale, 
            causal=causal
        )
        
        # Transpose back to [B, H, S, D]
        return out_fa.transpose(1, 2)
else:
    # Register stub that raises error
    @register("flashattn2")
    def fa2(*args, **kwargs):
        raise RuntimeError("flash-attn not installed. Run: pip install flash-attn")

