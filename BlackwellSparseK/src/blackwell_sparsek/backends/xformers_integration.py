"""xFormers integration for BlackwellSparseK.

Provides drop-in replacement for xFormers attention with BlackwellSparseK backend.
"""

import torch
from typing import Optional

try:
    from xformers.components.attention import Attention, AttentionBias
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    # Provide stub classes
    class Attention:
        pass
    class AttentionBias:
        pass

from blackwell_sparsek import attention_forward


class SparseKAttention(Attention):
    """
    xFormers-compatible attention using BlackwellSparseK kernels.
    
    Drop-in replacement for xFormers.Attention that uses optimized
    CUTLASS 4.3.0-based kernels for Hopper/Blackwell architectures.
    
    Example:
        >>> from blackwell_sparsek.backends import SparseKAttention
        >>> from xformers.components.attention import AttentionBias
        >>> 
        >>> attention = SparseKAttention()
        >>> 
        >>> q = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')
        >>> k = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')
        >>> v = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')
        >>> 
        >>> # Optional attention mask
        >>> mask = AttentionBias.from_bool(torch.ones(1, 512, 512, dtype=torch.bool))
        >>> 
        >>> output = attention(q, k, v, att_mask=mask)
    """
    
    def __init__(self, *args, **kwargs):
        if not XFORMERS_AVAILABLE:
            raise ImportError(
                "xFormers not installed. Install with: pip install xformers"
            )
        super().__init__(*args, **kwargs)
        self._scale = None
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[AttentionBias] = None,
        **kwargs
    ) -> torch::Tensor:
        """
        Compute attention with optional AttentionBias mask.
        
        Args:
            q: Query tensor [B, S, H, D] or [B, H, S, D]
            k: Key tensor [B, S, H, D] or [B, H, S, D]
            v: Value tensor [B, S, H, D] or [B, H, S, D]
            att_mask: Optional AttentionBias mask
        
        Returns:
            Output tensor same shape as q
        """
        # Handle input layout variations
        # xFormers uses [B, S, H, D] by default
        # BlackwellSparseK uses [B, H, S, D]
        needs_transpose = False
        if q.dim() == 4:
            if q.size(1) != k.size(1):  # Check if S dimension is dim 1
                # Input is [B, S, H, D], transpose to [B, H, S, D]
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                needs_transpose = True
        
        # Handle attention mask
        if att_mask is not None:
            # Convert AttentionBias to tensor
            if hasattr(att_mask, 'materialize'):
                # Standard AttentionBias types
                mask_tensor = att_mask.materialize(
                    (q.size(0), q.size(1), q.size(2), k.size(2))
                )
            else:
                # Direct tensor mask
                mask_tensor = att_mask
            
            # Apply mask by setting masked positions to -inf before softmax
            # This requires modifying the kernel or using PyTorch fallback
            # For now, use PyTorch SDPA with mask
            if self._scale is None:
                D = q.size(-1)
                self._scale = 1.0 / (D ** 0.5)
            
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask_tensor, scale=self._scale
            )
        else:
            # Use BlackwellSparseK kernel (no mask support yet)
            output = attention_forward(q, k, v)
        
        # Transpose back if needed
        if needs_transpose:
            output = output.transpose(1, 2).contiguous()
        
        return output


def create_sparsek_attention(**kwargs):
    """
    Factory function to create SparseKAttention instance.
    
    Returns:
        SparseKAttention instance
    """
    if not XFORMERS_AVAILABLE:
        raise ImportError(
            "xFormers not installed. Install with: pip install xformers"
        )
    return SparseKAttention(**kwargs)

