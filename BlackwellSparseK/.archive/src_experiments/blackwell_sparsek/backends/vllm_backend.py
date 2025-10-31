"""vLLM V1 backend integration for BlackwellSparseK.

Provides attention backend for vLLM inference server using BlackwellSparseK kernels.
"""

import torch
from typing import Any, Dict, List, Optional, Tuple, Type

try:
    from vllm.attention.backends.abstract import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadata,
        AttentionType,
    )
    from vllm.attention.selector import AttentionBackendRegistry
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Provide stubs
    class AttentionBackend:
        pass
    class AttentionImpl:
        pass
    class AttentionMetadata:
        pass
    AttentionBackendRegistry = None


from blackwell_sparsek import attention_forward


class SparseKAttentionImpl(AttentionImpl):
    """Attention implementation using BlackwellSparseK kernels."""
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Compute attention.
        
        Args:
            query: [B, S, H, D] or [B, H, S, D]
            key: [B, S, H, D] or [B, H, S, D]
            value: [B, S, H, D] or [B, H, S, D]
            kv_cache: Optional KV cache
            attn_metadata: Optional attention metadata
        
        Returns:
            Output tensor [B, S, H, D] or [B, H, S, D]
        """
        # Ensure [B, H, S, D] layout
        if query.dim() == 4 and query.size(1) != self.num_heads:
            # Input is [B, S, H, D], transpose
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()
            needs_transpose = True
        else:
            needs_transpose = False
        
        # Call BlackwellSparseK kernel
        output = attention_forward(query, key, value, scale=self.scale)
        
        # Transpose back if needed
        if needs_transpose:
            output = output.transpose(1, 2).contiguous()
        
        return output


class SparseKBackend(AttentionBackend):
    """
    vLLM attention backend using BlackwellSparseK kernels.
    
    Provides high-performance attention for vLLM inference on Hopper/Blackwell GPUs.
    
    Example usage in vLLM:
        $ python -m vllm.entrypoints.openai.api_server \\
            --model meta-llama/Llama-3.1-70B \\
            --attention-backend SPARSEK_XFORMERS
    """
    
    @staticmethod
    def get_name() -> str:
        """Backend name for registration."""
        return "SPARSEK_XFORMERS"
    
    @staticmethod
    def get_impl_cls() -> Type[AttentionImpl]:
        """Get attention implementation class."""
        return SparseKAttentionImpl
    
    @staticmethod
    def get_metadata_cls() -> Type[AttentionMetadata]:
        """Get attention metadata class (use default)."""
        if VLLM_AVAILABLE:
            from vllm.attention.backends.abstract import AttentionMetadata
            return AttentionMetadata
        return None
    
    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        """Supported head dimensions."""
        return [64, 128]
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        """
        Get KV cache shape.
        
        Uses standard paged attention layout:
        [num_blocks, block_size, num_kv_heads, head_size]
        """
        return (num_blocks, block_size, num_kv_heads, head_size)
    
    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        """Swap KV cache blocks (standard implementation)."""
        for src_idx, dst_idx in src_to_dst.items():
            dst_kv_cache[dst_idx].copy_(src_kv_cache[src_idx])
    
    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        """Copy KV cache blocks (standard implementation)."""
        for src_idx, dst_indices in src_to_dists.items():
            for dst_idx in dst_indices:
                for kv_cache in kv_caches:
                    kv_cache[dst_idx].copy_(kv_cache[src_idx])


def register_vllm_backend():
    """
    Register BlackwellSparseK backend with vLLM.
    
    Call this function to make the backend available:
        >>> from blackwell_sparsek.backends import register_vllm_backend
        >>> register_vllm_backend()
    
    Then use with:
        --attention-backend SPARSEK_XFORMERS
    """
    if not VLLM_AVAILABLE:
        raise ImportError(
            "vLLM not installed. Install with: pip install vllm>=0.11.0"
        )
    
    if AttentionBackendRegistry is None:
        raise ImportError("vLLM AttentionBackendRegistry not available")
    
    # Register the backend
    AttentionBackendRegistry.register_backend(
        "SPARSEK_XFORMERS",
        SparseKBackend
    )
    
    print("✅ BlackwellSparseK backend registered with vLLM")
    print("   Use: --attention-backend SPARSEK_XFORMERS")


# Auto-register on import if vLLM is available
if VLLM_AVAILABLE:
    try:
        register_vllm_backend()
    except Exception as e:
        # Silent fail on registration errors (may be due to vLLM version)
        pass

