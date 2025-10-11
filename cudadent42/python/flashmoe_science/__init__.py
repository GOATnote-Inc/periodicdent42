"""
FlashMoE-Science: High-Performance CUDA Kernels for Scientific AI

Public API for attention and MoE operations.
"""

__version__ = "0.1.0"
__author__ = "GOATnote Autonomous Research Lab Initiative"

# Import CUDA extensions
try:
    from flashmoe_science import _C
    _CUDA_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: CUDA extensions not available: {e}")
    print("Please build extensions with: python setup.py build_ext --inplace")
    _CUDA_AVAILABLE = False

# Import public API
from flashmoe_science.ops import (
    flash_attention_science,
    flash_attention_backward,
    fused_moe,
    is_cuda_available,
)

from flashmoe_science.layers import (
    FlashMoEScienceAttention,
    FlashMoELayer,
)

__all__ = [
    # Operations
    "flash_attention_science",
    "flash_attention_backward",
    "fused_moe",
    "is_cuda_available",
    # Layers
    "FlashMoEScienceAttention",
    "FlashMoELayer",
    # Metadata
    "__version__",
]

