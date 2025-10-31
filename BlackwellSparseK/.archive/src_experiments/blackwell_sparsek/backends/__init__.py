"""Framework integration backends for BlackwellSparseK."""

# Lazy imports to avoid dependency errors if frameworks not installed
__all__ = []

try:
    from .xformers_integration import SparseKAttention
    __all__.append("SparseKAttention")
except ImportError:
    pass

try:
    from .vllm_backend import SparseKBackend, register_vllm_backend
    __all__.extend(["SparseKBackend", "register_vllm_backend"])
except ImportError:
    pass

