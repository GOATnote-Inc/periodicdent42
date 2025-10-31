"""Test vLLM backend integration."""

import pytest
import torch

try:
    from blackwell_sparsek.backends import SparseKBackend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BACKEND_AVAILABLE,
    reason="vLLM backend not available"
)


@pytest.mark.integration
def test_backend_registration():
    """Test that backend can be registered."""
    backend = SparseKBackend()
    
    assert backend.get_name() == "SPARSEK_XFORMERS"
    assert 64 in backend.get_supported_head_sizes()
    assert 128 in backend.get_supported_head_sizes()


@pytest.mark.gpu
@pytest.mark.integration
def test_backend_kv_cache_shape():
    """Test KV cache shape computation."""
    backend = SparseKBackend()
    
    num_blocks = 100
    block_size = 16
    num_kv_heads = 8
    head_size = 64
    
    shape = backend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
    
    assert shape == (num_blocks, block_size, num_kv_heads, head_size)


@pytest.mark.gpu
@pytest.mark.integration
def test_sparsek_attention_impl():
    """Test SparseKAttentionImpl forward pass."""
    from blackwell_sparsek.backends.vllm_backend import SparseKAttentionImpl
    
    num_heads = 8
    head_size = 64
    scale = 1.0 / (head_size ** 0.5)
    
    impl = SparseKAttentionImpl(num_heads, head_size, scale)
    
    B, S = 1, 512
    
    query = torch.randn(B, num_heads, S, head_size, dtype=torch.float16, device='cuda')
    key = torch.randn(B, num_heads, S, head_size, dtype=torch.float16, device='cuda')
    value = torch.randn(B, num_heads, S, head_size, dtype=torch.float16, device='cuda')
    
    output = impl.forward(query, key, value)
    
    assert output.shape == query.shape, "Output shape mismatch"
    assert output.dtype == torch.float16, "Output dtype mismatch"

