"""Test xFormers integration."""

import pytest
import torch

try:
    from blackwell_sparsek.backends import SparseKAttention
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BACKEND_AVAILABLE or not torch.cuda.is_available(),
    reason="xFormers backend requires CUDA"
)


@pytest.mark.gpu
@pytest.mark.integration
def test_sparsek_attention_basic():
    """Test SparseKAttention with basic inputs."""
    attention = SparseKAttention()
    
    B, S, H, D = 1, 512, 8, 64
    
    # xFormers uses [B, S, H, D] layout
    q = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    
    # Forward pass
    output = attention(q, k, v)
    
    assert output.shape == (B, S, H, D), "Output shape mismatch"
    assert output.dtype == torch.float16, "Output dtype mismatch"
    assert output.device.type == 'cuda', "Output not on CUDA"


@pytest.mark.gpu
@pytest.mark.integration
def test_sparsek_attention_with_mask():
    """Test SparseKAttention with attention mask."""
    try:
        from xformers.components.attention import AttentionBias
    except ImportError:
        pytest.skip("xFormers not available")
    
    attention = SparseKAttention()
    
    B, S, H, D = 1, 512, 8, 64
    
    q = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    
    # Create causal mask
    mask = torch.tril(torch.ones(S, S, device='cuda', dtype=torch.bool))
    
    # Forward with mask (may fall back to PyTorch SDPA)
    output = attention(q, k, v, att_mask=mask)
    
    assert output.shape == (B, S, H, D), "Output shape mismatch with mask"

