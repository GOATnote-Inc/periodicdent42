"""
Test BlackwellSparseK CUDA kernels for correctness.

Compares kernel output against PyTorch SDPA baseline.
"""

import pytest
import torch

try:
    from blackwell_sparsek import attention_forward
    from blackwell_sparsek.utils import validate_correctness
    BLACKWELL_AVAILABLE = True
except ImportError:
    BLACKWELL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BLACKWELL_AVAILABLE or not torch.cuda.is_available(),
    reason="BlackwellSparseK requires CUDA"
)


@pytest.mark.gpu
@pytest.mark.parametrize("B,H,S,D", [
    (1, 8, 512, 64),
    (2, 16, 1024, 64),
    (4, 32, 2048, 64),
    (1, 8, 512, 128),
])
def test_kernel_correctness(B, H, S, D):
    """Test kernel correctness against PyTorch SDPA."""
    # Create random inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Compute reference with PyTorch SDPA
    scale = 1.0 / (D ** 0.5)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
    
    # Compute with BlackwellSparseK
    out = attention_forward(Q, K, V, scale=scale)
    
    # Validate correctness (FP16 tolerances)
    is_correct, metrics = validate_correctness(out, ref, rtol=1e-3, atol=2e-3)
    
    assert is_correct, (
        f"Kernel output does not match reference.\n"
        f"Config: B={B}, H={H}, S={S}, D={D}\n"
        f"Max diff: {metrics['max_diff']:.6f}\n"
        f"Mean diff: {metrics['mean_diff']:.6f}"
    )


@pytest.mark.gpu
def test_kernel_determinism():
    """Test that kernel produces consistent results across runs."""
    B, H, S, D = 1, 8, 512, 64
    
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Run twice
    out1 = attention_forward(Q, K, V)
    out2 = attention_forward(Q, K, V)
    
    # Should be exactly equal (same inputs, same kernel)
    assert torch.equal(out1, out2), "Kernel is not deterministic"


@pytest.mark.gpu
def test_input_validation():
    """Test input validation errors."""
    B, H, S, D = 1, 8, 512, 64
    
    # Valid input
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Should work
    _ = attention_forward(Q, K, V)
    
    # Wrong dtype
    Q_wrong = Q.float()
    with pytest.raises((ValueError, RuntimeError)):
        _ = attention_forward(Q_wrong, K, V)
    
    # Wrong device
    Q_cpu = Q.cpu()
    with pytest.raises((ValueError, RuntimeError)):
        _ = attention_forward(Q_cpu, K, V)
    
    # Wrong shape
    Q_wrong_shape = Q.unsqueeze(0)  # 5D instead of 4D
    with pytest.raises((ValueError, RuntimeError)):
        _ = attention_forward(Q_wrong_shape, K, V)


@pytest.mark.gpu
@pytest.mark.parametrize("D", [32, 256])
def test_unsupported_head_dim(D):
    """Test that unsupported head dimensions raise errors."""
    B, H, S = 1, 8, 512
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    with pytest.raises((ValueError, RuntimeError)):
        _ = attention_forward(Q, K, V)


@pytest.mark.gpu
def test_large_sequence_length():
    """Test kernel with large sequence length."""
    B, H, S, D = 1, 8, 4096, 64
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Should not crash
    out = attention_forward(Q, K, V)
    
    assert out.shape == (B, H, S, D), "Output shape mismatch"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"

