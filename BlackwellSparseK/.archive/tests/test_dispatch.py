"""Test architecture dispatch logic."""

import pytest
import torch

try:
    from blackwell_sparsek.core import get_build_info
    BLACKWELL_AVAILABLE = True
except ImportError:
    BLACKWELL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BLACKWELL_AVAILABLE,
    reason="BlackwellSparseK not available"
)


def test_build_info():
    """Test build information retrieval."""
    info = get_build_info()
    
    assert "torch_version" in info
    assert "cuda_available" in info
    
    if torch.cuda.is_available():
        assert "device_name" in info
        assert "device_capability" in info
        assert "compute_arch" in info


@pytest.mark.gpu
def test_architecture_detection():
    """Test GPU architecture detection."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    info = get_build_info()
    arch = info.get("compute_arch", "")
    
    # Should detect sm_XX format
    assert arch.startswith("sm_"), f"Invalid arch format: {arch}"
    
    # Extract major/minor version
    major, minor = torch.cuda.get_device_capability()
    expected_arch = f"sm_{major}{minor}"
    
    assert arch == expected_arch, f"Arch mismatch: {arch} != {expected_arch}"


@pytest.mark.gpu
def test_supported_architectures():
    """Test that current GPU is supported."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 100 + minor * 10
    
    # BlackwellSparseK requires sm_90a or sm_100
    supported = sm_version >= 900
    
    if not supported:
        pytest.skip(f"GPU sm_{major}{minor} not supported (requires sm_90a or sm_100)")
    
    # If we get here, GPU is supported
    from blackwell_sparsek import attention_forward
    
    # Should not raise architecture error
    B, H, S, D = 1, 8, 512, 64
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    _ = attention_forward(Q, K, V)

