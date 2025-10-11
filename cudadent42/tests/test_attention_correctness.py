"""
FlashAttention-Science: Correctness tests

Validates numerical accuracy against PyTorch reference implementation.
"""

import pytest
import torch
import torch.nn.functional as F

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for FlashMoE-Science tests"
)


class TestFlashAttentionCorrectness:
    """Test numerical correctness of FlashAttention kernels."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set random seed for reproducibility."""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("seq_len", [128, 512, 2048])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_forward_vs_pytorch(self, dtype, seq_len, head_dim):
        """Test forward pass matches PyTorch SDPA."""
        try:
            from flashmoe_science import flash_attention_science
        except ImportError:
            pytest.skip("FlashMoE-Science not built. Run: python setup.py build_ext --inplace")
        
        batch_size = 4
        num_heads = 8
        
        # Create random tensors
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        
        # PyTorch reference
        with torch.no_grad():
            ref_output = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        
        # FlashMoE-Science kernel
        with torch.no_grad():
            our_output = flash_attention_science(Q, K, V, causal=False)
        
        # Check numerical accuracy
        max_error = (ref_output - our_output).abs().max().item()
        mean_error = (ref_output - our_output).abs().mean().item()
        
        # Tolerance depends on dtype
        tol = 5e-2 if dtype == torch.bfloat16 else 1e-2
        
        assert max_error < tol, f"Max error {max_error} exceeds tolerance {tol}"
        assert mean_error < tol / 10, f"Mean error {mean_error} too large"
        
        print(f"✓ dtype={dtype}, seq_len={seq_len}, head_dim={head_dim}: "
              f"max_err={max_error:.2e}, mean_err={mean_error:.2e}")
    
    @pytest.mark.parametrize("seq_len", [128, 512])
    def test_causal_masking(self, seq_len):
        """Test causal attention mask is applied correctly."""
        try:
            from flashmoe_science import flash_attention_science
        except ImportError:
            pytest.skip("FlashMoE-Science not built")
        
        batch_size, num_heads, head_dim = 2, 4, 64
        dtype = torch.bfloat16
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        
        # PyTorch reference with causal mask
        with torch.no_grad():
            ref_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        # Our kernel with causal=True
        with torch.no_grad():
            our_output = flash_attention_science(Q, K, V, causal=True)
        
        # Check numerical accuracy
        max_error = (ref_output - our_output).abs().max().item()
        tol = 5e-2
        
        assert max_error < tol, f"Causal masking incorrect: max_err={max_error}"
        print(f"✓ Causal masking correct (seq_len={seq_len}, max_err={max_error:.2e})")
    
    def test_empty_tensor(self):
        """Test handling of edge case: empty tensors."""
        try:
            from flashmoe_science import flash_attention_science
        except ImportError:
            pytest.skip("FlashMoE-Science not built")
        
        # Edge case: zero sequence length
        Q = torch.randn(1, 1, 0, 64, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(1, 1, 0, 64, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(1, 1, 0, 64, device='cuda', dtype=torch.bfloat16)
        
        with torch.no_grad():
            output = flash_attention_science(Q, K, V)
        
        assert output.shape == Q.shape
        print("✓ Empty tensor handling correct")
    
    def test_numerical_stability(self):
        """Test numerical stability with large values."""
        try:
            from flashmoe_science import flash_attention_science
        except ImportError:
            pytest.skip("FlashMoE-Science not built")
        
        batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64
        dtype = torch.bfloat16
        
        # Create tensors with large values (test overflow handling)
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype) * 10.0
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype) * 10.0
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
        
        # Should not produce NaN or Inf
        with torch.no_grad():
            output = flash_attention_science(Q, K, V)
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        print("✓ Numerical stability check passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

