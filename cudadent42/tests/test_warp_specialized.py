"""
Unit tests for warp-specialized FlashAttention kernel (Phase 1).

Tests numerical correctness, performance, and GPU compatibility.

@author GOATnote Autonomous Research Lab Initiative
@date 2025-10-11
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import flashmoe_science_ext
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


@pytest.mark.skipif(not HAS_CUDA_EXT, reason="CUDA extension not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestWarpSpecializedAttention:
    """Test suite for warp-specialized FlashAttention kernel."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_warp_specialized_vs_pytorch(self, dtype, seq_len, head_dim):
        """
        Test warp-specialized kernel against PyTorch reference.
        
        Validates numerical correctness across different configurations.
        """
        batch_size = 2
        num_heads = 4
        
        # Create random inputs
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda', requires_grad=False)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda', requires_grad=False)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda', requires_grad=False)
        
        # Softmax scale: 1/sqrt(d_k)
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        # Compute with warp-specialized kernel
        O_warp = flashmoe_science_ext.flash_attention_warp_specialized(
            Q, K, V, causal=False, softmax_scale=softmax_scale
        )
        
        # Compute with PyTorch reference
        scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V)
        
        # Check shapes match
        assert O_warp.shape == O_ref.shape, \
            f"Shape mismatch: {O_warp.shape} vs {O_ref.shape}"
        
        # Check numerical correctness
        # BF16/FP16 have limited precision, so we use larger tolerance
        max_diff = torch.max(torch.abs(O_warp - O_ref)).item()
        mean_diff = torch.mean(torch.abs(O_warp - O_ref)).item()
        
        tolerance = 0.02 if dtype == torch.bfloat16 else 0.01
        
        assert max_diff < tolerance, \
            f"Max difference {max_diff:.6f} exceeds tolerance {tolerance}"
        assert mean_diff < tolerance / 2, \
            f"Mean difference {mean_diff:.6f} exceeds tolerance {tolerance/2}"
    
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("seq_len", [128, 256, 512])
    def test_warp_specialized_causal_mask(self, dtype, seq_len):
        """
        Test warp-specialized kernel with causal masking.
        
        Validates that future tokens are properly masked out.
        """
        batch_size = 1
        num_heads = 2
        head_dim = 64
        
        # Create random inputs
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        # Compute with warp-specialized kernel (causal=True)
        O_warp = flashmoe_science_ext.flash_attention_warp_specialized(
            Q, K, V, causal=True, softmax_scale=softmax_scale
        )
        
        # Compute with PyTorch reference (causal=True)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device='cuda'), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V)
        
        # Check numerical correctness
        max_diff = torch.max(torch.abs(O_warp - O_ref)).item()
        tolerance = 0.02 if dtype == torch.bfloat16 else 0.01
        
        assert max_diff < tolerance, \
            f"Causal mask: Max difference {max_diff:.6f} exceeds tolerance {tolerance}"
    
    @pytest.mark.parametrize("seq_len", [128, 512, 1024])
    def test_warp_specialized_vs_basic(self, seq_len):
        """
        Test warp-specialized kernel against basic implementation.
        
        Both should give numerically identical results.
        """
        batch_size = 2
        num_heads = 4
        head_dim = 64
        dtype = torch.bfloat16
        
        # Create random inputs
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        # Compute with warp-specialized kernel
        O_warp = flashmoe_science_ext.flash_attention_warp_specialized(
            Q, K, V, causal=False, softmax_scale=softmax_scale
        )
        
        # Compute with basic kernel (if available)
        try:
            O_basic = flashmoe_science_ext.flash_attention_forward(
                Q, K, V, causal=False, softmax_scale=softmax_scale
            )
            
            # Both should give similar results
            max_diff = torch.max(torch.abs(O_warp - O_basic)).item()
            tolerance = 0.02  # Allow for slight numerical differences
            
            assert max_diff < tolerance, \
                f"Warp vs basic: Max difference {max_diff:.6f} exceeds tolerance {tolerance}"
        except AttributeError:
            pytest.skip("Basic kernel not available for comparison")
    
    def test_warp_specialized_deterministic(self):
        """
        Test that warp-specialized kernel is deterministic.
        
        Multiple runs with same input should give identical output.
        """
        batch_size = 2
        num_heads = 4
        seq_len = 256
        head_dim = 64
        dtype = torch.bfloat16
        
        # Create random inputs (but fixed)
        torch.manual_seed(42)
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        # Run multiple times
        outputs = []
        for _ in range(3):
            O = flashmoe_science_ext.flash_attention_warp_specialized(
                Q, K, V, causal=False, softmax_scale=softmax_scale
            )
            outputs.append(O)
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            diff = torch.max(torch.abs(outputs[i] - outputs[0])).item()
            assert diff == 0.0, \
                f"Non-deterministic behavior: run {i} differs by {diff}"
    
    def test_warp_specialized_edge_cases(self):
        """Test warp-specialized kernel with edge cases."""
        
        # Test 1: Minimum sequence length
        Q = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16, device='cuda')
        
        O = flashmoe_science_ext.flash_attention_warp_specialized(
            Q, K, V, causal=False, softmax_scale=0.125
        )
        
        assert O.shape == Q.shape, "Edge case 1: Shape mismatch"
        
        # Test 2: Large batch size
        Q = torch.randn(16, 8, 128, 64, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(16, 8, 128, 64, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(16, 8, 128, 64, dtype=torch.bfloat16, device='cuda')
        
        O = flashmoe_science_ext.flash_attention_warp_specialized(
            Q, K, V, causal=False, softmax_scale=0.125
        )
        
        assert O.shape == Q.shape, "Edge case 2: Shape mismatch"
    
    @pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
    def test_warp_specialized_numerical_stability(self, seq_len):
        """
        Test numerical stability with large values.
        
        Softmax should handle large attention scores without overflow.
        """
        batch_size = 1
        num_heads = 1
        head_dim = 64
        dtype = torch.bfloat16
        
        # Create inputs with large values
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda') * 10.0  # Scale up
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda') * 10.0
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        # Should not produce NaN or Inf
        O = flashmoe_science_ext.flash_attention_warp_specialized(
            Q, K, V, causal=False, softmax_scale=softmax_scale
        )
        
        assert not torch.isnan(O).any(), "Output contains NaN"
        assert not torch.isinf(O).any(), "Output contains Inf"
        
        # Values should be reasonable
        max_val = torch.max(torch.abs(O)).item()
        assert max_val < 1000.0, f"Output values too large: {max_val}"


@pytest.mark.skipif(not HAS_CUDA_EXT, reason="CUDA extension not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestWarpSpecializedPerformance:
    """Performance tests for warp-specialized kernel."""
    
    @pytest.mark.benchmark(group="attention")
    @pytest.mark.parametrize("seq_len", [512, 1024, 2048])
    def test_warp_specialized_throughput(self, benchmark, seq_len):
        """
        Benchmark warp-specialized kernel throughput.
        
        Measures time per forward pass.
        """
        batch_size = 4
        num_heads = 8
        head_dim = 64
        dtype = torch.bfloat16
        
        # Create inputs
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=dtype, device='cuda')
        
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        def run_kernel():
            O = flashmoe_science_ext.flash_attention_warp_specialized(
                Q, K, V, causal=False, softmax_scale=softmax_scale
            )
            torch.cuda.synchronize()
            return O
        
        # Warmup
        for _ in range(5):
            run_kernel()
        
        # Benchmark
        result = benchmark(run_kernel)
        
        # Print stats
        print(f"\nSeq len {seq_len}: {result.stats['mean']*1000:.2f}ms per forward")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])

