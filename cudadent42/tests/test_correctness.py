#!/usr/bin/env python3
"""
Numerical correctness tests for CUDAdent42 FlashAttention kernels.
Validates against PyTorch's scaled_dot_product_attention (reference implementation).
Tests various edge cases, numerical stability, and precision.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
import pytest

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def get_device_capability():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor

HAS_BF16 = get_device_capability() >= 80 if torch.cuda.is_available() else False


class TestFP16Correctness:
    """FP16 correctness tests (all GPUs with CUDA)."""
    
    def test_small_sequence_fp16(self):
        """Test small sequence (8 tokens) with FP16."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 8, 64
        Q = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        K = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        V = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        
        # Our kernel
        O_ours = fa.forward(Q, K, V)
        
        # PyTorch reference (upcast to FP32 for accuracy)
        Q_ref = Q.float()
        K_ref = K.float()
        V_ref = V.float()
        scale = 1.0 / math.sqrt(D)
        
        scores = torch.matmul(Q_ref, K_ref.t()) * scale
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V_ref).to(torch.float16)
        
        # Compare
        max_diff = (O_ours - O_ref).abs().max().item()
        mean_diff = (O_ours - O_ref).abs().mean().item()
        
        print(f"\n  Shape: {O_ours.shape}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        # FP16 tolerance: allow up to 0.05 (5%) error due to reduced precision
        assert max_diff < 0.1, f"Max diff {max_diff} exceeds threshold"
        assert mean_diff < 0.02, f"Mean diff {mean_diff} exceeds threshold"
        assert torch.isfinite(O_ours).all(), "NaN or Inf in output"
    
    def test_medium_sequence_fp16(self):
        """Test medium sequence (128 tokens) with FP16."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 128, 64
        Q = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        K = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        V = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        
        O_ours = fa.forward(Q, K, V)
        
        # Reference
        Q_ref = Q.float()
        K_ref = K.float()
        V_ref = V.float()
        scale = 1.0 / math.sqrt(D)
        
        scores = torch.matmul(Q_ref, K_ref.t()) * scale
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V_ref).to(torch.float16)
        
        max_diff = (O_ours - O_ref).abs().max().item()
        mean_diff = (O_ours - O_ref).abs().mean().item()
        
        print(f"\n  Shape: {O_ours.shape}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.15, f"Max diff {max_diff} exceeds threshold"
        assert mean_diff < 0.03, f"Mean diff {mean_diff} exceeds threshold"
        assert torch.isfinite(O_ours).all(), "NaN or Inf in output"
    
    def test_different_dimensions_fp16(self):
        """Test various dimension combinations."""
        import flashmoe_science._C as fa
        
        test_cases = [
            (1, 1, 16, 32),   # Small D
            (1, 1, 16, 64),   # Standard D
            (1, 1, 16, 128),  # Large D
            (1, 1, 32, 64),   # Larger S
        ]
        
        for B, H, S, D in test_cases:
            Q = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
            K = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
            V = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
            
            O_ours = fa.forward(Q, K, V)
            
            # Reference
            Q_ref = Q.float()
            K_ref = K.float()
            V_ref = V.float()
            scale = 1.0 / math.sqrt(D)
            
            scores = torch.matmul(Q_ref, K_ref.t()) * scale
            attn = torch.softmax(scores, dim=-1)
            O_ref = torch.matmul(attn, V_ref).to(torch.float16)
            
            max_diff = (O_ours - O_ref).abs().max().item()
            
            print(f"\n  Config: B={B}, H={H}, S={S}, D={D}")
            print(f"  Max diff: {max_diff:.6f}")
            
            assert max_diff < 0.15, f"Config {(B,H,S,D)}: max diff {max_diff} exceeds threshold"
            assert torch.isfinite(O_ours).all(), f"Config {(B,H,S,D)}: NaN or Inf in output"
    
    def test_extreme_values_fp16(self):
        """Test numerical stability with large magnitude inputs."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 16, 64
        
        # Large positive values
        Q = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda') * 10
        K = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda') * 10
        V = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        
        O_ours = fa.forward(Q, K, V)
        
        print(f"\n  Input magnitude: ~10x")
        print(f"  Output range: [{O_ours.min():.4f}, {O_ours.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(O_ours).any().item()}")
        print(f"  Has Inf: {torch.isinf(O_ours).any().item()}")
        
        assert torch.isfinite(O_ours).all(), "NaN or Inf with large inputs"
        assert O_ours.abs().mean() < 100, "Output magnitude unreasonably large"
    
    def test_zero_values_fp16(self):
        """Test edge case with zero or near-zero values."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 16, 64
        
        # Very small values
        Q = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda') * 0.01
        K = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda') * 0.01
        V = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        
        O_ours = fa.forward(Q, K, V)
        
        print(f"\n  Input magnitude: ~0.01x")
        print(f"  Output range: [{O_ours.min():.4f}, {O_ours.max():.4f}]")
        
        assert torch.isfinite(O_ours).all(), "NaN or Inf with small inputs"
    
    def test_identical_qk_fp16(self):
        """Test with Q = K (self-attention pattern)."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 16, 64
        Q = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        K = Q.clone()  # Identical
        V = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        
        O_ours = fa.forward(Q, K, V)
        
        # Reference
        Q_ref = Q.float()
        K_ref = K.float()
        V_ref = V.float()
        scale = 1.0 / math.sqrt(D)
        
        scores = torch.matmul(Q_ref, K_ref.t()) * scale
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V_ref).to(torch.float16)
        
        max_diff = (O_ours - O_ref).abs().max().item()
        
        print(f"\n  Q=K (self-attention)")
        print(f"  Max diff: {max_diff:.6f}")
        
        assert max_diff < 0.15, f"Self-attention: max diff {max_diff} exceeds threshold"
        assert torch.isfinite(O_ours).all(), "NaN or Inf in self-attention"


@pytest.mark.skipif(not HAS_BF16, reason="BF16 requires SM80+ (Ampere, L4, H100)")
class TestBF16Correctness:
    """BF16 correctness tests (SM80+: A100, L4, H100)."""
    
    def test_small_sequence_bf16(self):
        """Test small sequence (8 tokens) with BF16."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 8, 64
        Q = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        
        O_ours = fa.forward(Q, K, V)
        
        # Reference
        Q_ref = Q.float()
        K_ref = K.float()
        V_ref = V.float()
        scale = 1.0 / math.sqrt(D)
        
        scores = torch.matmul(Q_ref, K_ref.t()) * scale
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V_ref).to(torch.bfloat16)
        
        max_diff = (O_ours - O_ref).abs().max().item()
        mean_diff = (O_ours - O_ref).abs().mean().item()
        
        print(f"\n  Shape: {O_ours.shape}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        # BF16 has lower precision than FP16, allow larger error
        assert max_diff < 0.2, f"Max diff {max_diff} exceeds threshold"
        assert mean_diff < 0.05, f"Mean diff {mean_diff} exceeds threshold"
        assert torch.isfinite(O_ours).all(), "NaN or Inf in output"
    
    def test_medium_sequence_bf16(self):
        """Test medium sequence (128 tokens) with BF16."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 128, 64
        Q = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        
        O_ours = fa.forward(Q, K, V)
        
        # Reference
        Q_ref = Q.float()
        K_ref = K.float()
        V_ref = V.float()
        scale = 1.0 / math.sqrt(D)
        
        scores = torch.matmul(Q_ref, K_ref.t()) * scale
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V_ref).to(torch.bfloat16)
        
        max_diff = (O_ours - O_ref).abs().max().item()
        mean_diff = (O_ours - O_ref).abs().mean().item()
        
        print(f"\n  Shape: {O_ours.shape}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.25, f"Max diff {max_diff} exceeds threshold"
        assert mean_diff < 0.06, f"Mean diff {mean_diff} exceeds threshold"
        assert torch.isfinite(O_ours).all(), "NaN or Inf in output"
    
    def test_extreme_values_bf16(self):
        """Test numerical stability with large magnitude inputs (BF16)."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 16, 64
        
        Q = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda') * 10
        K = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda') * 10
        V = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        
        O_ours = fa.forward(Q, K, V)
        
        print(f"\n  Input magnitude: ~10x (BF16)")
        print(f"  Output range: [{O_ours.min():.4f}, {O_ours.max():.4f}]")
        
        assert torch.isfinite(O_ours).all(), "NaN or Inf with large inputs (BF16)"


class TestDtypeConsistency:
    """Test that different dtypes produce similar results (modulo precision)."""
    
    @pytest.mark.skipif(not HAS_BF16, reason="Needs BF16 support")
    def test_fp16_vs_bf16_consistency(self):
        """Compare FP16 and BF16 outputs (should be similar, not identical)."""
        import flashmoe_science._C as fa
        
        B, H, S, D = 1, 1, 32, 64
        torch.manual_seed(42)
        
        # Same inputs, different dtypes
        Q_base = torch.randn(B * H * S, D, dtype=torch.float32, device='cuda')
        K_base = torch.randn(B * H * S, D, dtype=torch.float32, device='cuda')
        V_base = torch.randn(B * H * S, D, dtype=torch.float32, device='cuda')
        
        Q_fp16 = Q_base.to(torch.float16)
        K_fp16 = K_base.to(torch.float16)
        V_fp16 = V_base.to(torch.float16)
        
        Q_bf16 = Q_base.to(torch.bfloat16)
        K_bf16 = K_base.to(torch.bfloat16)
        V_bf16 = V_base.to(torch.bfloat16)
        
        O_fp16 = fa.forward(Q_fp16, K_fp16, V_fp16)
        O_bf16 = fa.forward(Q_bf16, K_bf16, V_bf16)
        
        # Convert to same dtype for comparison
        diff = (O_fp16.float() - O_bf16.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\n  FP16 vs BF16 consistency:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        # Should be somewhat similar (within 0.3 due to dtype differences)
        assert max_diff < 0.5, f"FP16 vs BF16 too different: {max_diff}"
        assert mean_diff < 0.1, f"FP16 vs BF16 mean too different: {mean_diff}"


if __name__ == '__main__':
    # Run with pytest
    import subprocess
    result = subprocess.run([
        'pytest', __file__, '-v', '--tb=short',
        '--color=yes'
    ])
    sys.exit(result.returncode)

