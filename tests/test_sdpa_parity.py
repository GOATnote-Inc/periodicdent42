#!/usr/bin/env python3
"""
SDPA Parity Tests - Phase 2: Correctness Validation
Tests our FlashAttention kernel against PyTorch SDPA across multiple configurations.

Test Grid:
- Dtypes: FP16, BF16
- Head dims: 64, 80, 96, 128
- Seq lens: 128, 512, 1024, 2048, 4096
- Batch sizes: 1, 4, 8
- Num heads: 8, 16
- Causal: True, False
- Dropout: 0 (deterministic)

Tolerances: atol=1e-2, rtol=1e-2 (FP16 precision)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "cudadent42"))

import torch
import torch.nn.functional as F
import pytest
from typing import Tuple

# Import our kernel
try:
    from bench.fa_inverted_prod import flash_attention_inverted_forward as our_kernel
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False
    print("⚠️  Warning: Production kernel not available, skipping tests")


def sdpa_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """PyTorch SDPA reference implementation"""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


def generate_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic test inputs"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    return Q, K, V


@pytest.mark.skipif(not KERNEL_AVAILABLE, reason="Kernel not compiled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 80, 96, 128])
@pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_parity(
    dtype: torch.dtype,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    causal: bool,
):
    """Test parity between our kernel and PyTorch SDPA"""
    
    # Generate inputs
    Q, K, V = generate_inputs(batch_size, num_heads, seq_len, head_dim, dtype)
    
    # Compute outputs
    out_sdpa = sdpa_reference(Q, K, V, is_causal=causal)
    out_ours = our_kernel(Q, K, V, is_causal=causal)
    
    # Check for NaN/Inf
    assert torch.isfinite(out_sdpa).all(), "SDPA output contains NaN/Inf"
    assert torch.isfinite(out_ours).all(), "Our kernel output contains NaN/Inf"
    
    # Check shape
    assert out_ours.shape == out_sdpa.shape, \
        f"Shape mismatch: {out_ours.shape} vs {out_sdpa.shape}"
    
    # Check values with tolerances
    max_diff = (out_ours - out_sdpa).abs().max().item()
    mean_diff = (out_ours - out_sdpa).abs().mean().item()
    
    atol = 1e-2  # FP16 precision
    rtol = 1e-2
    
    try:
        torch.testing.assert_close(
            out_ours,
            out_sdpa,
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as e:
        print(f"\n❌ Parity test failed:")
        print(f"   Config: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}, "
              f"causal={causal}, dtype={dtype}")
        print(f"   Max diff: {max_diff:.6f}")
        print(f"   Mean diff: {mean_diff:.6f}")
        raise


@pytest.mark.skipif(not KERNEL_AVAILABLE, reason="Kernel not compiled")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_long_sequence_stress():
    """Stress test with very long sequences"""
    
    for seq_len in [4096, 8192]:
        print(f"\n  Testing S={seq_len}... ", end="", flush=True)
        
        Q, K, V = generate_inputs(1, 8, seq_len, 128, torch.float16)
        
        try:
            out_sdpa = sdpa_reference(Q, K, V, is_causal=True)
            out_ours = our_kernel(Q, K, V, is_causal=True)
            
            assert torch.isfinite(out_ours).all(), "Output contains NaN/Inf"
            
            torch.testing.assert_close(
                out_ours,
                out_sdpa,
                atol=1e-2,
                rtol=1e-2,
            )
            
            print(f"✓ Passed (max_diff={( out_ours - out_sdpa).abs().max().item():.6f})")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise


def run_manual_tests():
    """Run tests manually without pytest"""
    if not KERNEL_AVAILABLE:
        print("❌ Kernel not available, skipping tests")
        return False
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tests")
        return False
    
    print("=" * 60)
    print("SDPA Parity Tests - Manual Execution")
    print("=" * 60)
    
    # Canonical shapes
    test_configs = [
        (1, 8, 512, 64, torch.float16, False),   # Small
        (4, 16, 2048, 128, torch.float16, True), # Large
        (1, 8, 4096, 128, torch.float16, True),  # Long seq
    ]
    
    passed = 0
    failed = 0
    
    for B, H, S, D, dtype, causal in test_configs:
        print(f"\n  Testing B={B}, H={H}, S={S}, D={D}, causal={causal}... ", end="", flush=True)
        
        try:
            Q, K, V = generate_inputs(B, H, S, D, dtype)
            out_sdpa = sdpa_reference(Q, K, V, is_causal=causal)
            out_ours = our_kernel(Q, K, V, is_causal=causal)
            
            assert torch.isfinite(out_ours).all()
            torch.testing.assert_close(out_ours, out_sdpa, atol=1e-2, rtol=1e-2)
            
            max_diff = (out_ours - out_sdpa).abs().max().item()
            print(f"✓ Passed (max_diff={max_diff:.6f})")
            passed += 1
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_manual_tests()
    sys.exit(0 if success else 1)

