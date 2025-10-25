#!/usr/bin/env python3
"""
FlashCore Correctness Tests

Validates kernel outputs against PyTorch SDPA reference implementation.
Tests multiple shapes and random seeds to prevent overfitting.

Test Coverage:
    - 5 shapes (tiny, small, medium, mission, multi_batch)
    - 3 random seeds per shape
    - Total: 15 test cases

Acceptance Criteria (FP16):
    - Max error ≤ 0.06
    - Mean error ≤ 0.02
    - No NaN or Inf values

Usage:
    pytest tests/test_correctness.py -v
    python tests/test_correctness.py  # Run directly
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def kernel():
    """Build kernel once per test session."""
    try:
        from build import build_baseline
        ext = build_baseline(verbose=False)
        return ext
    except Exception as e:
        pytest.fail(f"Failed to build kernel: {e}")

# ============================================================================
# Test Shapes
# ============================================================================

# Shape definitions: (name, config)
SHAPES = [
    ("tiny", {"B": 1, "H": 1, "S": 32, "D": 64}),
    ("small", {"B": 1, "H": 2, "S": 64, "D": 64}),
    ("medium", {"B": 1, "H": 4, "S": 128, "D": 64}),
    ("mission", {"B": 1, "H": 8, "S": 512, "D": 64}),  # Primary target
    ("multi_batch", {"B": 4, "H": 8, "S": 256, "D": 64}),
]

# Random seeds for reproducibility
SEEDS = [0, 42, 12345]

# Error thresholds (FP16 tolerance)
MAX_ERR_THRESHOLD = 0.06
MEAN_ERR_THRESHOLD = 0.02

# ============================================================================
# Helper Functions
# ============================================================================

def create_inputs(B, H, S, D, seed=0):
    """Create random Q, K, V tensors."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    return Q, K, V

def compute_pytorch_reference(Q, K, V, scale):
    """Compute reference output using PyTorch SDPA."""
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True
    ):
        O_ref = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale
        )
    return O_ref

def compute_errors(O_kernel, O_ref):
    """Compute error metrics."""
    diff = (O_kernel - O_ref).abs()
    
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (O_ref.abs() + 1e-8)).mean().item()
    
    # Percentage of "bad" elements (error > 0.1)
    bad_pct = (diff > 0.1).float().mean().item() * 100
    
    return {
        "max_err": max_err,
        "mean_err": mean_err,
        "rel_err": rel_err,
        "bad_pct": bad_pct,
    }

# ============================================================================
# Tests
# ============================================================================

@pytest.mark.parametrize("shape_name,shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_correctness(kernel, shape_name, shape, seed):
    """Test kernel correctness against PyTorch SDPA.
    
    Args:
        kernel: Compiled CUDA extension
        shape_name: Human-readable shape name
        shape: Dict with B, H, S, D
        seed: Random seed for reproducibility
    """
    
    B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
    
    # Create inputs
    Q, K, V = create_inputs(B, H, S, D, seed)
    scale = 1.0 / (D ** 0.5)
    
    # Kernel output
    O_kernel = kernel.forward(Q, K, V, scale)
    
    # PyTorch reference
    O_ref = compute_pytorch_reference(Q, K, V, scale)
    
    # Compute errors
    errors = compute_errors(O_kernel, O_ref)
    
    # Check for NaN/Inf
    has_nan = torch.isnan(O_kernel).any().item()
    has_inf = torch.isinf(O_kernel).any().item()
    
    # Print results
    status = "✅ PASS" if (
        errors["max_err"] < MAX_ERR_THRESHOLD and
        errors["mean_err"] < MEAN_ERR_THRESHOLD and
        not has_nan and not has_inf
    ) else "❌ FAIL"
    
    print(f"\n{status} | {shape_name:12s} (seed={seed:5d}) | "
          f"max={errors['max_err']:.4f} | mean={errors['mean_err']:.4f} | "
          f"rel={errors['rel_err']:.4f} | bad={errors['bad_pct']:.1f}%")
    
    # Assertions
    assert not has_nan, f"Output contains NaN (shape={shape_name}, seed={seed})"
    assert not has_inf, f"Output contains Inf (shape={shape_name}, seed={seed})"
    
    assert errors["max_err"] < MAX_ERR_THRESHOLD, \
        f"Max error {errors['max_err']:.4f} exceeds threshold {MAX_ERR_THRESHOLD} " \
        f"(shape={shape_name}, seed={seed})"
    
    assert errors["mean_err"] < MEAN_ERR_THRESHOLD, \
        f"Mean error {errors['mean_err']:.4f} exceeds threshold {MEAN_ERR_THRESHOLD} " \
        f"(shape={shape_name}, seed={seed})"

# ============================================================================
# Summary Test (Run Last)
# ============================================================================

@pytest.mark.parametrize("shape_name,shape", SHAPES)
def test_summary(kernel, shape_name, shape):
    """Summary test across all seeds (for reporting)."""
    B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
    
    errors_all = []
    for seed in SEEDS:
        Q, K, V = create_inputs(B, H, S, D, seed)
        scale = 1.0 / (D ** 0.5)
        
        O_kernel = kernel.forward(Q, K, V, scale)
        O_ref = compute_pytorch_reference(Q, K, V, scale)
        
        errors = compute_errors(O_kernel, O_ref)
        errors_all.append(errors)
    
    # Aggregate statistics
    max_err_all = max(e["max_err"] for e in errors_all)
    mean_err_all = sum(e["mean_err"] for e in errors_all) / len(errors_all)
    
    print(f"\n{shape_name:12s} | seeds={len(SEEDS)} | "
          f"max_err={max_err_all:.4f} | mean_err={mean_err_all:.4f}")
    
    # Final check
    assert max_err_all < MAX_ERR_THRESHOLD

# ============================================================================
# Main Entry Point (for direct execution)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FlashCore Correctness Tests")
    print("="*80)
    print(f"  Shapes: {len(SHAPES)}")
    print(f"  Seeds per shape: {len(SEEDS)}")
    print(f"  Total tests: {len(SHAPES) * len(SEEDS)}")
    print(f"  Max error threshold: {MAX_ERR_THRESHOLD}")
    print(f"  Mean error threshold: {MEAN_ERR_THRESHOLD}")
    print("="*80)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])

