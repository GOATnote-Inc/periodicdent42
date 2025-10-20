#!/usr/bin/env python3
"""
Unit test for FP8 WMMA kernel correctness gates.

This test enforces the three correctness gates on a tiny shape to catch regressions:
  1. max_abs_err ≤ 0.05
  2. mean_abs_err ≤ 0.01
  3. %(|err| > 0.05) ≤ 1%

Usage:
    pytest tests/test_fp8_wmma_correctness.py -v
"""

import pytest
import torch
import math
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.fp8_sdpa_stage_c_wmma.build import build_extension
from tasks.fp8_sdpa_stage_c_wmma.func_forward import (
    forward_ref,
    forward_kernel,
    quantize_sim_fp8_per_head,
    validate_correctness,
)

# Skip if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

@pytest.fixture(scope="module")
def extension():
    """Build extension once for all tests."""
    return build_extension(verbose=False)

def test_small_shape_correctness_seed0(extension):
    """Test correctness on small shape (B=1,H=1,S=32,D=64) with seed=0."""
    # Setup
    B, H, S, D = 1, 1, 32, 64
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Generate inputs
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)
    
    # Reference
    ref = forward_ref(Q, K, V, scale)
    
    # Quantize
    Q_q, Q_s = quantize_sim_fp8_per_head(Q)
    K_q, K_s = quantize_sim_fp8_per_head(K)
    V_q, V_s = quantize_sim_fp8_per_head(V)
    
    # Kernel
    out = forward_kernel(Q_q, K_q, V_q, Q_s, K_s, V_s, scale, extension)
    
    # Validate
    metrics = validate_correctness(ref, out, atol=0.05, rtol=0.05, pct_bad_max=1.0)
    
    # Assertions (all three gates)
    assert metrics["max_abs_err"] <= 0.05, (
        f"Gate 1 FAIL: max_abs_err={metrics['max_abs_err']:.4f} > 0.05"
    )
    assert metrics["mean_abs_err"] <= 0.01, (
        f"Gate 2 FAIL: mean_abs_err={metrics['mean_abs_err']:.4f} > 0.01"
    )
    assert metrics["pct_bad"] <= 1.0, (
        f"Gate 3 FAIL: pct_bad={metrics['pct_bad']:.2f}% > 1.0%"
    )
    
    # Overall
    assert metrics["pass"], "Correctness gates failed!"
    
    print(f"\n✅ Correctness PASS: max_err={metrics['max_abs_err']:.4f}, "
          f"mean_err={metrics['mean_abs_err']:.4f}, %bad={metrics['pct_bad']:.1f}%")

def test_small_shape_correctness_seed1(extension):
    """Test correctness on small shape with seed=1."""
    # Setup
    B, H, S, D = 1, 1, 32, 64
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Generate inputs
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)
    
    # Reference
    ref = forward_ref(Q, K, V, scale)
    
    # Quantize
    Q_q, Q_s = quantize_sim_fp8_per_head(Q)
    K_q, K_s = quantize_sim_fp8_per_head(K)
    V_q, V_s = quantize_sim_fp8_per_head(V)
    
    # Kernel
    out = forward_kernel(Q_q, K_q, V_q, Q_s, K_s, V_s, scale, extension)
    
    # Validate
    metrics = validate_correctness(ref, out, atol=0.05, rtol=0.05, pct_bad_max=1.0)
    
    # Assertions
    assert metrics["pass"], (
        f"Correctness gates failed: max_abs_err={metrics['max_abs_err']:.4f}, "
        f"mean_abs_err={metrics['mean_abs_err']:.4f}, pct_bad={metrics['pct_bad']:.1f}%"
    )
    
    print(f"\n✅ Correctness PASS: max_err={metrics['max_abs_err']:.4f}, "
          f"mean_err={metrics['mean_abs_err']:.4f}, %bad={metrics['pct_bad']:.1f}%")

def test_mission_shape_correctness_seed0(extension):
    """Test correctness on mission shape (B=1,H=8,S=512,D=64) with seed=0."""
    # Setup
    B, H, S, D = 1, 8, 512, 64
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Generate inputs
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)
    
    # Reference
    ref = forward_ref(Q, K, V, scale)
    
    # Quantize
    Q_q, Q_s = quantize_sim_fp8_per_head(Q)
    K_q, K_s = quantize_sim_fp8_per_head(K)
    V_q, V_s = quantize_sim_fp8_per_head(V)
    
    # Kernel
    out = forward_kernel(Q_q, K_q, V_q, Q_s, K_s, V_s, scale, extension)
    
    # Validate
    metrics = validate_correctness(ref, out, atol=0.05, rtol=0.05, pct_bad_max=1.0)
    
    # Assertions
    assert metrics["pass"], (
        f"Correctness gates failed: max_abs_err={metrics['max_abs_err']:.4f}, "
        f"mean_abs_err={metrics['mean_abs_err']:.4f}, pct_bad={metrics['pct_bad']:.1f}%"
    )
    
    print(f"\n✅ Correctness PASS (mission): max_err={metrics['max_abs_err']:.4f}, "
          f"mean_err={metrics['mean_abs_err']:.4f}, %bad={metrics['pct_bad']:.1f}%")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

