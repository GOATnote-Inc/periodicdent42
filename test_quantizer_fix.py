#!/usr/bin/env python3
"""Quick validation script for quantizer scale bug fix"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "cudadent42"))

from bench.sdpa_fp8_stage_c_wmma import quantize_sim_fp8_per_head

def test_quantizer_zero_fix():
    """Test that zero tensors get scale=1.0"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping test")
        return False
    
    print("🔍 Testing quantizer scale bug fix...")
    print("   Input: zeros(1, 2, 16, 64) [B, H, S, D]")
    
    zeros = torch.zeros(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    encoded, scales = quantize_sim_fp8_per_head(zeros)
    
    # Check 1: All encoded values should be 128 (midpoint)
    all_128 = torch.all(encoded == 128).item()
    print(f"   Check 1: encoded == 128 everywhere? {all_128}")
    if not all_128:
        print(f"      ❌ FAILED: Found values != 128")
        print(f"         unique values: {torch.unique(encoded).cpu().tolist()}")
        return False
    
    # Check 2: Scales should be 1.0 (not 1.0/448.0 = 0.0022...)
    scales_cpu = scales.cpu()
    expected = torch.ones(2)
    close = torch.allclose(scales_cpu, expected, atol=1e-6)
    print(f"   Check 2: scales == [1.0, 1.0]? {close}")
    if not close:
        print(f"      ❌ FAILED: scales = {scales_cpu.tolist()}")
        print(f"         expected: {expected.tolist()}")
        print(f"         diff: {(scales_cpu - expected).abs().tolist()}")
        return False
    
    print(f"   ✅ Both checks PASSED")
    print(f"      encoded: all 128 ✓")
    print(f"      scales: {scales_cpu.tolist()} ✓")
    return True


def test_quantizer_nonzero():
    """Test that non-zero tensors still work correctly"""
    if not torch.cuda.is_available():
        return True  # Skip
    
    print("\n🔍 Testing quantizer with non-zero input...")
    
    # Create a tensor with known range: [-10, +10]
    torch.manual_seed(42)
    tensor = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16) * 10.0
    
    encoded, scales = quantize_sim_fp8_per_head(tensor)
    
    # Scales should be ~= abs_max / 448.0
    abs_max = tensor.abs().amax(dim=(0, 2, 3))
    expected_scales = abs_max / 448.0
    
    close = torch.allclose(scales.cpu(), expected_scales.cpu(), rtol=1e-2)
    print(f"   Check: scales match abs_max/448.0? {close}")
    if not close:
        print(f"      ⚠️  scales: {scales.cpu().tolist()}")
        print(f"      ⚠️  expected: {expected_scales.cpu().tolist()}")
        return False
    
    print(f"   ✅ Non-zero quantization still works correctly")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("PRIORITY 1.1: Quantizer Scale Bug Fix Validation")
    print("=" * 70)
    
    test1_pass = test_quantizer_zero_fix()
    test2_pass = test_quantizer_nonzero()
    
    print("\n" + "=" * 70)
    if test1_pass and test2_pass:
        print("✅ VERDICT: Quantizer fix SUCCESSFUL")
        print("   - Zero tensors → scale=1.0 ✓")
        print("   - Non-zero tensors → scale=abs_max/448.0 ✓")
        print("=" * 70)
        sys.exit(0)
    else:
        print("❌ VERDICT: Quantizer fix FAILED")
        print("=" * 70)
        sys.exit(1)

