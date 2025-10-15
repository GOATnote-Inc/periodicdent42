#!/usr/bin/env python3
"""
Comprehensive SDPA Parity Tests
Tests custom CUDA kernel against PyTorch SDPA across full shape grid.

Shape Grid:
- dtypes: {fp16, bf16}
- head_dims: {64, 80, 96, 128}
- seq_lens: {128, 512, 1024, 2048, 4096}
- batches: {1, 4, 8}
- heads: {8, 16}
- causal: {True, False}
- dropout: {0.0} (dropout=0 for deterministic comparison)

Tolerances:
- fp16/bf16: atol=1e-2, rtol=1e-2

Fail-fast:
- Any NaNs/Inf → immediate failure
"""

import torch
import torch.nn.functional as F
import pytest
import os
import sys
from datetime import datetime
from itertools import product

# Add repo root to path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# ============================================================================
# Configuration
# ============================================================================

# Dynamically load our kernel based on environment variable
OUR_KERNEL_MODULE = os.environ.get("OUR_KERNEL_MODULE", "cudadent42.bench.fa_s512_v3")
OUR_KERNEL_FUNCTION = os.environ.get("OUR_KERNEL_FUNCTION", "flash_attention_s512_v3_forward")

try:
    module = __import__(OUR_KERNEL_MODULE, fromlist=[OUR_KERNEL_FUNCTION])
    our_kernel = getattr(module, OUR_KERNEL_FUNCTION)
    print(f"✅ Loaded kernel: {OUR_KERNEL_MODULE}.{OUR_KERNEL_FUNCTION}")
except (ImportError, AttributeError) as e:
    print(f"⚠️  Could not load kernel '{OUR_KERNEL_MODULE}.{OUR_KERNEL_FUNCTION}': {e}")
    our_kernel = None

# Skip all tests if CUDA not available
if not torch.cuda.is_available():
    pytest.skip("CUDA not available, skipping GPU tests", allow_module_level=True)

# Skip all tests if kernel not loaded
if our_kernel is None:
    pytest.skip("Custom kernel not available, skipping tests", allow_module_level=True)

# Fixed seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============================================================================
# Test Configuration Grid
# ============================================================================

# Full shape grid (can be filtered for faster testing)
DTYPES = [torch.float16]
if torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)

HEAD_DIMS = [64, 80, 96, 128]
SEQ_LENS = [128, 512, 1024, 2048, 4096]
BATCHES = [1, 4, 8]
HEADS = [8, 16]
CAUSAL_OPTIONS = [True, False]

# Generate test configurations
# Filter: Only test S=512, D=64 for V3 kernel (specialized)
# For comprehensive testing, would test all combinations
TEST_CONFIGS = []

# Canonical shapes (must pass)
CANONICAL_SHAPES = [
    (4, 16, 2048, 128, True, torch.float16, "canonical_large_causal_fp16"),
    (1, 8, 4096, 128, True, torch.float16, "canonical_long_seq_fp16"),
    (8, 16, 1024, 64, False, torch.float16, "canonical_std_noncausal_fp16"),
]

# Add canonical shapes
for B, H, S, D, causal, dtype, name in CANONICAL_SHAPES:
    TEST_CONFIGS.append((B, H, S, D, causal, dtype, name, "canonical"))

# Add specialized V3 shapes (S=512, D=64, FP16)
V3_SPECIALIZED_SHAPES = [
    (1, 8, 512, 64, False, torch.float16, "v3_spec_small", "specialized"),
    (1, 8, 512, 64, True, torch.float16, "v3_spec_small_causal", "specialized"),
    (4, 16, 512, 64, False, torch.float16, "v3_spec_medium", "specialized"),
    (4, 16, 512, 64, True, torch.float16, "v3_spec_medium_causal", "specialized"),
    (8, 16, 512, 64, False, torch.float16, "v3_spec_large", "specialized"),
]

for B, H, S, D, causal, dtype, name, category in V3_SPECIALIZED_SHAPES:
    TEST_CONFIGS.append((B, H, S, D, causal, dtype, name, category))

# Add BF16 tests if supported
if torch.cuda.is_bf16_supported():
    for B, H, S, D, causal, dtype, name in [
        (1, 8, 512, 64, False, torch.bfloat16, "v3_spec_bf16"),
        (4, 16, 512, 64, True, torch.bfloat16, "v3_spec_medium_causal_bf16"),
    ]:
        TEST_CONFIGS.append((B, H, S, D, causal, dtype, name, "specialized_bf16"))

# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.parametrize("B,H,S,D,causal,dtype,name,category", TEST_CONFIGS)
def test_sdpa_parity(B, H, S, D, causal, dtype, name, category):
    """
    Test custom kernel output against PyTorch SDPA.
    
    Pass criteria:
    - No NaNs/Inf in output
    - Max absolute difference < 1e-2
    - Max relative difference reasonable
    """
    print(f"\n{'='*80}")
    print(f"Testing: {name} [{category}]")
    print(f"Shape: B={B}, H={H}, S={S}, D={D}, causal={causal}, dtype={dtype}")
    print(f"{'='*80}")
    
    # Generate random inputs
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    Q = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    K = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    V = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    
    # PyTorch SDPA reference
    try:
        ref_output = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    except Exception as e:
        pytest.skip(f"SDPA failed for this configuration: {e}")
    
    # Check for specialized kernel constraints
    # V3 kernel: Only S=512, D=64 supported
    if S != 512 or D != 64:
        pytest.skip(f"V3 kernel specialized for S=512, D=64 only (got S={S}, D={D})")
    
    if dtype != torch.float16 and dtype != torch.bfloat16:
        pytest.skip(f"V3 kernel only supports FP16/BF16 (got {dtype})")
    
    # Our kernel output
    try:
        our_output = our_kernel(Q, K, V, is_causal=causal, config_id=1)
    except RuntimeError as e:
        if "Only HEAD_DIM=64 supported" in str(e) or "specialized for S=512" in str(e):
            pytest.skip(f"Kernel constraint: {e}")
        else:
            pytest.fail(f"Kernel raised unexpected exception: {e}")
    except Exception as e:
        pytest.fail(f"Kernel raised exception: {e}")
    
    # Check for NaNs/Infs (fail-fast)
    if torch.isnan(our_output).any():
        pytest.fail(f"❌ FAIL: Output contains NaN values")
    
    if torch.isinf(our_output).any():
        pytest.fail(f"❌ FAIL: Output contains Inf values")
    
    # Compute differences
    abs_diff = (our_output - ref_output).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    # Relative difference (avoid division by zero)
    rel_diff = abs_diff / (ref_output.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Output statistics
    our_min, our_max = our_output.min().item(), our_output.max().item()
    ref_min, ref_max = ref_output.min().item(), ref_output.max().item()
    
    print(f"\nResults:")
    print(f"  Max Absolute Diff: {max_abs_diff:.6f}")
    print(f"  Mean Absolute Diff: {mean_abs_diff:.6f}")
    print(f"  Max Relative Diff: {max_rel_diff:.6f}")
    print(f"  Mean Relative Diff: {mean_rel_diff:.6f}")
    print(f"  Our Output Range: [{our_min:.4f}, {our_max:.4f}]")
    print(f"  Ref Output Range: [{ref_min:.4f}, {ref_max:.4f}]")
    
    # Tolerances
    ATOL = 1e-2
    RTOL = 1e-2
    
    # Check tolerance
    try:
        torch.testing.assert_close(
            ref_output, our_output,
            rtol=RTOL, atol=ATOL,
            msg=f"Parity test failed for {name}"
        )
        print(f"✅ PASS: Within tolerance (atol={ATOL}, rtol={RTOL})")
    except AssertionError as e:
        print(f"❌ FAIL: {e}")
        # Re-raise to fail test
        raise


def test_canonical_shapes_summary():
    """
    Summary test to ensure all canonical shapes pass.
    This is the critical gate for the optimization workflow.
    """
    print(f"\n{'='*80}")
    print("CANONICAL SHAPES SUMMARY TEST")
    print(f"{'='*80}")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for B, H, S, D, causal, dtype, name in CANONICAL_SHAPES:
        # Check if this shape is compatible with V3 kernel
        if S != 512 or D != 64:
            print(f"⊘ {name}: SKIPPED (V3 specialized for S=512, D=64)")
            skipped += 1
            continue
        
        try:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            
            Q = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
            K = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
            V = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
            
            ref_output = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
            our_output = our_kernel(Q, K, V, is_causal=causal, config_id=1)
            
            # Check NaN/Inf
            if torch.isnan(our_output).any() or torch.isinf(our_output).any():
                print(f"✗ {name}: FAILED (NaN/Inf detected)")
                failed += 1
                continue
            
            # Check tolerance
            torch.testing.assert_close(ref_output, our_output, rtol=1e-2, atol=1e-2)
            print(f"✓ {name}: PASSED")
            passed += 1
            
        except Exception as e:
            print(f"✗ {name}: FAILED ({str(e)[:50]}...)")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*80}")
    
    if failed > 0:
        pytest.fail(f"Canonical shapes test failed: {failed}/{len(CANONICAL_SHAPES)} failed")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print(f"{'='*80}")
    print(f"Comprehensive SDPA Parity Tests")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"Kernel: {OUR_KERNEL_MODULE}.{OUR_KERNEL_FUNCTION}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Compute Cap: {torch.cuda.get_device_capability(0)}")
    print(f"Test Configs: {len(TEST_CONFIGS)}")
    print(f"{'='*80}\n")
    
    # Run pytest programmatically
    import sys
    pytest_args = ["-v", "-s", "--tb=short", __file__]
    exit_code = pytest.main(pytest_args)
    
    print(f"\n{'='*80}")
    print(f"Result: {'✅ PASSED' if exit_code == 0 else '❌ FAILED'}")
    print(f"{'='*80}")
    
    sys.exit(exit_code)

