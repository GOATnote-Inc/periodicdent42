#!/usr/bin/env python3
"""
SDPA-Oracled Correctness Test Suite
Tests V2 and V3 kernels against PyTorch SDPA (reference oracle)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json

# Import kernels
sys.path.insert(0, str(Path(__file__).parent.parent))
from build_v3_release import build_v3_release

# Import V2
try:
    from fa_inverted_v2_tensor_cores import flash_attention_inverted_forward as v2_forward
    V2_AVAILABLE = True
except ImportError:
    print("Warning: V2 kernel not available")
    V2_AVAILABLE = False


# ============================================================================
# Test Configuration
# ============================================================================

TEST_SHAPES = [
    # (B, S, H, D)
    (1, 128, 4, 64),
    (1, 256, 8, 64),
    (2, 512, 8, 64),
    (4, 512, 8, 64),
    (1, 512, 16, 64),
    (2, 256, 8, 64),
    (2, 128, 8, 64),
]

ATOL = 1e-2  # Absolute tolerance
RTOL = 5e-2  # Relative tolerance

SEED = 42


# ============================================================================
# Test Functions
# ============================================================================

def run_single_test(
    kernel_name,
    kernel_fn,
    B, H, S, D,
    is_causal,
    device="cuda",
    dtype=torch.float16
):
    """
    Run a single correctness test against SDPA
    
    Returns:
        dict with test results
    """
    
    # Create inputs
    torch.manual_seed(SEED)
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    # Reference: PyTorch SDPA
    with torch.no_grad():
        O_ref = F.scaled_dot_product_attention(
            Q, K, V, 
            is_causal=is_causal,
            scale=softmax_scale
        )
    
    # Test kernel
    try:
        with torch.no_grad():
            O_test = kernel_fn(Q, K, V, softmax_scale, is_causal)
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "max_abs_diff": None,
            "max_rel_diff": None,
            "mean_abs_diff": None,
            "mean_rel_diff": None,
            "first_bad_idx": None,
        }
    
    # Compare
    abs_diff = (O_test - O_ref).abs()
    rel_diff = abs_diff / (O_ref.abs() + 1e-8)
    
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Check pass/fail
    passed = (max_abs_diff <= ATOL and max_rel_diff <= RTOL)
    
    # Find first bad index if failed
    first_bad_idx = None
    if not passed:
        mask = (abs_diff > ATOL) | (rel_diff > RTOL)
        if mask.any():
            flat_idx = mask.flatten().nonzero()[0].item()
            unraveled = torch.unravel_index(torch.tensor(flat_idx), O_ref.shape)
            multi_idx = tuple(int(x) for x in unraveled)
            
            first_bad_idx = {
                "flat": flat_idx,
                "multi": multi_idx,
                "ref_val": O_ref.flatten()[flat_idx].item(),
                "test_val": O_test.flatten()[flat_idx].item(),
                "abs_diff": abs_diff.flatten()[flat_idx].item(),
                "rel_diff": rel_diff.flatten()[flat_idx].item(),
            }
    
    result = {
        "passed": passed,
        "error": None,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_rel_diff": mean_rel_diff,
        "first_bad_idx": first_bad_idx,
    }
    
    return result


def test_kernel_all_shapes(kernel_name, kernel_fn):
    """
    Test a kernel against all shapes and both causal/non-causal
    
    Returns:
        dict with all test results
    """
    
    print(f"\n{'=' * 80}")
    print(f"Testing {kernel_name}")
    print(f"{'=' * 80}\n")
    
    all_results = {}
    all_passed = True
    
    for B, S, H, D in TEST_SHAPES:
        for is_causal in [False, True]:
            test_name = f"B{B}_S{S}_H{H}_D{D}_{'causal' if is_causal else 'noncausal'}"
            
            print(f"  {test_name}...", end=" ")
            
            result = run_single_test(
                kernel_name, kernel_fn,
                B, H, S, D, is_causal
            )
            
            all_results[test_name] = result
            
            if result["passed"]:
                print("✅ PASS")
            else:
                print(f"❌ FAIL (max_abs={result['max_abs_diff']:.6f}, max_rel={result['max_rel_diff']:.6f})")
                if result["first_bad_idx"]:
                    idx = result["first_bad_idx"]
                    print(f"    First bad: {idx['multi']}, ref={idx['ref_val']:.6f}, test={idx['test_val']:.6f}")
                if result["error"]:
                    print(f"    Error: {result['error']}")
                all_passed = False
    
    # Summary
    total = len(all_results)
    passed = sum(1 for r in all_results.values() if r["passed"])
    
    print(f"\n{'=' * 80}")
    print(f"{kernel_name} Summary: {passed}/{total} tests passed")
    print(f"{'=' * 80}\n")
    
    return {
        "kernel": kernel_name,
        "all_passed": all_passed,
        "passed_count": passed,
        "total_count": total,
        "tests": all_results,
    }


# ============================================================================
# Kernel Wrappers
# ============================================================================

def v2_wrapper(Q, K, V, softmax_scale, is_causal):
    """Wrapper for V2 kernel to match expected signature"""
    return v2_forward(Q, K, V, softmax_scale=softmax_scale, is_causal=is_causal)


def v3_wrapper(Q, K, V, softmax_scale, is_causal):
    """Wrapper for V3 kernel to match expected signature"""
    # V3 only supports S=512, D=64
    B, H, S, D = Q.shape
    if S != 512 or D != 64:
        raise ValueError(f"V3 only supports S=512, D=64, got S={S}, D={D}")
    
    module = build_v3_release()
    config_id = 1  # Config 1: BLOCK_M=32, BLOCK_N=64, NUM_WARPS=4
    return module.forward(Q, K, V, softmax_scale, is_causal, config_id)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Output directory
    out_dir = Path(__file__).parent.parent.parent.parent / "artifacts" / "correctness"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Test V2
    if V2_AVAILABLE:
        v2_results = test_kernel_all_shapes("V2", v2_wrapper)
        all_results["v2"] = v2_results
        
        # Save V2 results
        with open(out_dir / "v2_correctness.json", "w") as f:
            json.dump(v2_results, f, indent=2)
    else:
        print("⚠️ V2 kernel not available, skipping")
    
    # Test V3 (only for S=512, D=64 shapes)
    v3_shapes = [(B, S, H, D) for (B, S, H, D) in TEST_SHAPES if S == 512 and D == 64]
    
    if v3_shapes:
        print(f"\nNote: V3 kernel specialized for S=512, D=64")
        print(f"Testing {len(v3_shapes)} compatible shapes out of {len(TEST_SHAPES)} total")
        
        # Temporarily override TEST_SHAPES for V3
        original_shapes = TEST_SHAPES
        TEST_SHAPES = v3_shapes
        
        v3_results = test_kernel_all_shapes("V3", v3_wrapper)
        all_results["v3"] = v3_results
        
        # Restore original shapes
        TEST_SHAPES = original_shapes
        
        # Save V3 results
        with open(out_dir / "v3_correctness.json", "w") as f:
            json.dump(v3_results, f, indent=2)
    else:
        print("\n⚠️ No compatible shapes for V3 (requires S=512, D=64)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for kernel_name, results in all_results.items():
        status = "✅ ALL PASS" if results["all_passed"] else "❌ SOME FAILED"
        print(f"{kernel_name.upper()}: {results['passed_count']}/{results['total_count']} {status}")
    
    # Exit with appropriate code
    all_passed = all(r["all_passed"] for r in all_results.values())
    sys.exit(0 if all_passed else 1)

