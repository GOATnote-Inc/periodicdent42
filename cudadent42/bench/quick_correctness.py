#!/usr/bin/env python3
"""
Quick correctness check: V3 vs PyTorch SDPA
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Import V3 release build
sys.path.insert(0, str(Path(__file__).parent))
from build_v3_release import build_v3_release


def run_correctness_test(B, H, S, D, is_causal=False, atol=1e-2, rtol=5e-2):
    """
    Test V3 kernel against PyTorch SDPA
    
    Args:
        B: Batch size
        H: Number of heads
        S: Sequence length (must be 512 for V3)
        D: Head dimension (must be 64 for V3)
        is_causal: Whether to use causal masking
        atol: Absolute tolerance
        rtol: Relative tolerance
    
    Returns:
        dict with test results
    """
    
    assert S == 512, "V3 kernel specialized for S=512"
    assert D == 64, "V3 kernel specialized for D=64"
    
    device = "cuda"
    dtype = torch.float16
    
    print(f"\n{'=' * 80}")
    print(f"Correctness Test: B={B}, H={H}, S={S}, D={D}, causal={is_causal}")
    print(f"{'=' * 80}")
    
    # Create inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    # Reference: PyTorch SDPA
    print("Running PyTorch SDPA (reference)...")
    with torch.no_grad():
        O_ref = F.scaled_dot_product_attention(
            Q, K, V, 
            is_causal=is_causal,
            scale=softmax_scale
        )
    
    # V3 kernel
    print("Running V3 kernel...")
    module = build_v3_release()
    
    config_id = 1  # Config 1: BLOCK_M=32, BLOCK_N=64, NUM_WARPS=4
    
    with torch.no_grad():
        O_v3 = module.forward(Q, K, V, softmax_scale, is_causal, config_id)
    
    # Compare
    print("\nComparing outputs...")
    
    abs_diff = (O_v3 - O_ref).abs()
    rel_diff = abs_diff / (O_ref.abs() + 1e-8)
    
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Check for NaN/Inf
    has_nan = torch.isnan(O_v3).any().item()
    has_inf = torch.isinf(O_v3).any().item()
    
    # Pass/fail
    passed = (
        max_abs_diff <= atol and
        max_rel_diff <= rtol and
        not has_nan and
        not has_inf
    )
    
    result = {
        "shape": {"B": B, "H": H, "S": S, "D": D},
        "is_causal": is_causal,
        "tolerances": {"atol": atol, "rtol": rtol},
        "metrics": {
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mean_abs_diff": mean_abs_diff,
            "mean_rel_diff": mean_rel_diff,
            "has_nan": has_nan,
            "has_inf": has_inf,
        },
        "passed": passed,
    }
    
    print(f"\nResults:")
    print(f"  Max absolute diff: {max_abs_diff:.6f} (threshold: {atol})")
    print(f"  Max relative diff: {max_rel_diff:.6f} (threshold: {rtol})")
    print(f"  Mean absolute diff: {mean_abs_diff:.6f}")
    print(f"  Mean relative diff: {mean_rel_diff:.6f}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    
    return result


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Run both causal and non-causal
    results = {}
    
    for causal in [False, True]:
        key = "causal" if causal else "noncausal"
        results[key] = run_correctness_test(
            B=2, H=8, S=512, D=64,
            is_causal=causal
        )
    
    # Save results
    out_dir = Path(__file__).parent.parent.parent / "artifacts" / "correctness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "v3_quick.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {out_file}")
    print(f"{'=' * 80}")
    
    # Exit with success if all passed
    all_passed = all(r["passed"] for r in results.values())
    sys.exit(0 if all_passed else 1)

