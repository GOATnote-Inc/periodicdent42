#!/usr/bin/env python3
"""
V3 Tile-Level Oracle: Test Single Tile (First BLOCK_M x BLOCK_N) Against SDPA
Goal: Identify if S, P, or O diverges first (localize bug)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
import numpy as np

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import V3 build function
from build_v3_release import build_v3_release

# ============================================================================
# Configuration
# ============================================================================

# V3 operates on S=512, but we can test with small batch/head
# to isolate a single tile's behavior
B, H, S, D = 1, 1, 512, 64  # V3 specialized for S=512
SEED = 42
DEVICE = "cuda"
DTYPE = torch.float16

# Output directory
OUT_DIR = Path(__file__).parent.parent.parent.parent / "artifacts" / "oracle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Golden Oracle (SDPA)
# ============================================================================

def compute_golden_oracle(Q, K, V, is_causal=False):
    """
    Compute SDPA reference in float32 for precision
    
    Returns:
        torch.Tensor: O_ref [B, H, S, D]
    """
    softmax_scale = 1.0 / (D ** 0.5)
    
    with torch.no_grad():
        O_ref = F.scaled_dot_product_attention(
            Q.float(),
            K.float(),
            V.float(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=softmax_scale
        )
    
    return O_ref


# ============================================================================
# V3 Kernel Test
# ============================================================================

def test_v3_oracle(Q, K, V, is_causal=False, config_idx=0):
    """
    Test V3 kernel on S=512
    
    Args:
        Q, K, V: Input tensors [B, H, S, D]
        is_causal: Causal mask flag
        config_idx: Which V3 config to test (0, 1, or 2)
    
    Returns:
        dict with O_v3 and metadata
    """
    
    print("\n" + "=" * 80)
    print(f"V3 Tile Oracle Test (Config {config_idx})")
    print("=" * 80)
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    try:
        # Build and load V3
        v3_module = build_v3_release()
        
        # Select config
        if config_idx == 0:
            v3_forward = v3_module.forward_32_64_4_2_1_1
            config_name = "32_64_4_2_1_1 (BLOCK_M=32, BLOCK_N=64, WARPS=4)"
        elif config_idx == 1:
            v3_forward = v3_module.forward_32_32_4_2_1_1
            config_name = "32_32_4_2_1_1 (BLOCK_M=32, BLOCK_N=32, WARPS=4)"
        elif config_idx == 2:
            v3_forward = v3_module.forward_48_64_8_2_1_1
            config_name = "48_64_8_2_1_1 (BLOCK_M=48, BLOCK_N=64, WARPS=8)"
        else:
            raise ValueError(f"Invalid config_idx: {config_idx}")
        
        print(f"  Config: {config_name}")
        print(f"  Input: B={B}, H={H}, S={S}, D={D}, causal={is_causal}")
        
        # Run V3
        O_v3 = v3_forward(Q, K, V, softmax_scale, is_causal)
        
        # Check for NaN/Inf
        has_nan = torch.isnan(O_v3).any().item()
        has_inf = torch.isinf(O_v3).any().item()
        
        if has_nan or has_inf:
            print(f"  ❌ Output has NaN={has_nan}, Inf={has_inf}")
            print(f"     NaN count: {torch.isnan(O_v3).sum().item()}/{O_v3.numel()}")
            print(f"     Inf count: {torch.isinf(O_v3).sum().item()}/{O_v3.numel()}")
            
            # Find first NaN/Inf position
            if has_nan:
                nan_mask = torch.isnan(O_v3)
                first_nan = torch.nonzero(nan_mask)[0]
                print(f"     First NaN at: {first_nan.tolist()}")
            
            if has_inf:
                inf_mask = torch.isinf(O_v3)
                first_inf = torch.nonzero(inf_mask)[0]
                print(f"     First Inf at: {first_inf.tolist()}")
        else:
            print(f"  ✅ Output has no NaN/Inf")
        
        # Stats
        O_v3_float = O_v3.float()
        print(f"  Output shape: {O_v3.shape}")
        print(f"  Output range: [{O_v3_float.min():.6f}, {O_v3_float.max():.6f}]")
        print(f"  Output mean: {O_v3_float.mean():.6f}")
        print(f"  Output std: {O_v3_float.std():.6f}")
        
        result = {
            "O": O_v3.squeeze().float().cpu().numpy(),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "config": config_name,
        }
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Comparison and Analysis
# ============================================================================

def compare_outputs(O_ref, O_test, name):
    """
    Compare output tensors and diagnose differences
    
    Returns:
        dict with comparison metrics
    """
    
    if O_test is None:
        print(f"\n  {name}: ⚠️ Not available")
        return None
    
    # Convert to numpy
    if isinstance(O_ref, torch.Tensor):
        O_ref = O_ref.squeeze().float().cpu().numpy()
    if isinstance(O_test, torch.Tensor):
        O_test = O_test.squeeze().float().cpu().numpy()
    
    # Check shapes match
    if O_ref.shape != O_test.shape:
        print(f"\n  {name}: ❌ Shape mismatch!")
        print(f"    Reference: {O_ref.shape}")
        print(f"    Test: {O_test.shape}")
        return None
    
    # Compute differences
    abs_diff = np.abs(O_test - O_ref)
    rel_diff = abs_diff / (np.abs(O_ref) + 1e-8)
    
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    max_rel = rel_diff.max()
    mean_rel = rel_diff.mean()
    
    # Find worst elements
    worst_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
    
    # Tolerance (same as correctness suite)
    passed = max_abs < 0.01
    
    print(f"\n  {name}:")
    print(f"    Max abs diff: {max_abs:.6f} {'✅' if passed else '❌'}")
    print(f"    Mean abs diff: {mean_abs:.6f}")
    print(f"    Max rel diff: {max_rel:.6f}")
    print(f"    Mean rel diff: {mean_rel:.6f}")
    print(f"    Worst at {worst_idx}:")
    print(f"      ref={O_ref[worst_idx]:.6f}, test={O_test[worst_idx]:.6f}")
    
    # Check for systematic errors
    if not passed:
        # Analyze error pattern
        nan_count = np.isnan(O_test).sum()
        inf_count = np.isinf(O_test).sum()
        zero_count = (O_test == 0).sum()
        
        print(f"\n    Error Analysis:")
        print(f"      NaN count: {nan_count}/{O_test.size}")
        print(f"      Inf count: {inf_count}/{O_test.size}")
        print(f"      Zero count: {zero_count}/{O_test.size}")
        
        # Sample worst 5 elements
        flat_diff = abs_diff.flatten()
        worst_5_idx = np.argsort(-flat_diff)[:5]
        print(f"\n    Top 5 worst elements:")
        for i, flat_idx in enumerate(worst_5_idx):
            idx = np.unravel_index(flat_idx, abs_diff.shape)
            ref_val = O_ref[idx]
            test_val = O_test[idx]
            diff = abs_diff[idx]
            print(f"      #{i+1} at {idx}: ref={ref_val:.6f}, test={test_val:.6f}, diff={diff:.6f}")
    
    return {
        "max_abs_diff": float(max_abs),
        "mean_abs_diff": float(mean_abs),
        "max_rel_diff": float(max_rel),
        "mean_rel_diff": float(mean_rel),
        "worst_idx": [int(x) for x in worst_idx],
        "ref_val": float(O_ref[worst_idx]),
        "test_val": float(O_test[worst_idx]),
        "passed": bool(passed),
    }


# ============================================================================
# Main Test
# ============================================================================

def run_v3_oracle_test(is_causal=False, config_idx=0):
    """
    Run V3 oracle test
    
    Returns:
        dict with results
    """
    
    causal_str = "causal" if is_causal else "noncausal"
    
    print("\n" * 2)
    print("=" * 80)
    print(f"V3 ORACLE TEST: B={B}, H={H}, S={S}, D={D}, {causal_str}, config={config_idx}")
    print("=" * 80)
    
    # Create inputs
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    Q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    K = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    V = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    
    print(f"\nInput shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Input dtype: {Q.dtype}")
    
    # Compute golden oracle
    print("\n" + "-" * 80)
    print("Computing Golden Oracle (SDPA)")
    print("-" * 80)
    
    O_ref = compute_golden_oracle(Q, K, V, is_causal)
    
    print(f"  O_ref shape: {O_ref.shape}")
    print(f"  O_ref range: [{O_ref.min():.6f}, {O_ref.max():.6f}]")
    print(f"  O_ref mean: {O_ref.mean():.6f}")
    print(f"  O_ref std: {O_ref.std():.6f}")
    
    # Test V3
    v3_result = test_v3_oracle(Q, K, V, is_causal, config_idx)
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON AGAINST ORACLE")
    print("=" * 80)
    
    results = {
        "shape": {"B": B, "H": H, "S": S, "D": D},
        "is_causal": is_causal,
        "config_idx": config_idx,
        "oracle": {
            "O_range": [float(O_ref.min()), float(O_ref.max())],
            "O_mean": float(O_ref.mean()),
            "O_std": float(O_ref.std()),
        }
    }
    
    if v3_result:
        comparison = compare_outputs(O_ref, v3_result['O'], "V3 vs Oracle")
        
        if comparison:
            results['v3'] = {
                "comparison": comparison,
                "has_nan": v3_result['has_nan'],
                "has_inf": v3_result['has_inf'],
                "config": v3_result['config'],
            }
            
            print(f"\n  V3 Overall: {'✅ PASSED' if comparison['passed'] else '❌ FAILED'}")
        else:
            results['v3'] = {
                "error": "Comparison failed",
                "has_nan": v3_result.get('has_nan', True),
                "has_inf": v3_result.get('has_inf', True),
            }
    else:
        results['v3'] = {"error": "V3 execution failed"}
    
    # Save results
    out_file = OUT_DIR / causal_str / f"v3_oracle_config{config_idx}_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save numpy arrays for deeper analysis
    np.save(OUT_DIR / causal_str / f"v3_config{config_idx}_O_ref.npy", O_ref.squeeze().float().cpu().numpy())
    if v3_result and 'O' in v3_result:
        np.save(OUT_DIR / causal_str / f"v3_config{config_idx}_O_test.npy", v3_result['O'])
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {out_file}")
    print(f"{'=' * 80}\n")
    
    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V3 Tile Oracle Test")
    parser.add_argument("--config", type=int, default=0, choices=[0, 1, 2],
                        help="V3 config to test (0, 1, or 2)")
    parser.add_argument("--causal", action="store_true",
                        help="Test with causal mask")
    parser.add_argument("--noncausal", action="store_true",
                        help="Test without causal mask")
    parser.add_argument("--both", action="store_true",
                        help="Test both causal and non-causal")
    
    args = parser.parse_args()
    
    # Determine which tests to run
    test_cases = []
    if args.both:
        test_cases = [False, True]
    elif args.causal:
        test_cases = [True]
    elif args.noncausal:
        test_cases = [False]
    else:
        # Default: test non-causal only
        test_cases = [False]
    
    # Run tests
    all_passed = True
    for is_causal in test_cases:
        result = run_v3_oracle_test(is_causal=is_causal, config_idx=args.config)
        
        if 'v3' in result and 'comparison' in result['v3']:
            if not result['v3']['comparison']['passed']:
                all_passed = False
        else:
            all_passed = False
    
    # Exit code
    sys.exit(0 if all_passed else 1)

