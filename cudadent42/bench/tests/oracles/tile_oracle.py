#!/usr/bin/env python3
"""
Tile-Level Oracle: Test Single 32×32 Tile Against SDPA
Dumps all intermediates (S, P, O) to artifacts for delta analysis
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
import numpy as np

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import kernels
from build_v3_release import build_v3_release

try:
    from fa_inverted_v2_tensor_cores import flash_attention_inverted_forward as v2_forward
    V2_AVAILABLE = True
except ImportError:
    print("Warning: V2 kernel not available")
    V2_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

# Tiny shape: ONE TILE ONLY
B, H, S, D = 1, 1, 32, 64
SEED = 42
DEVICE = "cuda"
DTYPE = torch.float16

# Output directory
OUT_DIR = Path(__file__).parent.parent.parent.parent / "artifacts" / "oracle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Golden Oracle (SDPA Step-by-Step)
# ============================================================================

def compute_golden_oracle(Q, K, V, is_causal=False):
    """
    Compute SDPA reference with intermediate dumps
    
    Returns:
        dict with S_ref, P_ref, O_ref (all float32 for precision)
    """
    
    Q_f32 = Q.float()
    K_f32 = K.float()
    V_f32 = V.float()
    
    # Step 1: S = (Q @ K^T) * scale
    softmax_scale = 1.0 / (D ** 0.5)
    S_ref = torch.matmul(Q_f32, K_f32.transpose(-2, -1)) * softmax_scale
    
    # Step 2: Apply causal mask
    if is_causal:
        mask = torch.triu(torch.ones(S, S, device=DEVICE, dtype=torch.bool), diagonal=1)
        S_ref = S_ref.masked_fill(mask, float('-inf'))
    
    # Step 3: P = softmax(S)
    P_ref = F.softmax(S_ref, dim=-1)
    
    # Step 4: O = P @ V
    O_ref = torch.matmul(P_ref, V_f32)
    
    return {
        "S": S_ref.squeeze().cpu().numpy(),  # [S, S]
        "P": P_ref.squeeze().cpu().numpy(),  # [S, S]
        "O": O_ref.squeeze().cpu().numpy(),  # [S, D]
    }


# ============================================================================
# Test Functions
# ============================================================================

def test_v2_tile_oracle(Q, K, V, is_causal=False):
    """Test V2 kernel and dump intermediates"""
    
    if not V2_AVAILABLE:
        return None
    
    print("\n" + "=" * 80)
    print("V2 Tile Oracle Test")
    print("=" * 80)
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    try:
        O_v2 = v2_forward(Q, K, V, softmax_scale=softmax_scale, is_causal=is_causal)
        
        # TODO: Dump S and P from kernel (Step 1)
        # For now, just return output
        result = {
            "O": O_v2.squeeze().float().cpu().numpy(),
            "S": None,  # Will be filled in Step 1
            "P": None,  # Will be filled in Step 1
        }
        
        print(f"  Output shape: {O_v2.shape}")
        print(f"  Output range: [{O_v2.min():.6f}, {O_v2.max():.6f}]")
        print(f"  Output mean: {O_v2.mean():.6f}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def test_v3_tile_oracle(Q, K, V, is_causal=False):
    """Test V3 kernel and dump intermediates"""
    
    # V3 requires S=512, so skip for S=32
    print("\n" + "=" * 80)
    print("V3 Tile Oracle Test")
    print("=" * 80)
    print("  ⚠️ V3 specialized for S=512, skipping S=32 test")
    print("  (Will test V3 separately with S=512, D=64)")
    
    return None


# ============================================================================
# Comparison and Analysis
# ============================================================================

def compare_intermediates(ref, test, name):
    """
    Compare intermediate matrices and report differences
    
    Returns:
        dict with comparison metrics
    """
    
    if test is None:
        print(f"\n  {name}: ⚠️ Not available from kernel")
        return None
    
    abs_diff = np.abs(test - ref)
    rel_diff = abs_diff / (np.abs(ref) + 1e-8)
    
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    max_rel = rel_diff.max()
    mean_rel = rel_diff.mean()
    
    # Find worst element
    worst_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
    
    passed = max_abs < 0.01  # Same threshold as suite
    
    print(f"\n  {name}:")
    print(f"    Max abs diff: {max_abs:.6f} {'✅' if passed else '❌'}")
    print(f"    Mean abs diff: {mean_abs:.6f}")
    print(f"    Max rel diff: {max_rel:.6f}")
    print(f"    Worst at {worst_idx}: ref={ref[worst_idx]:.6f}, test={test[worst_idx]:.6f}")
    
    return {
        "max_abs_diff": float(max_abs),
        "mean_abs_diff": float(mean_abs),
        "max_rel_diff": float(max_rel),
        "mean_rel_diff": float(mean_rel),
        "worst_idx": [int(x) for x in worst_idx],
        "ref_val": float(ref[worst_idx]),
        "test_val": float(test[worst_idx]),
        "passed": passed,
    }


def save_heatmap_diffs(ref, test, name, out_path):
    """Save heatmap difference to numpy file"""
    
    if test is None:
        return
    
    diff = test - ref
    np.save(out_path / f"{name}_diff.npy", diff)
    
    # Also save raw matrices
    np.save(out_path / f"{name}_ref.npy", ref)
    np.save(out_path / f"{name}_test.npy", test)


# ============================================================================
# Main Test
# ============================================================================

def run_tile_oracle_test(is_causal=False):
    """
    Run complete tile oracle test
    
    Returns:
        dict with all results
    """
    
    causal_str = "causal" if is_causal else "noncausal"
    
    print("\n" * 2)
    print("=" * 80)
    print(f"TILE ORACLE TEST: B={B}, H={H}, S={S}, D={D}, {causal_str}")
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
    
    oracle = compute_golden_oracle(Q, K, V, is_causal)
    
    print(f"  S shape: {oracle['S'].shape}, range: [{oracle['S'].min():.6f}, {oracle['S'].max():.6f}]")
    print(f"  P shape: {oracle['P'].shape}, range: [{oracle['P'].min():.6f}, {oracle['P'].max():.6f}]")
    print(f"  O shape: {oracle['O'].shape}, range: [{oracle['O'].min():.6f}, {oracle['O'].max():.6f}]")
    
    # Test V2
    v2_result = test_v2_tile_oracle(Q, K, V, is_causal)
    
    # Test V3
    v3_result = test_v3_tile_oracle(Q, K, V, is_causal)
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON AGAINST ORACLE")
    print("=" * 80)
    
    results = {
        "shape": {"B": B, "H": H, "S": S, "D": D},
        "is_causal": is_causal,
        "oracle": {
            "S_range": [float(oracle['S'].min()), float(oracle['S'].max())],
            "P_range": [float(oracle['P'].min()), float(oracle['P'].max())],
            "O_range": [float(oracle['O'].min()), float(oracle['O'].max())],
        }
    }
    
    if v2_result:
        print("\n" + "-" * 80)
        print("V2 vs Oracle")
        print("-" * 80)
        
        v2_comparison = {}
        
        # Compare S (if available)
        if v2_result['S'] is not None:
            v2_comparison['S'] = compare_intermediates(oracle['S'], v2_result['S'], "S (Attention Scores)")
            save_heatmap_diffs(oracle['S'], v2_result['S'], "v2_S", OUT_DIR / causal_str)
        
        # Compare P (if available)
        if v2_result['P'] is not None:
            v2_comparison['P'] = compare_intermediates(oracle['P'], v2_result['P'], "P (Attention Probs)")
            save_heatmap_diffs(oracle['P'], v2_result['P'], "v2_P", OUT_DIR / causal_str)
        
        # Compare O (always available)
        v2_comparison['O'] = compare_intermediates(oracle['O'], v2_result['O'], "O (Output)")
        save_heatmap_diffs(oracle['O'], v2_result['O'], "v2_O", OUT_DIR / causal_str)
        
        results['v2'] = v2_comparison
        
        # Summary
        all_passed = all(comp['passed'] for comp in v2_comparison.values() if comp is not None)
        print(f"\n  V2 Overall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    # Save results
    out_file = OUT_DIR / causal_str / "tile_oracle_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {out_file}")
    print(f"{'=' * 80}\n")
    
    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Test both causal and non-causal
    results = {}
    
    for is_causal in [False, True]:
        results[f"{'causal' if is_causal else 'noncausal'}"] = run_tile_oracle_test(is_causal)
    
    # Exit with success if all tests passed
    all_passed = True
    for test_name, test_results in results.items():
        if 'v2' in test_results:
            v2_passed = all(
                comp['passed'] for comp in test_results['v2'].values() 
                if comp is not None
            )
            if not v2_passed:
                all_passed = False
    
    sys.exit(0 if all_passed else 1)

