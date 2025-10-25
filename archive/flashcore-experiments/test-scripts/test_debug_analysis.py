#!/usr/bin/env python3
"""
Debug analysis: Compare our kernel's intermediate values with PyTorch reference
to identify where the numerical error originates.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.expanduser('~/flashcore'))

from build_fused import build_fused

def main():
    print("=" * 80)
    print("FlashCore Error Analysis")
    print("=" * 80)
    
    # Build kernel
    print("\n1. Building kernel...")
    ext = build_fused()
    
    # Test shape
    B, H, S, D = 1, 8, 512, 64
    device = torch.device('cuda')
    dtype = torch.float16
    
    # Create test inputs (deterministic)
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    scale = 1.0 / (D ** 0.5)
    
    # Reference: PyTorch SDPA
    print("\n2. Computing PyTorch reference...")
    with torch.no_grad():
        O_ref = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
        )
    
    # Our kernel
    print("\n3. Computing our kernel...")
    O_ours = torch.zeros_like(Q)
    with torch.no_grad():
        ext.forward(Q, K, V, O_ours)
    
    # Compute errors
    diff = (O_ours - O_ref).float()
    abs_diff = diff.abs()
    
    print("\n4. Error Analysis:")
    print(f"  max_err:      {abs_diff.max().item():.4f}")
    print(f"  mean_err:     {abs_diff.mean().item():.4f}")
    print(f"  median_err:   {abs_diff.median().item():.4f}")
    print(f"  std_err:      {abs_diff.std().item():.4f}")
    
    # Percentiles
    print(f"\n  Percentiles:")
    for p in [50, 75, 90, 95, 99, 99.9]:
        val = torch.quantile(abs_diff, p/100).item()
        print(f"    {p:5.1f}%: {val:.4f}")
    
    # Per-head analysis
    print(f"\n5. Per-head errors:")
    for h in range(H):
        head_diff = abs_diff[0, h]
        print(f"  Head {h}: max={head_diff.max().item():.4f}, mean={head_diff.mean().item():.4f}")
    
    # Per-row analysis (first 10 rows)
    print(f"\n6. Per-row errors (first 10 rows):")
    for i in range(min(10, S)):
        row_diff = abs_diff[0, 0, i]  # Head 0
        print(f"  Row {i:3d}: max={row_diff.max().item():.4f}, mean={row_diff.mean().item():.4f}")
    
    # Check for patterns
    print(f"\n7. Error patterns:")
    
    # Are errors worse at tile boundaries?
    tile_size = 32
    boundary_mask = torch.zeros(S, dtype=torch.bool, device=device)
    for i in range(0, S, tile_size):
        boundary_mask[max(0, i-2):min(S, i+3)] = True
    
    boundary_err = abs_diff[0, 0, boundary_mask].mean().item()
    interior_err = abs_diff[0, 0, ~boundary_mask].mean().item()
    print(f"  Tile boundary errors: {boundary_err:.4f}")
    print(f"  Tile interior errors: {interior_err:.4f}")
    print(f"  Ratio: {boundary_err / (interior_err + 1e-8):.2f}x")
    
    # Are errors correlated with magnitude?
    ref_mag = O_ref.abs()
    high_mag_mask = ref_mag > ref_mag.median()
    high_mag_err = abs_diff[high_mag_mask].mean().item()
    low_mag_err = abs_diff[~high_mag_mask].mean().item()
    print(f"\n  High magnitude errors: {high_mag_err:.4f}")
    print(f"  Low magnitude errors:  {low_mag_err:.4f}")
    print(f"  Ratio: {high_mag_err / (low_mag_err + 1e-8):.2f}x")
    
    # Sample comparisons
    print(f"\n8. Sample value comparisons (first row, first head, first 8 dims):")
    print(f"  {'Dim':<5} {'Ref':<10} {'Ours':<10} {'Diff':<10}")
    for d in range(8):
        ref_val = O_ref[0, 0, 0, d].item()
        our_val = O_ours[0, 0, 0, d].item()
        diff_val = abs(ref_val - our_val)
        print(f"  {d:<5} {ref_val:<10.4f} {our_val:<10.4f} {diff_val:<10.4f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

