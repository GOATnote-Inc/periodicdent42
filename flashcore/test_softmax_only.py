#!/usr/bin/env python3
"""
FlashCore Softmax-Only Test
Tests Q@K^T → Softmax (without P@V) to isolate softmax bugs
"""

import torch
import sys

print("=" * 80)
print("FlashCore Softmax-Only Test")
print("=" * 80)

# Import build function
import importlib.util
spec = importlib.util.spec_from_file_location("build_fused", "build_fused.py")
build_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_module)
build_fused = build_module.build_fused

print("\nBuilding with DEBUG_SOFTMAX_ONLY=1...")
ext = build_fused(extra_cflags=['-DDEBUG_SOFTMAX_ONLY=1'])

# Test shape
B, H, S, D = 1, 8, 512, 64
device = 'cuda'
dtype = torch.float16

print(f"\nTest shape: B={B}, H={H}, S={S}, D={D}")

# Generate test inputs
torch.manual_seed(42)
Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
K = torch.randn(B, H, S, D, device=device, dtype=dtype)
V = torch.randn(B, H, S, D, device=device, dtype=dtype)

# Compute reference softmax
scale = 1.0 / (D ** 0.5)
Q_scaled = Q * scale
S_scores = torch.matmul(Q_scaled, K.transpose(-2, -1))  # [B, H, S, S]
P_ref = torch.softmax(S_scores, dim=-1)  # [B, H, S, S]

print("PyTorch reference softmax computed")

# Run our kernel (DEBUG_SOFTMAX_ONLY mode)
O = ext.forward(Q, K, V, scale)

# Our kernel stores P[:,:,:32] in O[:,:,:32]
tile_n = 32
P_ours = O[:, :, :, :tile_n]
P_ref_tile = P_ref[:, :, :, :tile_n]

# Compute error
diff = (P_ours.float() - P_ref_tile.float()).abs()
max_err = diff.max().item()
mean_err = diff.mean().item()

# Check row sums (should be ≈ 1.0 for full rows, or fraction for first tile)
row_sums_ref = P_ref[:, :, :, :32].sum(dim=-1)  # Sum over first 32 columns
row_sums_ours = P_ours.sum(dim=-1)

print("\n" + "=" * 80)
print("SOFTMAX-ONLY CORRECTNESS TEST")
print("=" * 80)
print(f"  Comparing first {tile_n} columns (one tile)")
print(f"  max_err:  {max_err:.4f}")
print(f"  mean_err: {mean_err:.4f}")
print(f"  Target:   max_err < 0.001")

# Check row sums
print(f"\n  Row sum check (first tile, should be < 1.0):")
print(f"  Reference: {row_sums_ref[0, 0, :4].cpu().numpy()}")
print(f"  Ours:      {row_sums_ours[0, 0, :4].cpu().numpy()}")

if max_err < 0.001:
    print(f"  Status:   ✅ PASS - Softmax is CORRECT!")
    print("\n🐛 Bug is in P@V or final normalization")
else:
    print(f"  Status:   ❌ FAIL - Softmax has errors")
    print("\n🐛 Bug is in online softmax computation")
    
    # Show sample values
    print("\nSample P values (first query, first 8 keys):")
    print("Reference:", P_ref_tile[0, 0, 0, :8].cpu().numpy())
    print("Ours:     ", P_ours[0, 0, 0, :8].cpu().numpy())
    print("Diff:     ", diff[0, 0, 0, :8].cpu().numpy())

print("=" * 80)

sys.exit(0 if max_err < 0.001 else 1)

