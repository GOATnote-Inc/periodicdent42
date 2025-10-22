#!/usr/bin/env python3
"""
FlashCore QK-Only Debug Test
Isolates Q@K^T computation to identify bug location
"""

import torch
import sys
import os

# Build with DEBUG_QK_ONLY=1
print("=" * 80)
print("FlashCore QK-Only Debug Test")
print("=" * 80)

# Import build function
import importlib.util
spec = importlib.util.spec_from_file_location("build_fused", "build_fused.py")
build_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_module)
build_fused = build_module.build_fused

print("\nBuilding with DEBUG_QK_ONLY=1...")
ext = build_fused(extra_cflags=['-DDEBUG_QK_ONLY=1'])

print("✅ Build successful!")

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

# Compute reference: S = (Q @ K.T) * scale
scale = 1.0 / (D ** 0.5)
Q_scaled = Q * scale
S_ref = torch.matmul(Q_scaled, K.transpose(-2, -1))  # [B, H, S, S]

print("\nPyTorch reference computed")

# Run our kernel (DEBUG_QK_ONLY mode)
# In DEBUG_QK_ONLY mode, kernel stores S[:,:,:32] in O[:,:,:32]
O = ext.forward(Q, K, V, scale)

# Our kernel stores S in the output (only first 32 columns due to tile size)
# Extract the first tile (32 columns)
tile_n = 32
S_ours = O[:, :, :, :tile_n]  # [B, H, S, 32]
S_ref_tile = S_ref[:, :, :, :tile_n]

# Compute error
diff = (S_ours.float() - S_ref_tile.float()).abs()
max_err = diff.max().item()
mean_err = diff.mean().item()

print("\n" + "=" * 80)
print("QK-ONLY CORRECTNESS TEST")
print("=" * 80)
print(f"  Comparing first {tile_n} columns (one tile)")
print(f"  max_err:  {max_err:.4f}")
print(f"  mean_err: {mean_err:.4f}")
print(f"  Target:   max_err < 0.001 (FP16 precision)")

if max_err < 0.001:
    print(f"  Status:   ✅ PASS - Q@K^T is CORRECT!")
    print("\n🎯 Bug is in SOFTMAX or P@V (not in QK)")
else:
    print(f"  Status:   ❌ FAIL - Q@K^T has errors")
    print("\n🐛 Bug is in Q@K^T computation (WMMA load/layout issue)")
    
    # Show sample values for debugging
    print("\nSample values (first query, first 8 keys):")
    print("Reference:", S_ref_tile[0, 0, 0, :8].cpu().numpy())
    print("Ours:     ", S_ours[0, 0, 0, :8].cpu().numpy())
    print("Diff:     ", diff[0, 0, 0, :8].cpu().numpy())

print("=" * 80)

sys.exit(0 if max_err < 0.001 else 1)

