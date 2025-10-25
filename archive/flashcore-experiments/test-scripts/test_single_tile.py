#!/usr/bin/env python3
"""
FlashCore Single Tile Test
Tests with S=32 (only 1 KV tile) to isolate multi-tile accumulation bugs
"""

import torch
import sys

print("=" * 80)
print("FlashCore Single Tile Test (S=32)")
print("=" * 80)

# Import build function
import importlib.util
spec = importlib.util.spec_from_file_location("build_fused", "build_fused.py")
build_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_module)
build_fused = build_module.build_fused

print("\nBuilding kernel...")
ext = build_fused()

# Test with SINGLE KV TILE (S=32, TILE_N=32 → only 1 tile!)
B, H, S, D = 1, 8, 32, 64  # ← S=32 means only 1 KV tile!
device = 'cuda'
dtype = torch.float16

print(f"\nTest shape: B={B}, H={H}, S={S}, D={D}")
print("→ Only 1 KV tile (no multi-tile accumulation)")

# Generate test inputs
torch.manual_seed(42)
Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
K = torch.randn(B, H, S, D, device=device, dtype=dtype)
V = torch.randn(B, H, S, D, device=device, dtype=dtype)

# Compute reference with PyTorch SDPA
scale = 1.0 / (D ** 0.5)
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)

print("PyTorch SDPA reference computed")

# Run our kernel
O_ours = ext.forward(Q, K, V, scale)

# Compute error
diff = (O_ours.float() - O_ref.float()).abs()
max_err = diff.max().item()
mean_err = diff.mean().item()

print("\n" + "=" * 80)
print("SINGLE TILE CORRECTNESS TEST")
print("=" * 80)
print(f"  max_err:  {max_err:.4f}")
print(f"  mean_err: {mean_err:.4f}")
print(f"  Target:   max_err < 0.05")

if max_err < 0.05:
    print(f"  Status:   ✅ PASS - Single tile works!")
    print("\n🎯 Bug is in MULTI-TILE accumulation (m/l/U rescaling)")
    sys.exit(0)
else:
    print(f"  Status:   ❌ FAIL - Bug is in single tile")
    print("\n🐛 Bug is in P@V or final normalization (not multi-tile issue)")
    
    # Show sample values
    print("\nSample output values (first head, first 4 queries, first 4 dims):")
    print("Reference:")
    print(O_ref[0, 0, :4, :4].cpu().numpy())
    print("\nOurs:")
    print(O_ours[0, 0, :4, :4].cpu().numpy())
    print("\nDiff:")
    print(diff[0, 0, :4, :4].cpu().numpy())
    
    sys.exit(1)

