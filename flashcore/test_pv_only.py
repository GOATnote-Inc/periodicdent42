#!/usr/bin/env python3
"""
FlashCore P@V-Only Test
Tests P@V with UNIFORM attention (bypassing softmax) to isolate P@V bugs
"""

import torch
import sys

print("=" * 80)
print("FlashCore P@V-Only Test (Uniform Attention)")
print("=" * 80)

# Import build function
import importlib.util
spec = importlib.util.spec_from_file_location("build_fused", "build_fused.py")
build_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_module)
build_fused = build_module.build_fused

print("\nBuilding with DEBUG_PV_ONLY=1...")
ext = build_fused(extra_cflags=['-DDEBUG_PV_ONLY=1'])

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

# Compute reference with UNIFORM attention
# P[i,j] = 1/S for all i,j
# O = P @ V = (1/S) * sum(V, dim=-2)
P_uniform = torch.ones(B, H, S, S, device=device, dtype=dtype) / S
O_ref = torch.matmul(P_uniform, V)  # [B, H, S, D]

print("PyTorch uniform attention reference computed")

# Run our kernel (DEBUG_PV_ONLY mode)
scale = 1.0 / (D ** 0.5)
O_ours = ext.forward(Q, K, V, scale)

# Compute error
diff = (O_ours.float() - O_ref.float()).abs()
max_err = diff.max().item()
mean_err = diff.mean().item()

print("\n" + "=" * 80)
print("P@V-ONLY CORRECTNESS TEST (Uniform Attention)")
print("=" * 80)
print(f"  max_err:  {max_err:.4f}")
print(f"  mean_err: {mean_err:.4f}")
print(f"  Target:   max_err < 0.05")

if max_err < 0.05:
    print(f"  Status:   ✅ PASS - P@V is CORRECT!")
    print("\n🐛 Bug is in SOFTMAX computation")
else:
    print(f"  Status:   ❌ FAIL - P@V has errors")
    print("\n🐛 Bug is in P@V accumulation (atomicAdd or WMMA mapping)")
    
    # Show sample values
    print("\nSample output values (first head, first 4 queries, first 4 dims):")
    print("Reference:")
    print(O_ref[0, 0, :4, :4].cpu().numpy())
    print("\nOurs:")
    print(O_ours[0, 0, :4, :4].cpu().numpy())
    print("\nDiff:")
    print(diff[0, 0, :4, :4].cpu().numpy())

print("=" * 80)

sys.exit(0 if max_err < 0.05 else 1)

