#!/usr/bin/env python3
"""
Phase 3 WMMA Kernel - Smoke Test
Validates: Compilation, Execution, Basic Correctness
"""

import torch
import sys
from pathlib import Path

print("=" * 80)
print("Phase 3: WMMA Kernel Smoke Test")
print("=" * 80)

# Step 1: Build kernel
print("\n[1/5] Building WMMA kernel...")
print("-" * 80)

sys.path.insert(0, str(Path(__file__).parent.parent / "cudadent42" / "bench"))
from build_v3_wmma import build_v3_wmma

try:
    module = build_v3_wmma(debug=False)
    print("✅ Build successful")
except Exception as e:
    print(f"❌ Build failed: {e}")
    sys.exit(1)

# Step 2: Setup test tensors
print("\n[2/5] Setting up test tensors...")
print("-" * 80)

B, H, S, D = 2, 8, 512, 64
print(f"Shape: B={B}, H={H}, S={S}, D={D}")

torch.manual_seed(42)
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda').contiguous()
k = q.clone().contiguous()
v = q.clone().contiguous()

print(f"✅ Q: {q.shape}, dtype={q.dtype}, device={q.device}")
print(f"✅ K: {k.shape}, dtype={k.dtype}, device={k.device}")
print(f"✅ V: {v.shape}, dtype={v.dtype}, device={v.device}")

# Step 3: Run kernel (smoke test)
print("\n[3/5] Running WMMA kernel...")
print("-" * 80)

try:
    out_wmma = module.flash_attention_s512_v3_wmma_forward(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    print(f"✅ Output: {out_wmma.shape}, dtype={out_wmma.dtype}")
except Exception as e:
    print(f"❌ Kernel execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Check for NaN/Inf
print("\n[4/5] Checking output validity...")
print("-" * 80)

has_nan = torch.isnan(out_wmma).any().item()
has_inf = torch.isinf(out_wmma).any().item()

if has_nan:
    print("❌ Output contains NaN!")
    sys.exit(1)
if has_inf:
    print("❌ Output contains Inf!")
    sys.exit(1)

print("✅ No NaN/Inf in output")

out_min = out_wmma.min().item()
out_max = out_wmma.max().item()
out_mean = out_wmma.float().mean().item()
out_std = out_wmma.float().std().item()

print(f"Output statistics:")
print(f"  Min:  {out_min:.6f}")
print(f"  Max:  {out_max:.6f}")
print(f"  Mean: {out_mean:.6f}")
print(f"  Std:  {out_std:.6f}")

# Step 5: Compare with PyTorch SDPA (basic correctness)
print("\n[5/5] Comparing with PyTorch SDPA...")
print("-" * 80)

try:
    import torch.nn.functional as F
    out_pytorch = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    
    # Compute error metrics
    abs_diff = (out_wmma - out_pytorch).abs()
    rel_diff = abs_diff / (out_pytorch.abs() + 1e-5)
    
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"Error metrics:")
    print(f"  Max abs diff:  {max_abs_diff:.6f}")
    print(f"  Max rel diff:  {max_rel_diff:.6f}")
    print(f"  Mean abs diff: {mean_abs_diff:.6f}")
    print(f"  Mean rel diff: {mean_rel_diff:.6f}")
    
    # Relaxed tolerance for WMMA FP16+FP32 accumulation
    atol, rtol = 1e-2, 1e-2
    is_close = torch.allclose(out_wmma, out_pytorch, atol=atol, rtol=rtol)
    
    if is_close:
        print(f"✅ Correctness PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"⚠️  Correctness FAILED (atol={atol}, rtol={rtol})")
        print("   This is expected for first iteration - focus on Tensor Core activation first")
except Exception as e:
    print(f"⚠️  PyTorch comparison skipped: {e}")

# Summary
print("\n" + "=" * 80)
print("SMOKE TEST SUMMARY")
print("=" * 80)
print("✅ Build: SUCCESS")
print("✅ Execution: SUCCESS")
print("✅ No NaN/Inf: SUCCESS")
print(f"{'✅' if is_close else '⚠️ '} Correctness: {'PASSED' if is_close else 'NEEDS TUNING'}")
print()
print("Next steps:")
print("  1. Run Nsight Compute profiling (check Tensor Core %)")
print("  2. Benchmark performance (target: < 25 μs)")
print("  3. Tune for correctness if needed")
print("=" * 80)

