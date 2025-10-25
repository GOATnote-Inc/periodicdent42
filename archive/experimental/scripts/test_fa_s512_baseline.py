#!/usr/bin/env python3
"""
Test existing fa_s512.cu kernel baseline on L4
Verify it still works and measure current performance
"""

import torch
import sys
from pathlib import Path

print("=" * 80)
print("Testing fa_s512.cu Baseline on L4")
print("=" * 80)

# Check if kernel exists
kernel_path = Path("cudadent42/bench/kernels/fa_s512.cu")
if not kernel_path.exists():
    print(f"❌ Kernel not found: {kernel_path}")
    sys.exit(1)

print(f"✅ Kernel found: {kernel_path}")
print()

# Try to import existing bindings
try:
    # Check for existing build
    import importlib.util
    spec = importlib.util.find_spec("flash_attention_s512")
    if spec is None:
        print("⚠️  No existing build found. Need to build first.")
        print()
        print("Run this on GPU instance:")
        print("  cd ~/periodicdent42")
        print("  python3 cudadent42/bench/build_fa_s512.py")
        sys.exit(0)
    
    print("✅ Found existing build, importing...")
    import flash_attention_s512 as fa_s512
    print(f"✅ Module loaded: {fa_s512}")
    
except Exception as e:
    print(f"⚠️  Could not import: {e}")
    print()
    print("This is expected on local machine. Run on GPU instance.")
    sys.exit(0)

print()
print("Setup test tensors...")
B, H, S, D = 2, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda').contiguous()
k, v = q.clone(), q.clone()

print(f"Shape: B={B}, H={H}, S={S}, D={D}")
print()

# Benchmark
print("Benchmarking (100 iterations)...")
times = []
for _ in range(100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fa_s512.forward(q, k, v)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # ms -> μs

times.sort()
p50 = times[50]

print(f"✅ Baseline p50: {p50:.2f} μs")
print()
print("Expected: ~321 μs (from documentation)")
print(f"Actual:   {p50:.2f} μs")
print()

if p50 < 100:
    print("🎉 AMAZING! Kernel is already fast!")
elif p50 < 200:
    print("✅ Good baseline, room for optimization")
else:
    print("⚠️  Slower than expected, investigate")

print()
print("Next: Create EvoEngineer optimization prompt")
