#!/usr/bin/env python3
"""
Phase 0: Verify PyTorch SDPA Baseline on L4 (sm_89)

This script determines the TRUE PyTorch SDPA performance, which sets all subsequent targets.
Expected: 5-10μs (optimal) or 20-50μs (suboptimal)

If < 15μs: Phase 1 target = 20-40μs (2-4× slower)
If > 15μs: Phase 1 target = 100-200μs (2-4× slower)
"""

import torch
import torch.nn.functional as F
import sys

print("=" * 80)
print("Phase 0: PyTorch SDPA Baseline Verification (L4 sm_89)")
print("=" * 80)

# 1. Environment check
print("\n[1/5] Environment Check")
print("-" * 80)
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA not available!")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
print(f"✓ GPU: {device_name}")

if "L4" not in device_name:
    print(f"⚠️  WARNING: Expected 'L4' in GPU name, got '{device_name}'")
    print("   Continuing anyway, but results may not be L4-specific.")

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA version: {torch.version.cuda}")

# 2. Tensor setup
print("\n[2/5] Tensor Setup")
print("-" * 80)
B, H, S, D = 2, 8, 512, 64
print(f"Shape: B={B}, H={H}, S={S}, D={D}")
print(f"Total elements: {B * H * S * D:,}")
print(f"Memory: Q,K,V = {3 * B * H * S * D * 2 / 1024 / 1024:.2f} MB")

torch.manual_seed(42)
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda').contiguous()
k = q.clone().contiguous()
v = q.clone().contiguous()

print(f"✓ Q: {q.shape}, dtype={q.dtype}, device={q.device}, contiguous={q.is_contiguous()}")
print(f"✓ K: {k.shape}, dtype={k.dtype}, device={k.device}, contiguous={k.is_contiguous()}")
print(f"✓ V: {v.shape}, dtype={v.dtype}, device={v.device}, contiguous={v.is_contiguous()}")

# 3. Warm-up (exclude compilation overhead)
print("\n[3/5] Warm-up (10 iterations)")
print("-" * 80)
for i in range(10):
    _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    if i == 0:
        torch.cuda.synchronize()
        print("✓ First iteration complete (compilation done)")
torch.cuda.synchronize()
print("✓ Warm-up complete")

# 4. Benchmark with CUDA events
print("\n[4/5] Benchmark (100 iterations)")
print("-" * 80)

# Try to force Flash Attention backend (may not be available in older PyTorch)
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    print("✓ Using sdpa_kernel to force Flash Attention backend")
    use_backend_context = True
except ImportError:
    print("⚠️  torch.nn.attention not available (PyTorch < 2.0)")
    print("   Using default backend (results may vary)")
    use_backend_context = False

times_us = []

if use_backend_context:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for i in range(100):
            start_event.record()
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            end_event.record()
            torch.cuda.synchronize()
            times_us.append(start_event.elapsed_time(end_event) * 1000)  # ms → μs
            
            if i == 0:
                print(f"✓ First timed iteration: {times_us[0]:.2f} μs")
else:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for i in range(100):
        start_event.record()
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        end_event.record()
        torch.cuda.synchronize()
        times_us.append(start_event.elapsed_time(end_event) * 1000)  # ms → μs
        
        if i == 0:
            print(f"✓ First timed iteration: {times_us[0]:.2f} μs")

print(f"✓ Collected {len(times_us)} timing measurements")

# 5. Statistical analysis
print("\n[5/5] Statistical Analysis")
print("-" * 80)

times_us.sort()
p50 = times_us[50]
p90 = times_us[90]
p99 = times_us[99]
mean = sum(times_us) / len(times_us)
std = (sum((t - mean) ** 2 for t in times_us) / len(times_us)) ** 0.5

print(f"Latency Statistics:")
print(f"  p50:  {p50:7.2f} μs")
print(f"  p90:  {p90:7.2f} μs")
print(f"  p99:  {p99:7.2f} μs")
print(f"  mean: {mean:7.2f} μs")
print(f"  std:  {std:7.2f} μs")

# Theoretical analysis
flops = 4 * B * H * S * S * D  # 2 GEMMs: QK^T and P@V
flops_billions = flops / 1e9
throughput_tflops = flops_billions / (p50 / 1e6) / 1000  # TFLOPS

print(f"\nTheoretical Analysis:")
print(f"  FLOPs:       {flops_billions:.2f} billion (1.07 GFLOPS)")
print(f"  Throughput:  {throughput_tflops:.2f} TFLOPS")

# L4 peak: 242 TFLOPS @ FP16 Tensor Core
l4_peak_tflops = 242
utilization = (throughput_tflops / l4_peak_tflops) * 100
print(f"  Utilization: {utilization:.1f}% of L4 peak (242 TFLOPS @ FP16 TC)")

# 6. Determine targets based on baseline
print("\n" + "=" * 80)
print("RESULTS & TARGET ADJUSTMENT")
print("=" * 80)

print(f"\nPyTorch SDPA Baseline (p50): {p50:.2f} μs")

if p50 < 15:
    print(f"\n🚨 TRUE BASELINE: 5-10μs range (well-optimized)")
    print(f"   Your kernel is competing with production-quality Flash Attention!")
    print(f"\n   ADJUSTED TARGETS:")
    print(f"   - Phase 1 (Scalar):       20-40 μs  (2-4× slower)")
    print(f"   - Phase 2 (Memory):       10-20 μs  (competitive)")
    print(f"   - Phase 3 (Tensor Cores): < 10 μs   (BEAT PyTorch!)")
    print(f"   - Phase 3.5 (L2):         < 8 μs    (faster than PyTorch)")
    print(f"\n   Strategy: Focus on Tensor Cores + L2 to beat PyTorch.")
    baseline_category = "OPTIMAL"
elif p50 < 30:
    print(f"\n⚠️  MODERATE BASELINE: 15-30μs range (suboptimal)")
    print(f"   PyTorch SDPA is not using optimal Flash Attention backend.")
    print(f"\n   ORIGINAL TARGETS:")
    print(f"   - Phase 1 (Scalar):       50-100 μs  (2-4× slower)")
    print(f"   - Phase 2 (Memory):       25-50 μs   (competitive)")
    print(f"   - Phase 3 (Tensor Cores): 12-20 μs   (faster than PyTorch)")
    print(f"   - Phase 3.5 (L2):         10-15 μs   (faster than PyTorch)")
    print(f"\n   Strategy: Stick to roadmap targets, should beat PyTorch easily.")
    baseline_category = "MODERATE"
else:
    print(f"\n✓ BASELINE CONFIRMED: 30-50μs range (unoptimized)")
    print(f"   PyTorch SDPA is using a slow backend or has cold start overhead.")
    print(f"\n   ORIGINAL TARGETS:")
    print(f"   - Phase 1 (Scalar):       100-200 μs  (2-4× slower)")
    print(f"   - Phase 2 (Memory):       50-80 μs    (competitive)")
    print(f"   - Phase 3 (Tensor Cores): 15-25 μs    (2-3× faster)")
    print(f"   - Phase 3.5 (L2):         12-13 μs    (faster than PyTorch)")
    print(f"\n   Strategy: Use roadmap targets as written. Easy wins available.")
    baseline_category = "SLOW"

print(f"\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Baseline Category: {baseline_category}")
print(f"✓ PyTorch SDPA p50:  {p50:.2f} μs")
print(f"✓ Targets adjusted:  See above")
print(f"\nNext: Proceed to Phase 1 (implement minimal scalar kernel)")
print("=" * 80)

# Save results to file
with open("phase0_baseline_results.txt", "w") as f:
    f.write(f"Phase 0: PyTorch SDPA Baseline Results\n")
    f.write(f"=" * 80 + "\n\n")
    f.write(f"GPU: {device_name}\n")
    f.write(f"Shape: B={B}, H={H}, S={S}, D={D}\n\n")
    f.write(f"Baseline p50: {p50:.2f} μs\n")
    f.write(f"Baseline p90: {p90:.2f} μs\n")
    f.write(f"Baseline p99: {p99:.2f} μs\n")
    f.write(f"Mean: {mean:.2f} μs\n")
    f.write(f"Std: {std:.2f} μs\n\n")
    f.write(f"Category: {baseline_category}\n")
    f.write(f"Throughput: {throughput_tflops:.2f} TFLOPS\n")
    f.write(f"Utilization: {utilization:.1f}% of L4 peak\n")

print(f"\n✓ Results saved to: phase0_baseline_results.txt")

