#!/usr/bin/env python3
"""
FP8 SDPA Stage A - Proper overlap + LUT dequant

Stage A Optimizations:
1. 2-stage cp.async pipeline (double-buffered K/V)
2. 16B cp.async copies (with 8B/4B fallbacks)
3. LUT dequant (256-entry per head) - cheap indexed loads
4. Prefetch+compute ping-pong (real overlap!)

Target: 1.3-1.6× speedup → 900-1100 μs
"""

import torch
import time
import argparse
from pathlib import Path

def quantize_per_tensor(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric per-tensor quantization to simulated FP8 E4M3 (uint8)"""
    fp8_max = 448.0
    abs_max = t.abs().max().item()
    scale = abs_max / fp8_max if abs_max > 1e-8 else 1.0
    
    t_norm = t / abs_max if abs_max > 1e-8 else t
    t_uint8 = ((t_norm + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    
    return t_uint8, scale

def build_module():
    print("\n" + "="*80)
    print("🚀 BUILDING FP8 SDPA STAGE A")
    print("="*80)
    print("\nTarget: L4 (sm_89, Ada)")
    print("Stage A Optimizations:")
    print("  ✅ 2-stage cp.async pipeline (double-buffered K/V)")
    print("  ✅ 16B cp.async (with 8B/4B fallbacks for tails)")
    print("  ✅ LUT dequant (256-entry, cheap indexed loads)")
    print("  ✅ Prefetch+compute ping-pong (TRUE overlap!)")
    print("\nCompiling sdpa_fp8_stage_a.cu...")
    
    from torch.utils.cpp_extension import load
    
    kernel_dir = Path(__file__).parent.parent / "cudadent42" / "bench" / "kernels"
    
    module = load(
        name="sdpa_fp8_stage_a",
        sources=[
            str(kernel_dir / "sdpa_fp8_stage_a.cu"),
            str(kernel_dir / "sdpa_fp8_stage_a_bindings.cpp"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-lineinfo",
            "-Xptxas", "-v",
            "-std=c++17",
            "--generate-code=arch=compute_89,code=sm_89",
            "-DCUDA_ARCH=89",
        ],
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("✅ BUILD SUCCESSFUL")
    print("="*80)
    
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    args = parser.parse_args()
    
    print("="*80)
    print("🧪 CYCLE 3: Stage A - Overlap + LUT Dequant")
    print("="*80)
    
    B, H, S, D = 1, 8, 512, 64
    print(f"\nShape: B={B}, H={H}, S={S}, D={D}")
    
    # Generate test data
    torch.manual_seed(42)
    Q_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Quantize per-head
    print("\nQuantizing to FP8 (per-head)...")
    Q_fp8_list, K_fp8_list, V_fp8_list = [], [], []
    Q_scales, K_scales, V_scales = [], [], []
    
    for h in range(H):
        q_uint8, q_scale = quantize_per_tensor(Q_fp16[0, h])
        k_uint8, k_scale = quantize_per_tensor(K_fp16[0, h])
        v_uint8, v_scale = quantize_per_tensor(V_fp16[0, h])
        
        Q_fp8_list.append(q_uint8)
        K_fp8_list.append(k_uint8)
        V_fp8_list.append(v_uint8)
        
        Q_scales.append(q_scale)
        K_scales.append(k_scale)
        V_scales.append(v_scale)
    
    Q_fp8 = torch.stack(Q_fp8_list).unsqueeze(0)
    K_fp8 = torch.stack(K_fp8_list).unsqueeze(0)
    V_fp8 = torch.stack(V_fp8_list).unsqueeze(0)
    
    Q_scale_tensor = torch.tensor(Q_scales, dtype=torch.float32, device='cuda')
    K_scale_tensor = torch.tensor(K_scales, dtype=torch.float32, device='cuda')
    V_scale_tensor = torch.tensor(V_scales, dtype=torch.float32, device='cuda')
    
    # Build module
    module = build_module()
    
    # Warmup
    print(f"\n🔥 Warmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        out = module.forward(Q_fp8, K_fp8, V_fp8, Q_scale_tensor, K_scale_tensor, V_scale_tensor)
    torch.cuda.synchronize()
    print("  ✅ Done")
    
    # Benchmark
    print(f"\n⏱️  Benchmarking ({args.iters} iterations)...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(args.iters):
        out_fp16 = module.forward(Q_fp8, K_fp8, V_fp8, Q_scale_tensor, K_scale_tensor, V_scale_tensor)
    end.record()
    torch.cuda.synchronize()
    
    latency_ms = start.elapsed_time(end) / args.iters
    latency_us = latency_ms * 1000
    print("  ✅ Done")
    
    # Correctness check
    print("\n🔍 Checking correctness...")
    with torch.no_grad():
        fp8_max = 448.0
        Q_dequant_list, K_dequant_list, V_dequant_list = [], [], []
        for h in range(H):
            q_deq = ((Q_fp8[0, h].float() / 255.0) * (2 * fp8_max) - fp8_max) * Q_scales[h]
            k_deq = ((K_fp8[0, h].float() / 255.0) * (2 * fp8_max) - fp8_max) * K_scales[h]
            v_deq = ((V_fp8[0, h].float() / 255.0) * (2 * fp8_max) - fp8_max) * V_scales[h]
            
            Q_dequant_list.append(q_deq.to(torch.float16))
            K_dequant_list.append(k_deq.to(torch.float16))
            V_dequant_list.append(v_deq.to(torch.float16))
        
        Q_dequant = torch.stack(Q_dequant_list).unsqueeze(0)
        K_dequant = torch.stack(K_dequant_list).unsqueeze(0)
        V_dequant = torch.stack(V_dequant_list).unsqueeze(0)
        
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q_dequant, K_dequant, V_dequant,
            attn_mask=None, dropout_p=0.0, is_causal=False
        )
    
    max_diff = (out_fp16 - ref).abs().max().item()
    threshold = 0.1
    correct = max_diff <= threshold
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Correct: {'✅' if correct else '❌'} (threshold: {threshold})")
    
    # Results
    print("\n" + "="*80)
    print("📊 RESULTS: Stage A")
    print("="*80)
    print(f"Latency:    {latency_us:.2f} μs")
    print(f"Correct:    {correct}")
    print(f"Max diff:   {max_diff:.6f}")
    
    # vs Previous cycles
    baseline_us = 1596.75  # Expert patch
    cycle2a_us = 1453.93   # Coalesced (blocking)
    
    print(f"\n📈 vs Baseline ({baseline_us:.2f} μs):")
    speedup_baseline = baseline_us / latency_us
    print(f"  {speedup_baseline:.2f}× faster")
    
    print(f"\n📈 vs Cycle 2a ({cycle2a_us:.2f} μs, blocking cp.async):")
    speedup_cycle2a = cycle2a_us / latency_us
    print(f"  {speedup_cycle2a:.2f}× faster")
    
    # vs Target
    target_low, target_high = 900, 1100
    print(f"\n🎯 Stage A Target: {target_low}-{target_high} μs (1.3-1.6× improvement)")
    if latency_us <= target_high:
        print(f"  ✅ Target met! ({latency_us:.2f} ≤ {target_high})")
    else:
        print(f"  ⚠️ Above target ({latency_us:.2f} > {target_high})")
    
    # Success criteria
    print("\n" + "="*80)
    if correct and speedup_cycle2a >= 1.3:
        print("✅ CYCLE 3 SUCCESS!")
        print("="*80)
        print("\n🎉 Achievements:")
        print("  ✅ 2-stage pipeline working")
        print("  ✅ LUT dequant validated")
        print("  ✅ Real overlap achieved")
        print(f"  ✅ {speedup_cycle2a:.2f}× speedup over blocking cp.async")
        print("\n🎯 Next: Stage B - Move to Tensor Cores (WMMA)")
        print("  Expected: 3-5× additional speedup → 200-300 μs")
    else:
        if not correct:
            print("❌ CORRECTNESS FAILURE")
            print("="*80)
            print(f"\nmax_diff={max_diff:.6f} > {threshold}")
        else:
            print("⚠️ PERFORMANCE BELOW TARGET")
            print("="*80)
            print(f"\nSpeedup: {speedup_cycle2a:.2f}× < 1.3× target")
            print("→ Need profiling to identify bottleneck")

if __name__ == "__main__":
    main()

