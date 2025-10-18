#!/usr/bin/env python3
"""
FP8 SDPA Coalesced Kernel - Cycle 2a (Fixed)

Coalesced scalar loads optimizations:
1. tid-based indexing → natural hardware coalescing
2. Q dequant once per row/lane (2 elements cached)
3. cp.async for K/V tiles (Ampere/Ada, 1-byte granularity)
4. No alignment requirements (robust!)
5. Numerical robustness (epsilon on l_final)

Target: 600-800 μs (2-2.7× from 1596 μs baseline)
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
    
    # Quantize: [-abs_max, abs_max] → [0, 255]
    t_norm = t / abs_max if abs_max > 1e-8 else t
    t_uint8 = ((t_norm + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    
    return t_uint8, scale

def build_module():
    print("\n" + "="*80)
    print("BUILDING FP8 SDPA COALESCED KERNEL")
    print("="*80)
    print("\nTarget: L4 (sm_89, Ada)")
    print("Optimizations:")
    print("  ✅ Coalesced scalar loads (tid-based, naturally coalesced)")
    print("  ✅ Q dequant once per row/lane (2 elements cached)")
    print("  ✅ cp.async for K/V (1-byte, Ampere/Ada)")
    print("  ✅ No alignment requirements (robust!)")
    print("  ✅ Numerical robustness (epsilon guard)")
    print("\nCompiling sdpa_fp8_coalesced.cu...")
    
    from torch.utils.cpp_extension import load
    
    kernel_dir = Path(__file__).parent.parent / "cudadent42" / "bench" / "kernels"
    
    module = load(
        name="sdpa_fp8_coalesced",
        sources=[
            str(kernel_dir / "sdpa_fp8_coalesced.cu"),
            str(kernel_dir / "sdpa_fp8_coalesced_bindings.cpp"),
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
    print("\nModule: sdpa_fp8_coalesced")
    print("\nNext: Running correctness and performance tests...")
    
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    args = parser.parse_args()
    
    print("="*80)
    print("BENCHMARK: FP8 SDPA COALESCED (Cycle 2a Fixed)")
    print("="*80)
    
    B, H, S, D = 1, 8, 512, 64
    print(f"\nShape: B={B}, H={H}, S={S}, D={D}")
    print("Precision: FP8 E4M3 (inputs), FP32 (compute), FP16 (output)")
    
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
    
    Q_fp8 = torch.stack(Q_fp8_list).unsqueeze(0)  # [1, H, S, D]
    K_fp8 = torch.stack(K_fp8_list).unsqueeze(0)
    V_fp8 = torch.stack(V_fp8_list).unsqueeze(0)
    
    Q_scale_tensor = torch.tensor(Q_scales, dtype=torch.float32, device='cuda')
    K_scale_tensor = torch.tensor(K_scales, dtype=torch.float32, device='cuda')
    V_scale_tensor = torch.tensor(V_scales, dtype=torch.float32, device='cuda')
    
    # Build module
    module = build_module()
    
    # Warmup
    print(f"\nWarmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        out = module.forward(Q_fp8, K_fp8, V_fp8, Q_scale_tensor, K_scale_tensor, V_scale_tensor)
    torch.cuda.synchronize()
    print("  ✅ Done")
    
    # Benchmark
    print(f"\nBenchmarking ({args.iters} iterations)...")
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
    
    # Correctness check (vs dequantized reference)
    print("\nChecking correctness...")
    with torch.no_grad():
        # Dequantize using same formula as CUDA kernel
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
    threshold = 0.1  # FP8 tolerance
    correct = max_diff <= threshold
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Correct: {'✅' if correct else '❌'} (threshold: {threshold})")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS: FP8 SDPA COALESCED (Cycle 2a Fixed)")
    print("="*80)
    print(f"Latency:    {latency_us:.2f} μs")
    print(f"Correct:    {correct}")
    print(f"Max diff:   {max_diff:.6f}")
    
    # vs Baseline
    baseline_us = 1596.75
    speedup_vs_baseline = baseline_us / latency_us
    print(f"\nComparison vs Scalar Baseline ({baseline_us:.2f} μs):")
    if speedup_vs_baseline >= 1.0:
        print(f"  🚀 {speedup_vs_baseline:.2f}× FASTER!")
    else:
        print(f"  ⚠️ {1.0/speedup_vs_baseline:.2f}× slower (regression)")
    
    # vs Champion
    champion_us = 24.22
    print(f"\nComparison vs xFormers Champion ({champion_us} μs):")
    speedup_vs_champion = champion_us / latency_us
    if latency_us < champion_us:
        print(f"  🎉 {speedup_vs_champion:.2f}× FASTER than champion!")
    else:
        print(f"  ⚠️ {latency_us/champion_us:.2f}× slower (gap remaining: {latency_us - champion_us:.2f} μs)")
    
    # vs Target
    target_us = 800.0
    print(f"\nCycle 2a Target: ≤ {target_us:.1f} μs (coalesced loads)")
    if latency_us <= target_us:
        print(f"  ✅ Target met! ({latency_us:.2f} <= {target_us})")
    else:
        print(f"  ⚠️ Above target ({latency_us:.2f} > {target_us})")
    
    # Final verdict
    print("\n" + "="*80)
    if correct and speedup_vs_baseline >= 2.0:
        print("✅ CYCLE 2a SUCCESS - Coalesced loads working!")
        print("="*80)
        print("\nNext steps:")
        print("  ✅ Coalesced loads validated")
        print("  ✅ cp.async integrated")
        if latency_us <= 800:
            print("  ✅ Target met - Ready for final documentation!")
            print(f"\nFinal speedup: {speedup_vs_baseline:.2f}× vs baseline")
        else:
            print("  → Consider further optimizations if time allows")
    else:
        print("⚠️ NEEDS INVESTIGATION")
        print("="*80)
        if not correct:
            print(f"\nCorrectness failed: max_diff={max_diff:.6f} > {threshold}")
        if speedup_vs_baseline < 2.0:
            print(f"\nSpeedup below expected: {speedup_vs_baseline:.2f}× < 2.0×")

if __name__ == "__main__":
    main()

