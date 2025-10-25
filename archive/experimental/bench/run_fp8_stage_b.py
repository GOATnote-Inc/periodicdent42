#!/usr/bin/env python3
"""
FP8 SDPA Stage B - FP16 Arithmetic (Tensor Core Prep)

Stage B: FP8→FP16 conversion + FP16 compute
- Convert Q/K/V to FP16 once in SMEM
- Use FP16 arithmetic (faster than FP32 scalar)
- Stepping stone to full WMMA

Target: 2-3× speedup → 500-700 μs
"""

import torch
import time
import argparse
from pathlib import Path

def quantize_per_tensor(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    fp8_max = 448.0
    abs_max = t.abs().max().item()
    scale = abs_max / fp8_max if abs_max > 1e-8 else 1.0
    t_norm = t / abs_max if abs_max > 1e-8 else t
    t_uint8 = ((t_norm + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return t_uint8, scale

def build_module():
    print("\n" + "="*80)
    print("🚀 BUILDING FP8 SDPA STAGE B")
    print("="*80)
    print("\nStage B: FP16 Arithmetic")
    print("  ✅ FP8→FP16 conversion in SMEM (LUT-based)")
    print("  ✅ FP16 dot products (faster than FP32 scalar)")
    print("  ✅ Prep for full WMMA (if needed)")
    print("\nCompiling sdpa_fp8_stage_b.cu...")
    
    from torch.utils.cpp_extension import load
    
    kernel_dir = Path(__file__).parent.parent / "cudadent42" / "bench" / "kernels"
    
    module = load(
        name="sdpa_fp8_stage_b",
        sources=[
            str(kernel_dir / "sdpa_fp8_stage_b.cu"),
            str(kernel_dir / "sdpa_fp8_stage_b_bindings.cpp"),
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    
    print("="*80)
    print("🧪 CYCLE 4: Stage B - FP16 Arithmetic")
    print("="*80)
    
    B, H, S, D = 1, 8, 512, 64
    print(f"\nShape: B={B}, H={H}, S={S}, D={D}")
    
    torch.manual_seed(42)
    Q_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
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
    
    module = build_module()
    
    print(f"\n🔥 Warmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        out = module.forward(Q_fp8, K_fp8, V_fp8, Q_scale_tensor, K_scale_tensor, V_scale_tensor)
    torch.cuda.synchronize()
    print("  ✅ Done")
    
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
    
    # Correctness
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
    print(f"  Correct: {'✅' if correct else '❌'}")
    
    # Results
    print("\n" + "="*80)
    print("📊 RESULTS: Stage B")
    print("="*80)
    print(f"Latency:    {latency_us:.2f} μs")
    print(f"Correct:    {correct}")
    
    cycle2a_us = 1453.93
    print(f"\n📈 vs Cycle 2a ({cycle2a_us:.2f} μs):")
    speedup = cycle2a_us / latency_us
    print(f"  {speedup:.2f}× {'faster' if speedup > 1 else 'slower'}")
    
    if correct and speedup >= 2.0:
        print("\n✅ CYCLE 4 SUCCESS!")
        print("\n🎯 Next: Full WMMA matrix ops for additional speedup")
    elif correct and speedup >= 1.2:
        print("\n⚠️ MODEST IMPROVEMENT")
        print("\n→ Full WMMA needed for target performance")
    else:
        print("\n❌ NEEDS INVESTIGATION")

if __name__ == "__main__":
    main()

