#!/usr/bin/env python3
"""
Cycle 5: Full WMMA (Tensor Core) implementation
Target: 200-400 μs (expert estimate with Tensor Cores)
"""

import torch
import torch.utils.cpp_extension
import os
import sys
import argparse

def build_module():
    """Build the CUDA extension."""
    print("\n" + "="*80)
    print("🚀 BUILDING FP8 SDPA WITH WMMA (TENSOR CORES)")
    print("="*80)
    print("\nStage C: Full WMMA")
    print("  ✅ WMMA for Q@K^T (16×16×16 tiles)")
    print("  ✅ FP16 Tensor Core operations")
    print("  ✅ Proper matrix layouts (row/col-major)")
    print("\nCompiling sdpa_fp8_wmma.cu...")
    
    source_dir = os.path.join(os.path.dirname(__file__), "..", "cudadent42", "bench", "kernels")
    
    module = torch.utils.cpp_extension.load(
        name="sdpa_fp8_wmma",
        sources=[
            os.path.join(source_dir, "sdpa_fp8_wmma.cu"),
            os.path.join(source_dir, "sdpa_fp8_wmma_bindings.cpp"),
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
    print("="*80 + "\n")
    
    return module

def quantize_fp16_to_sim_fp8(x_fp16, per_tensor=True):
    """Quantize FP16 tensor to simulated FP8 (uint8)."""
    x_float = x_fp16.float()
    
    if per_tensor:
        abs_max = x_float.abs().max()
    else:
        # Per-head quantization
        abs_max = x_float.abs().amax(dim=(-2, -1), keepdim=True)
    
    fp8_max = 448.0
    scale = abs_max / fp8_max
    scale = torch.clamp(scale, min=1e-10)
    
    x_scaled = x_float / scale
    x_clipped = torch.clamp(x_scaled, -fp8_max, fp8_max)
    x_normalized = (x_clipped + fp8_max) / (2 * fp8_max)
    x_uint8 = (x_normalized * 255).round().clamp(0, 255).to(torch.uint8)
    
    return x_uint8, scale.squeeze()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    args = parser.parse_args()
    
    print("="*80)
    print("🧪 CYCLE 5: Full WMMA (Tensor Cores)")
    print("="*80)
    
    # Test shape
    B, H, S, D = 1, 8, 512, 64
    print(f"\nShape: B={B}, H={H}, S={S}, D={D}\n")
    
    # Build module
    module = build_module()
    
    print("🔥 Warmup ({} iterations)...".format(args.warmup))
    
    # Create test inputs
    torch.manual_seed(42)
    Q_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    K_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    V_fp16 = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda') * 0.1
    
    # Quantize (per-head)
    Q_fp8, q_scale = quantize_fp16_to_sim_fp8(Q_fp16, per_tensor=False)
    K_fp8, k_scale = quantize_fp16_to_sim_fp8(K_fp16, per_tensor=False)
    V_fp8, v_scale = quantize_fp16_to_sim_fp8(V_fp16, per_tensor=False)
    
    # Ensure scales are [H] shape and float32
    q_scale = q_scale.view(H).to(dtype=torch.float32, device='cuda')
    k_scale = k_scale.view(H).to(dtype=torch.float32, device='cuda')
    v_scale = v_scale.view(H).to(dtype=torch.float32, device='cuda')
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    # Warmup
    for _ in range(args.warmup):
        out = module.forward(Q_fp8, K_fp8, V_fp8, q_scale, k_scale, v_scale, softmax_scale)
        torch.cuda.synchronize()
    
    print("  ✅ Done\n")
    
    # Benchmark
    print(f"⏱️  Benchmarking ({args.iters} iterations)...")
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(args.iters):
        out = module.forward(Q_fp8, K_fp8, V_fp8, q_scale, k_scale, v_scale, softmax_scale)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latency_us = (elapsed_ms / args.iters) * 1000
    
    print("  ✅ Done\n")
    
    # Correctness check
    print("🔍 Checking correctness...")
    with torch.no_grad():
        # Dequantize using same formula as CUDA kernel
        fp8_max = 448.0
        q_dequant = ((Q_fp8.float() / 255.0) * (2 * fp8_max) - fp8_max) * q_scale.unsqueeze(-1).unsqueeze(-1)
        k_dequant = ((K_fp8.float() / 255.0) * (2 * fp8_max) - fp8_max) * k_scale.unsqueeze(-1).unsqueeze(-1)
        v_dequant = ((V_fp8.float() / 255.0) * (2 * fp8_max) - fp8_max) * v_scale.unsqueeze(-1).unsqueeze(-1)
        
        q_dequant = q_dequant.to(torch.float16)
        k_dequant = k_dequant.to(torch.float16)
        v_dequant = v_dequant.to(torch.float16)
        
        ref = torch.nn.functional.scaled_dot_product_attention(
            q_dequant, k_dequant, v_dequant,
            attn_mask=None, dropout_p=0.0, is_causal=False
        )
    
    out_fp16 = out.to(torch.float16)
    max_diff = (out_fp16 - ref).abs().max().item()
    correct = max_diff <= 5e-3  # FP8 tolerance
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Correct: {'✅' if correct else '❌'}\n")
    
    # Results
    print("="*80)
    print("📊 RESULTS: Full WMMA")
    print("="*80)
    print(f"Latency:    {latency_us:.2f} μs")
    print(f"Correct:    {correct}")
    
    # Compare to Stage B
    stage_b_latency = 1381.99
    speedup = stage_b_latency / latency_us
    print(f"\n📈 vs Stage B ({stage_b_latency:.2f} μs):")
    print(f"  {speedup:.2f}× {'faster' if speedup > 1 else 'slower'}")
    
    if not correct:
        print("\n❌ CORRECTNESS FAILURE - needs debugging")
    elif latency_us < 500:
        print("\n✅ EXCELLENT - Tensor Cores working!")
    elif latency_us < 1000:
        print("\n⚠️  GOOD - Some speedup, but not full TC benefit")
    else:
        print("\n❌ REGRESSION - WMMA not helping")
    
    print()

if __name__ == "__main__":
    main()

