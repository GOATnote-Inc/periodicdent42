#!/usr/bin/env python3
"""
Benchmark FP8 SDPA kernel.

Usage:
  python bench/run_fp8_sdpa.py --shape S=512,D=64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
import time
import numpy as np

def quantize_to_fp8(tensor, per_channel=True):
    """
    Quantize FP16 tensor to FP8 E4M3 (simulated as uint8).
    
    Returns:
        fp8_data: uint8 tensor (FP8 as bytes)
        scale: per-tensor or per-channel scale
    """
    if per_channel:
        # Per-channel (per-head) quantization
        # tensor: [B, H, S, D]
        B, H, S, D = tensor.shape
        
        # Compute per-head max
        abs_max = tensor.abs().view(B, H, -1).max(dim=2, keepdim=True)[0]  # [B, H, 1]
        abs_max = abs_max.view(B, H, 1, 1)
        
        # FP8 E4M3 range: ~[-448, 448]
        fp8_max = 448.0
        scale = abs_max / fp8_max
        scale = scale.clamp(min=1e-12)  # Avoid division by zero
        
        # Quantize
        tensor_scaled = tensor / scale
        tensor_clipped = tensor_scaled.clamp(-fp8_max, fp8_max)
        
        # Convert to uint8 (simulated FP8)
        # In real FP8, would use proper conversion
        # For now, linearly map to [0, 255]
        tensor_uint8 = ((tensor_clipped + fp8_max) / (2 * fp8_max) * 255).round().to(torch.uint8)
        
        # Return per-head scale (shape: [H])
        scale_per_head = scale[0, :, 0, 0]  # [H]
        
        # Ensure scale is float32 and on CUDA
        scale_per_head = scale_per_head.to(dtype=torch.float32, device='cuda')
        
        return tensor_uint8, scale_per_head
    else:
        raise NotImplementedError("Per-tensor quantization not implemented")

def dequantize_from_fp8(fp8_data, scale):
    """Dequantize for verification."""
    fp8_max = 448.0
    tensor_float = fp8_data.float() / 255.0 * (2 * fp8_max) - fp8_max
    scale = scale.view(1, -1, 1, 1)  # [1, H, 1, 1]
    return (tensor_float * scale).to(torch.float16)

def benchmark_fp8_sdpa(B=1, H=8, S=512, D=64, warmup=10, iters=100):
    """Benchmark FP8 SDPA kernel."""
    
    print("=" * 80)
    print("BENCHMARK: FP8 SDPA (Cycle 1 Baseline)")
    print("=" * 80)
    print()
    print(f"Shape: B={B}, H={H}, S={S}, D={D}")
    print(f"Precision: FP8 E4M3 (inputs), FP32 (compute), FP16 (output)")
    print()
    
    # Generate test data (FP16)
    torch.manual_seed(42)
    q_fp16 = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k_fp16 = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v_fp16 = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Quantize to FP8
    print("Quantizing to FP8...")
    q_fp8, q_scale = quantize_to_fp8(q_fp16)
    k_fp8, k_scale = quantize_to_fp8(k_fp16)
    v_fp8, v_scale = quantize_to_fp8(v_fp16)
    print(f"  Q scale range: [{q_scale.min():.6f}, {q_scale.max():.6f}]")
    print(f"  K scale range: [{k_scale.min():.6f}, {k_scale.max():.6f}]")
    print(f"  V scale range: [{v_scale.min():.6f}, {v_scale.max():.6f}]")
    print()
    
    # Load module
    try:
        from build_fp8_sdpa import build_fp8_sdpa
        sdpa_fp8_baseline_v2 = build_fp8_sdpa()
    except Exception as e:
        print(f"❌ Module build/load failed: {e}")
        sys.exit(1)
    
    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        _ = sdpa_fp8_baseline_v2.forward(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)
    torch.cuda.synchronize()
    print("  ✅ Done")
    print()
    
    # Benchmark
    print(f"Benchmarking ({iters} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iters):
        out_fp16 = sdpa_fp8_baseline_v2.forward(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    latency_us = (elapsed / iters) * 1e6
    print(f"  ✅ Done")
    print()
    
    # Correctness check (vs FP16 SDPA)
    print("Checking correctness...")
    with torch.no_grad():
        # Dequantize for reference
        q_dequant = dequantize_from_fp8(q_fp8, q_scale)
        k_dequant = dequantize_from_fp8(k_fp8, k_scale)
        v_dequant = dequantize_from_fp8(v_fp8, v_scale)
        
        ref = torch.nn.functional.scaled_dot_product_attention(
            q_dequant, k_dequant, v_dequant,
            attn_mask=None, dropout_p=0.0, is_causal=False
        )
    
    max_diff = (out_fp16 - ref).abs().max().item()
    correct = max_diff <= 5e-3  # FP8 tolerance
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Correct: {'✅' if correct else '❌'} (threshold: 5e-3)")
    print()
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Latency:    {latency_us:.2f} μs")
    print(f"Correct:    {correct}")
    print(f"Max diff:   {max_diff:.6f}")
    print()
    
    # Compare vs champion
    champion_latency = 24.22
    print(f"Comparison vs xFormers Champion ({champion_latency:.2f} μs):")
    speedup = champion_latency / latency_us
    if speedup > 1.0:
        print(f"  ✅ {speedup:.2f}× faster!")
    else:
        print(f"  ⚠️ {1/speedup:.2f}× slower (expected for Cycle 1 baseline)")
    print()
    
    # Cycle 1 target
    print("Cycle 1 Target: 30-40 μs (baseline)")
    if 30 <= latency_us <= 40:
        print("  ✅ Within target range!")
    elif latency_us < 30:
        print(f"  ✅✅ Better than target! ({latency_us:.2f} < 30 μs)")
    else:
        print(f"  ⚠️ Slower than target ({latency_us:.2f} > 40 μs)")
    print()
    
    return {
        'latency_us': latency_us,
        'correct': correct,
        'max_diff': max_diff,
        'speedup_vs_champion': speedup,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='S=512,D=64')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    
    # Parse shape
    shape = {}
    for part in args.shape.split(','):
        k, v = part.split('=')
        shape[k] = int(v)
    
    S = shape.get('S', 512)
    D = shape.get('D', 64)
    
    results = benchmark_fp8_sdpa(
        B=1, H=8, S=S, D=D,
        warmup=args.warmup,
        iters=args.iters,
    )
    
    return 0 if results['correct'] else 1

if __name__ == '__main__':
    sys.exit(main())

