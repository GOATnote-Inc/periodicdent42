"""
Calculate L4 GPU Theoretical Limits for FlashAttention
======================================================

This script implements Step 1 of Optimization Through Inversion:
Calculate hardware theoretical limits before implementing.

Author: periodicdent42
Date: October 14, 2025
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class GPUSpecs:
    """L4 GPU Specifications (SM_89, Ada Lovelace)"""
    # Compute
    fp16_tflops: float = 242.0  # Tensor Core peak
    cuda_cores: int = 7680
    sm_count: int = 60
    warps_per_sm: int = 48
    max_threads_per_sm: int = 1536
    
    # Memory
    memory_bw_gbps: float = 300.0  # GB/s
    memory_bw: float = 300e9  # bytes/sec
    smem_per_sm: int = 49152  # 48 KB
    l2_cache: int = 4 * 1024 * 1024  # 4 MB
    registers_per_sm: int = 65536
    
    # Precision
    fp16_bytes: int = 2
    fp32_bytes: int = 4


@dataclass
class AttentionWorkload:
    """FlashAttention workload characteristics"""
    batch_size: int = 4
    num_heads: int = 8
    seq_len: int = 512
    head_dim: int = 64
    
    def flops(self) -> int:
        """Total FLOPs for attention computation"""
        B, H, S, D = self.batch_size, self.num_heads, self.seq_len, self.head_dim
        
        # Q @ K^T: (B, H, S, D) @ (B, H, D, S) = (B, H, S, S)
        flops_qk = B * H * S * S * D
        
        # Softmax: exp, max, sum, normalize (approximate as 5 ops per element)
        flops_softmax = B * H * S * S * 5
        
        # attention @ V: (B, H, S, S) @ (B, H, S, D) = (B, H, S, D)
        flops_ov = B * H * S * S * D
        
        return flops_qk + flops_softmax + flops_ov
    
    def bytes_naive(self, gpu: GPUSpecs) -> int:
        """Memory traffic for naive attention (materialize full matrix)"""
        B, H, S, D = self.batch_size, self.num_heads, self.seq_len, self.head_dim
        
        # Read Q, K, V
        bytes_read = B * H * S * D * gpu.fp16_bytes * 3
        
        # Write O
        bytes_write = B * H * S * D * gpu.fp16_bytes
        
        # Intermediate: attention scores (S, S) in FP32
        bytes_intermediate = B * H * S * S * gpu.fp32_bytes
        
        return bytes_read + bytes_write + bytes_intermediate
    
    def bytes_flashattention(self, gpu: GPUSpecs, tile_m: int, tile_n: int) -> int:
        """Memory traffic for FlashAttention with tiling"""
        B, H, S, D = self.batch_size, self.num_heads, self.seq_len, self.head_dim
        
        tiles_m = math.ceil(S / tile_m)
        tiles_n = math.ceil(S / tile_n)
        
        # Q: Loaded once per M-tile (outer loop)
        bytes_q = B * H * tiles_m * tile_m * D * gpu.fp16_bytes
        
        # K, V: Loaded tiles_M times (inner loop for each M-tile)
        bytes_k = B * H * tiles_m * tiles_n * tile_n * D * gpu.fp16_bytes
        bytes_v = B * H * tiles_m * tiles_n * tile_n * D * gpu.fp16_bytes
        
        # O: Written once
        bytes_o = B * H * S * D * gpu.fp16_bytes
        
        return bytes_q + bytes_k + bytes_v + bytes_o


def calculate_optimal_tile_size(gpu: GPUSpecs, workload: AttentionWorkload) -> Tuple[int, int]:
    """
    Calculate optimal tile sizes to maximize SMEM usage.
    
    SMEM layout:
    - Q_tile: TILE_M × D (FP16)
    - K_tile: TILE_N × D (FP16)
    - V_tile: TILE_N × D (FP16)
    - S_tile: TILE_M × TILE_N (FP32, attention scores)
    
    For double-buffering, need 2× the SMEM for Q/K/V tiles.
    """
    D = workload.head_dim
    available_smem = gpu.smem_per_sm
    fp16 = gpu.fp16_bytes
    fp32 = gpu.fp32_bytes
    
    # Reserve space for S_tile (not double-buffered)
    # Try different tile sizes and find largest that fits
    best_tile = (64, 64)
    best_utilization = 0.0
    
    for tile_size in [64, 80, 96, 112, 128]:
        tile_m = tile_n = tile_size
        
        # SMEM needed for double-buffering Q, K, V
        smem_q = 2 * tile_m * (D + 1) * fp16  # +1 for padding to avoid bank conflicts
        smem_k = 2 * tile_n * (D + 1) * fp16
        smem_v = 2 * tile_n * (D + 1) * fp16
        smem_s = tile_m * tile_n * fp32
        
        total_smem = smem_q + smem_k + smem_v + smem_s
        
        if total_smem <= available_smem:
            utilization = total_smem / available_smem
            if utilization > best_utilization:
                best_utilization = utilization
                best_tile = (tile_m, tile_n)
                
    return best_tile


def calculate_optimal_num_warps(tile_m: int, tensor_core_size: int = 16) -> int:
    """
    Calculate optimal number of warps based on tile size.
    
    Tensor Cores operate on 16×16 matrices, so NUM_WARPS should be chosen
    such that TILE_M / NUM_WARPS is a multiple of 16.
    """
    # Try warp counts that evenly divide tile_m
    for num_warps in [4, 6, 8, 12, 16]:
        rows_per_warp = tile_m / num_warps
        if rows_per_warp >= tensor_core_size and tile_m % num_warps == 0:
            return num_warps
    
    # Fallback
    return 4


def analyze_performance(gpu: GPUSpecs, workload: AttentionWorkload) -> Dict:
    """
    Complete performance analysis for FlashAttention on L4.
    
    Returns dict with:
    - Theoretical limits (compute-bound, memory-bound)
    - Optimal configuration (tile sizes, warp count)
    - Expected performance
    """
    # === Naive Attention (Baseline) ===
    flops_total = workload.flops()
    bytes_naive = workload.bytes_naive(gpu)
    
    compute_time_naive = flops_total / (gpu.fp16_tflops * 1e12)  # seconds
    memory_time_naive = bytes_naive / gpu.memory_bw  # seconds
    
    # === FlashAttention with Optimal Tiling ===
    tile_m, tile_n = calculate_optimal_tile_size(gpu, workload)
    bytes_fa = workload.bytes_flashattention(gpu, tile_m, tile_n)
    
    compute_time_fa = flops_total / (gpu.fp16_tflops * 1e12)
    memory_time_fa = bytes_fa / gpu.memory_bw
    
    # Arithmetic intensity
    arithmetic_intensity_naive = flops_total / bytes_naive
    arithmetic_intensity_fa = flops_total / bytes_fa
    
    # Bottleneck analysis
    bottleneck_naive = "compute" if compute_time_naive > memory_time_naive else "memory"
    bottleneck_fa = "compute" if compute_time_fa > memory_time_fa else "memory"
    
    # Optimal warp configuration
    num_warps = calculate_optimal_num_warps(tile_m)
    
    # Target performance (90% efficiency)
    peak_time_fa = max(compute_time_fa, memory_time_fa)
    target_time_fa = peak_time_fa / 0.90
    
    return {
        # Workload
        "flops_total": flops_total,
        "flops_gflops": flops_total / 1e9,
        
        # Naive attention
        "bytes_naive": bytes_naive,
        "bytes_naive_mb": bytes_naive / 1e6,
        "compute_time_naive_ms": compute_time_naive * 1000,
        "memory_time_naive_ms": memory_time_naive * 1000,
        "bottleneck_naive": bottleneck_naive,
        "arithmetic_intensity_naive": arithmetic_intensity_naive,
        
        # FlashAttention
        "tile_m": tile_m,
        "tile_n": tile_n,
        "num_warps": num_warps,
        "num_threads": num_warps * 32,
        "bytes_fa": bytes_fa,
        "bytes_fa_mb": bytes_fa / 1e6,
        "compute_time_fa_ms": compute_time_fa * 1000,
        "memory_time_fa_ms": memory_time_fa * 1000,
        "bottleneck_fa": bottleneck_fa,
        "arithmetic_intensity_fa": arithmetic_intensity_fa,
        
        # Speedup
        "memory_reduction": bytes_naive / bytes_fa,
        "speedup_theoretical": memory_time_naive / memory_time_fa,
        
        # Target performance
        "peak_time_fa_ms": peak_time_fa * 1000,
        "target_time_fa_ms": target_time_fa * 1000,
        "target_efficiency": 0.90,
        
        # GPU utilization targets
        "target_tc_utilization": 0.90,
        "target_bandwidth_utilization": 0.85,
    }


def print_analysis(analysis: Dict):
    """Pretty-print analysis results"""
    print("=" * 80)
    print("L4 THEORETICAL LIMITS ANALYSIS: FlashAttention S=512")
    print("=" * 80)
    print()
    
    print("📊 WORKLOAD")
    print(f"  Total FLOPs: {analysis['flops_gflops']:.2f} GFLOPS")
    print()
    
    print("🔴 NAIVE ATTENTION (Materialize full S×S matrix)")
    print(f"  Memory Traffic: {analysis['bytes_naive_mb']:.1f} MB")
    print(f"  Compute Time: {analysis['compute_time_naive_ms']:.3f} ms")
    print(f"  Memory Time: {analysis['memory_time_naive_ms']:.3f} ms")
    print(f"  Bottleneck: {analysis['bottleneck_naive'].upper()}")
    print(f"  Arithmetic Intensity: {analysis['arithmetic_intensity_naive']:.1f} FLOPS/byte")
    print()
    
    print("🟢 FLASHATTENTION (Optimal Tiling)")
    print(f"  Optimal Tile Size: {analysis['tile_m']}×{analysis['tile_n']}")
    print(f"  Optimal Warps: {analysis['num_warps']} ({analysis['num_threads']} threads)")
    print(f"  Memory Traffic: {analysis['bytes_fa_mb']:.1f} MB ({analysis['memory_reduction']:.1f}× reduction)")
    print(f"  Compute Time: {analysis['compute_time_fa_ms']:.3f} ms")
    print(f"  Memory Time: {analysis['memory_time_fa_ms']:.3f} ms")
    print(f"  Bottleneck: {analysis['bottleneck_fa'].upper()}")
    print(f"  Arithmetic Intensity: {analysis['arithmetic_intensity_fa']:.1f} FLOPS/byte")
    print()
    
    print("🎯 THEORETICAL PEAK PERFORMANCE")
    print(f"  Peak Time: {analysis['peak_time_fa_ms']:.3f} ms")
    print(f"  Target Time (90% eff): {analysis['target_time_fa_ms']:.3f} ms")
    print(f"  Target TC Utilization: {analysis['target_tc_utilization']*100:.0f}%")
    print(f"  Target Bandwidth: {analysis['target_bandwidth_utilization']*100:.0f}%")
    print()
    
    print("📈 COMPARISON TO BASELINES")
    print(f"  Theoretical Speedup vs Naive: {analysis['speedup_theoretical']:.1f}×")
    print(f"  PyTorch SDPA (measured): 0.163 ms")
    print(f"  Theoretical Optimal: {analysis['target_time_fa_ms']:.3f} ms")
    print(f"  Potential Speedup vs PyTorch: {0.163 / analysis['target_time_fa_ms']:.1f}×")
    print()
    
    print("✅ OPTIMAL CONFIGURATION FOR IMPLEMENTATION")
    print(f"  #define TILE_M {analysis['tile_m']}")
    print(f"  #define TILE_N {analysis['tile_n']}")
    print(f"  #define NUM_WARPS {analysis['num_warps']}")
    print(f"  #define NUM_THREADS {analysis['num_threads']}")
    print(f"  #define HEAD_DIM 64")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    # L4 GPU specifications
    gpu = GPUSpecs()
    
    # FlashAttention workload (B=4, H=8, S=512, D=64)
    workload = AttentionWorkload()
    
    # Calculate theoretical limits
    analysis = analyze_performance(gpu, workload)
    
    # Print results
    print_analysis(analysis)
    
    # Save to file for reference
    import json
    with open("l4_theoretical_limits.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("💾 Saved detailed analysis to l4_theoretical_limits.json")

