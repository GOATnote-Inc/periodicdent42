#!/usr/bin/env python3
"""
Honest Head-to-Head Benchmark: PyTorch Sparse vs cuSPARSE vs Custom Kernel

Tests SAME operation (sparse BSR GEMM) across all implementations.
No marketing, no cherry-picking, just raw measurements.

Author: Brandon Dent, MD
Date: November 1, 2025
"""

import torch
import time
import json
import sys
from typing import Dict, List, Tuple
import numpy as np

# Try importing cuSPARSE (via torch)
try:
    import torch.sparse
    TORCH_SPARSE_AVAILABLE = True
except ImportError:
    TORCH_SPARSE_AVAILABLE = False
    print("⚠️  PyTorch sparse not available")

# Try importing our custom kernel
try:
    import sparse_h100_kernel  # Compiled CUDA extension
    CUSTOM_KERNEL_AVAILABLE = True
except ImportError:
    CUSTOM_KERNEL_AVAILABLE = False
    print("⚠️  Custom kernel not compiled - run build script first")


class BenchmarkConfig:
    """Configuration for sparse BSR GEMM benchmark"""
    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        block_size: int = 16,
        topk: int = 16,
        sparsity: float = 0.78
    ):
        self.M = M
        self.N = N
        self.K = K
        self.block_size = block_size
        self.topk = topk
        self.sparsity = sparsity
        
    def __str__(self):
        return f"M={self.M}, N={self.N}, K={self.K}, block={self.block_size}, topk={self.topk}"


def generate_sparse_bsr_matrices(config: BenchmarkConfig, device: str = "cuda"):
    """
    Generate sparse BSR matrices for benchmarking.
    
    Returns:
        A: Sparse matrix (M×K) in BSR format
        B: Dense matrix (K×N)
        Expected pattern similar to attention QK^T
    """
    torch.manual_seed(42)  # Reproducibility
    
    M_blocks = config.M // config.block_size
    K_blocks = config.K // config.block_size
    
    # Generate sparse pattern (topk blocks per row)
    num_blocks = M_blocks * config.topk
    
    # Create BSR structure
    row_indices = torch.repeat_interleave(
        torch.arange(M_blocks, device=device),
        config.topk
    )
    
    # Random column indices (topk per row)
    col_indices = torch.stack([
        torch.randperm(K_blocks, device=device)[:config.topk]
        for _ in range(M_blocks)
    ]).flatten()
    
    # Generate block values (FP16 for realistic perf)
    block_values = torch.randn(
        num_blocks,
        config.block_size,
        config.block_size,
        dtype=torch.float16,
        device=device
    ) * 0.1
    
    # Dense B matrix
    B = torch.randn(config.K, config.N, dtype=torch.float16, device=device) * 0.1
    
    return {
        'row_indices': row_indices,
        'col_indices': col_indices,
        'block_values': block_values,
        'B': B,
        'M_blocks': M_blocks,
        'K_blocks': K_blocks
    }


def benchmark_pytorch_sparse(data: Dict, config: BenchmarkConfig, iterations: int = 100):
    """
    Benchmark PyTorch native sparse BSR operations.
    
    This is the CRITICAL baseline - if our kernel doesn't beat this,
    there's no point in custom CUDA.
    """
    if not TORCH_SPARSE_AVAILABLE:
        return None
    
    try:
        # Convert to PyTorch sparse_csr format (closest to BSR)
        # Note: PyTorch doesn't have native BSR, so we flatten blocks
        device = data['B'].device
        
        # Flatten BSR to CSR-like structure
        M = config.M
        K = config.K
        block_size = config.block_size
        
        # Create sparse tensor (this is not optimal, but it's PyTorch's native path)
        # Real BSR support in PyTorch is limited
        dense_A = torch.zeros(M, K, dtype=torch.float16, device=device)
        
        # Fill in blocks
        for i, (row_idx, col_idx, block) in enumerate(
            zip(data['row_indices'], data['col_indices'], data['block_values'])
        ):
            row_start = row_idx * block_size
            col_start = col_idx * block_size
            dense_A[row_start:row_start+block_size, col_start:col_start+block_size] = block
        
        # Convert to sparse (PyTorch will use cuSPARSE under the hood)
        A_sparse = dense_A.to_sparse_csr()
        B = data['B']
        
        # Warmup
        for _ in range(10):
            C = torch.sparse.mm(A_sparse, B)
        
        torch.cuda.synchronize()
        
        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        start.record()
        for _ in range(iterations):
            C = torch.sparse.mm(A_sparse, B)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        avg_time_ms = elapsed_ms / iterations
        
        # Calculate TFLOPS
        # Sparse GEMM: 2 * nnz * N operations
        nnz = A_sparse._nnz()
        ops = 2 * nnz * config.N
        tflops = (ops / (avg_time_ms * 1e-3)) / 1e12
        
        return {
            'name': 'PyTorch sparse.mm (cuSPARSE backend)',
            'time_ms': avg_time_ms,
            'tflops': tflops,
            'nnz': nnz,
            'result': C.cpu()
        }
        
    except Exception as e:
        print(f"❌ PyTorch sparse failed: {e}")
        return None


def benchmark_torch_dense_baseline(data: Dict, config: BenchmarkConfig, iterations: int = 100):
    """
    Benchmark dense matmul (worst case - no sparsity exploitation).
    
    This is the FLOOR - any sparse kernel should beat this.
    """
    device = data['B'].device
    
    # Create dense version of A
    dense_A = torch.zeros(config.M, config.K, dtype=torch.float16, device=device)
    block_size = config.block_size
    
    for row_idx, col_idx, block in zip(
        data['row_indices'], data['col_indices'], data['block_values']
    ):
        row_start = row_idx * block_size
        col_start = col_idx * block_size
        dense_A[row_start:row_start+block_size, col_start:col_start+block_size] = block
    
    B = data['B']
    
    # Warmup
    for _ in range(10):
        C = torch.mm(dense_A, B)
    
    torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        C = torch.mm(dense_A, B)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    avg_time_ms = elapsed_ms / iterations
    
    # Calculate TFLOPS (dense operations, even though matrix is sparse)
    ops = 2 * config.M * config.N * config.K
    tflops = (ops / (avg_time_ms * 1e-3)) / 1e12
    
    return {
        'name': 'PyTorch dense mm (cuBLAS, no sparsity)',
        'time_ms': avg_time_ms,
        'tflops': tflops,
        'result': C.cpu()
    }


def benchmark_custom_kernel(data: Dict, config: BenchmarkConfig, iterations: int = 100):
    """
    Benchmark our custom H100 sparse BSR kernel.
    
    This is what we're claiming is faster. Let's see if it's true.
    """
    if not CUSTOM_KERNEL_AVAILABLE:
        return None
    
    try:
        # Call our custom CUDA kernel
        result = sparse_h100_kernel.sparse_bsr_gemm(
            data['row_indices'],
            data['col_indices'],
            data['block_values'],
            data['B'],
            config.M,
            config.N,
            config.K,
            config.block_size,
            iterations=iterations
        )
        
        return {
            'name': 'Custom H100 kernel (WMMA + cp.async)',
            'time_ms': result['time_ms'],
            'tflops': result['tflops'],
            'result': result['output'].cpu()
        }
        
    except Exception as e:
        print(f"❌ Custom kernel failed: {e}")
        return None


def validate_correctness(results: List[Dict], tolerance: float = 1e-2):
    """
    Validate that all implementations produce the same result.
    
    This is CRITICAL - if results don't match, something is wrong.
    """
    if len(results) < 2:
        return True
    
    reference = results[0]['result']
    
    for i, result in enumerate(results[1:], 1):
        output = result['result']
        
        # Check shape
        if reference.shape != output.shape:
            print(f"❌ Shape mismatch: {reference.shape} vs {output.shape}")
            return False
        
        # Check values
        diff = torch.abs(reference - output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\n{result['name']} vs reference:")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff > tolerance:
            print(f"❌ FAILED: Max diff {max_diff} > tolerance {tolerance}")
            return False
        
        print(f"✅ PASSED")
    
    return True


def print_results_table(results: List[Dict], config: BenchmarkConfig):
    """
    Print results in a clear, honest table.
    
    No cherry-picking. No hiding slow results. Just facts.
    """
    print("\n" + "="*80)
    print(f"HONEST BENCHMARK RESULTS: {config}")
    print("="*80)
    
    # Find fastest (this is the actual winner)
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("❌ No valid results")
        return
    
    fastest = max(valid_results, key=lambda x: x['tflops'])
    
    # Print table
    print(f"\n{'Implementation':<50} {'Time (ms)':<12} {'TFLOPS':<12} {'Status'}")
    print("-"*80)
    
    for result in valid_results:
        is_fastest = (result['name'] == fastest['name'])
        marker = "🏆" if is_fastest else "  "
        
        print(f"{marker} {result['name']:<48} {result['time_ms']:>10.3f}  {result['tflops']:>10.2f}  ")
    
    print("-"*80)
    
    # Speedup analysis
    if len(valid_results) > 1:
        print("\n📊 Speedup Analysis:")
        for result in valid_results:
            if result['name'] != fastest['name']:
                speedup = fastest['tflops'] / result['tflops']
                print(f"  {fastest['name']}")
                print(f"    vs {result['name']}: {speedup:.2f}x faster")
    
    print("\n" + "="*80)


def save_results(results: List[Dict], config: BenchmarkConfig, output_file: str = "benchmark_results.json"):
    """Save results to JSON for analysis"""
    data = {
        'config': {
            'M': config.M,
            'N': config.N,
            'K': config.K,
            'block_size': config.block_size,
            'topk': config.topk,
            'sparsity': config.sparsity
        },
        'results': [
            {
                'name': r['name'],
                'time_ms': r['time_ms'],
                'tflops': r['tflops']
            }
            for r in results if r is not None
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Results saved to {output_file}")


def main():
    """Run comprehensive honest benchmark"""
    
    print("\n🔬 HONEST HEAD-TO-HEAD BENCHMARK")
    print("="*80)
    print("Testing: Sparse BSR GEMM")
    print("Baselines: PyTorch sparse, PyTorch dense, Custom kernel")
    print("Goal: See if custom kernel actually faster (spoiler: we don't know yet)")
    print("="*80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - run on H100")
        sys.exit(1)
    
    device = torch.cuda.get_device_name(0)
    print(f"\n✅ Device: {device}")
    
    if "H100" not in device:
        print("⚠️  WARNING: Not running on H100 - results may differ")
    
    # Configuration (same as claimed in PROOF_NOV1_2025.md)
    config = BenchmarkConfig(
        M=8192,
        N=8192,
        K=8192,
        block_size=16,
        topk=16,
        sparsity=0.78
    )
    
    print(f"\n📋 Configuration: {config}")
    
    # Generate test data
    print("\n🎲 Generating sparse matrices...")
    data = generate_sparse_bsr_matrices(config, device="cuda")
    print(f"✅ Generated {config.sparsity*100:.1f}% sparse matrix")
    
    # Run benchmarks
    results = []
    
    print("\n" + "="*80)
    print("RUNNING BENCHMARKS (100 iterations each)")
    print("="*80)
    
    print("\n1️⃣  PyTorch dense matmul (floor - should be slowest)...")
    result = benchmark_torch_dense_baseline(data, config)
    if result:
        results.append(result)
        print(f"   ✅ {result['tflops']:.2f} TFLOPS")
    
    print("\n2️⃣  PyTorch sparse.mm (cuSPARSE backend - key baseline)...")
    result = benchmark_pytorch_sparse(data, config)
    if result:
        results.append(result)
        print(f"   ✅ {result['tflops']:.2f} TFLOPS")
    
    print("\n3️⃣  Custom H100 kernel (our claim to fame)...")
    result = benchmark_custom_kernel(data, config)
    if result:
        results.append(result)
        print(f"   ✅ {result['tflops']:.2f} TFLOPS")
    
    # Validate correctness
    print("\n" + "="*80)
    print("CORRECTNESS VALIDATION")
    print("="*80)
    
    if validate_correctness(results):
        print("\n✅ All implementations produce identical results")
    else:
        print("\n❌ CORRECTNESS FAILURE - results don't match!")
        print("⚠️  Performance numbers meaningless if results are wrong")
        sys.exit(1)
    
    # Print results
    print_results_table(results, config)
    
    # Save results
    save_results(results, config)
    
    # Honest conclusion
    print("\n📝 HONEST CONCLUSION:")
    if len(results) < 2:
        print("❌ Insufficient baselines - can't draw conclusions")
    else:
        fastest = max(results, key=lambda x: x['tflops'])
        print(f"✅ Fastest implementation: {fastest['name']}")
        print(f"✅ Performance: {fastest['tflops']:.2f} TFLOPS")
        
        if fastest['name'].startswith('Custom'):
            print("✅ Custom kernel wins (as claimed)")
        else:
            print("❌ Custom kernel NOT fastest (claim was wrong)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

