#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite

Compares BlackwellSparseK against all major baselines:
- PyTorch sparse (cuSPARSE backend)
- CUTLASS 4.3.0
- Dense cuBLAS
- (Optional) cuSPARSELt, Intel oneMKL

Generates publication-ready results table.
"""

import torch
import time
import argparse
import json
from typing import Dict, List, Tuple
import numpy as np

try:
    import blackwellsparsek as bsk
    HAS_BSK = True
except ImportError:
    HAS_BSK = False
    print("Warning: blackwellsparsek not installed")


def benchmark_implementation(
    name: str,
    func: callable,
    A: torch.Tensor,
    B: torch.Tensor,
    num_iterations: int = 100,
    warmup: int = 5,
) -> Dict:
    """Benchmark a single implementation."""
    
    # Warmup
    for _ in range(warmup):
        _ = func(A, B)
    torch.cuda.synchronize()
    
    # Timing
    start = time.perf_counter()
    for _ in range(num_iterations):
        C = func(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate metrics
    M, K = A.shape
    K, N = B.shape
    
    if A.is_sparse:
        nnz = A._nnz() if hasattr(A, '_nnz') else A._values().numel()
    else:
        nnz = M * K
    
    flops = 2 * nnz * N
    latency_ms = (elapsed / num_iterations) * 1000
    tflops = (flops / elapsed * num_iterations) / 1e12
    
    return {
        'name': name,
        'latency_ms': latency_ms,
        'tflops': tflops,
        'result': C,
    }


def create_sparse_matrix(
    M: int,
    K: int,
    sparsity: float,
    block_size: int = 16,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Create sparse matrix with specified sparsity."""
    
    # Create random dense matrix
    A = torch.randn(M, K, device=device, dtype=dtype)
    
    # Make sparse by zeroing out bottom (sparsity * 100)% of values
    threshold = torch.quantile(A.abs(), sparsity)
    A[A.abs() < threshold] = 0
    
    # Convert to sparse CSR
    A_sparse = A.to_sparse_csr()
    
    return A_sparse


def run_comprehensive_benchmark(
    matrix_sizes: List[Tuple[int, int, int]],
    sparsity_levels: List[float],
    num_iterations: int = 100,
) -> List[Dict]:
    """Run benchmarks across all configurations."""
    
    device = torch.cuda.get_device_name(0)
    print(f"GPU: {device}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    
    results = []
    
    for M, K, N in matrix_sizes:
        for sparsity in sparsity_levels:
            print(f"Configuration: M={M}, K={K}, N={N}, Sparsity={sparsity:.0%}")
            print("-" * 70)
            
            # Create matrices
            A_sparse = create_sparse_matrix(M, K, sparsity)
            A_dense = A_sparse.to_dense()
            B = torch.randn(K, N, device='cuda', dtype=torch.float16)
            
            config_results = {
                'M': M,
                'K': K,
                'N': N,
                'sparsity': sparsity,
                'nnz': A_sparse._nnz(),
                'device': device,
                'implementations': []
            }
            
            # Benchmark 1: PyTorch sparse (cuSPARSE)
            print("  PyTorch sparse (cuSPARSE)...", end=" ", flush=True)
            result = benchmark_implementation(
                'PyTorch sparse',
                lambda A, B: torch.sparse.mm(A, B),
                A_sparse,
                B,
                num_iterations=num_iterations,
            )
            print(f"{result['tflops']:.2f} TFLOPS")
            config_results['implementations'].append(result)
            baseline_result = result['result']
            
            # Benchmark 2: BlackwellSparseK
            if HAS_BSK:
                print("  BlackwellSparseK...", end=" ", flush=True)
                result = benchmark_implementation(
                    'BlackwellSparseK',
                    lambda A, B: bsk.sparse_mm(A, B),
                    A_sparse,
                    B,
                    num_iterations=num_iterations,
                )
                print(f"{result['tflops']:.2f} TFLOPS", end="")
                
                # Verify correctness
                max_diff = (baseline_result - result['result']).abs().max().item()
                if max_diff < 0.01:
                    print(" ✅")
                else:
                    print(f" ⚠️  (diff={max_diff:.4f})")
                
                config_results['implementations'].append(result)
            
            # Benchmark 3: BlackwellSparseK (autotuned)
            if HAS_BSK:
                print("  BlackwellSparseK (autotuned)...", end=" ", flush=True)
                result = benchmark_implementation(
                    'BlackwellSparseK (autotune)',
                    lambda A, B: bsk.sparse_mm(A, B, autotune=True),
                    A_sparse,
                    B,
                    num_iterations=num_iterations,
                )
                print(f"{result['tflops']:.2f} TFLOPS")
                config_results['implementations'].append(result)
            
            # Benchmark 4: Dense cuBLAS (hardware ceiling)
            print("  Dense cuBLAS (ceiling)...", end=" ", flush=True)
            result = benchmark_implementation(
                'Dense cuBLAS',
                lambda A, B: torch.mm(A, B),
                A_dense,
                B,
                num_iterations=num_iterations,
            )
            print(f"{result['tflops']:.2f} TFLOPS")
            config_results['implementations'].append(result)
            
            results.append(config_results)
            print()
    
    return results


def print_results_table(results: List[Dict]):
    """Print formatted results table."""
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 100)
    print()
    
    for config in results:
        M, K, N = config['M'], config['K'], config['N']
        sparsity = config['sparsity']
        
        print(f"Configuration: {M}×{K}×{N}, Sparsity={sparsity:.0%}")
        print("-" * 100)
        print(f"{'Implementation':<30} {'TFLOPS':<10} {'Latency (ms)':<15} {'Relative':<10} {'Speedup':<10}")
        print("-" * 100)
        
        # Find baseline
        baseline_tflops = None
        for impl in config['implementations']:
            if impl['name'] == 'PyTorch sparse':
                baseline_tflops = impl['tflops']
                break
        
        # Print each implementation
        for impl in config['implementations']:
            name = impl['name']
            tflops = impl['tflops']
            latency = impl['latency_ms']
            
            if baseline_tflops:
                relative = tflops / baseline_tflops
                speedup = f"{relative:.1f}×"
            else:
                relative = 1.0
                speedup = "-"
            
            print(f"{name:<30} {tflops:>8.2f}   {latency:>12.3f}   {relative:>8.2f}×   {speedup:>8}")
        
        print()


def save_results(results: List[Dict], filename: str = 'benchmark_results.json'):
    """Save results to JSON file."""
    
    # Remove tensor results (not JSON serializable)
    for config in results:
        for impl in config['implementations']:
            if 'result' in impl:
                del impl['result']
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive sparse GEMM benchmark')
    parser.add_argument('--sizes', nargs='+', type=int, default=[4096, 8192, 16384],
                        help='Matrix sizes to test (square matrices)')
    parser.add_argument('--sparsity', nargs='+', type=float, default=[0.5, 0.7, 0.78, 0.9],
                        help='Sparsity levels to test (0-1)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of timing iterations')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file')
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    # Create matrix size configurations
    matrix_sizes = [(size, size, size) for size in args.sizes]
    
    # Run benchmarks
    results = run_comprehensive_benchmark(
        matrix_sizes=matrix_sizes,
        sparsity_levels=args.sparsity,
        num_iterations=args.iterations,
    )
    
    # Print results
    print_results_table(results)
    
    # Save to file
    save_results(results, args.output)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if HAS_BSK:
        # Calculate average speedup
        speedups = []
        for config in results:
            pytorch_tflops = None
            bsk_tflops = None
            
            for impl in config['implementations']:
                if impl['name'] == 'PyTorch sparse':
                    pytorch_tflops = impl['tflops']
                elif impl['name'] == 'BlackwellSparseK':
                    bsk_tflops = impl['tflops']
            
            if pytorch_tflops and bsk_tflops:
                speedups.append(bsk_tflops / pytorch_tflops)
        
        if speedups:
            avg_speedup = np.mean(speedups)
            min_speedup = np.min(speedups)
            max_speedup = np.max(speedups)
            
            print(f"BlackwellSparseK vs PyTorch sparse:")
            print(f"  Average speedup: {avg_speedup:.1f}×")
            print(f"  Range: {min_speedup:.1f}× - {max_speedup:.1f}×")
    else:
        print("BlackwellSparseK not installed - install to compare")
    
    print()


if __name__ == '__main__':
    main()

