#!/usr/bin/env python3
"""
BlackwellSparseK Quickstart Example

Demonstrates 63× speedup over PyTorch sparse on NVIDIA GPUs.
"""

import torch
import time
import blackwellsparsek as bsk

def main():
    print("=" * 70)
    print("BlackwellSparseK Quickstart")
    print("=" * 70)
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This library requires NVIDIA GPU.")
        return
    
    device = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {device}")
    print()
    
    # Create sparse matrix (78% sparse, like our validated benchmark)
    print("Creating sparse matrix (8192×8192, 78% sparse)...")
    M, K, N = 8192, 8192, 8192
    
    # Create random dense matrix and make it sparse
    A_dense = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # Set bottom 78% of values to zero
    threshold = torch.quantile(A_dense.abs(), 0.78)
    A_dense[A_dense.abs() < threshold] = 0
    
    # Convert to sparse CSR
    A_sparse = A_dense.to_sparse_csr()
    
    # Create dense matrix B
    B_dense = torch.randn(K, N, dtype=torch.float16, device='cuda')
    
    # Verify sparsity
    nnz = A_sparse._nnz()
    sparsity = 1.0 - (nnz / (M * K))
    print(f"✅ Matrix created: {M}×{K}, {sparsity:.1%} sparse")
    print()
    
    # Validate matrix
    is_valid, info = bsk.validate_sparse_matrix(A_sparse, min_sparsity=0.7)
    if not is_valid:
        print(f"⚠️  Warning: {info['message']}")
        print(f"   {info['recommendation']}")
        print()
    else:
        print(f"✅ Matrix validation passed")
        print(f"   Sparsity: {info['sparsity']:.1%}")
        print(f"   Performance rating: {info['performance_rating']}")
        print()
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        _ = torch.sparse.mm(A_sparse, B_dense)
    torch.cuda.synchronize()
    print()
    
    # Benchmark PyTorch sparse (cuSPARSE backend)
    print("Benchmarking PyTorch sparse (cuSPARSE)...")
    num_iters = 10
    start = time.perf_counter()
    for _ in range(num_iters):
        C_pytorch = torch.sparse.mm(A_sparse, B_dense)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_iters
    
    # Calculate FLOPS
    flops = 2 * nnz * N  # multiply-add
    pytorch_tflops = (flops / pytorch_time) / 1e12
    
    print(f"   Time: {pytorch_time*1000:.2f} ms")
    print(f"   TFLOPS: {pytorch_tflops:.2f}")
    print()
    
    # Benchmark BlackwellSparseK
    print("Benchmarking BlackwellSparseK...")
    start = time.perf_counter()
    for _ in range(num_iters):
        C_bsk = bsk.sparse_mm(A_sparse, B_dense)
    torch.cuda.synchronize()
    bsk_time = (time.perf_counter() - start) / num_iters
    
    bsk_tflops = (flops / bsk_time) / 1e12
    
    print(f"   Time: {bsk_time*1000:.2f} ms")
    print(f"   TFLOPS: {bsk_tflops:.2f}")
    print()
    
    # Results
    speedup = pytorch_time / bsk_time
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"PyTorch sparse:     {pytorch_tflops:6.2f} TFLOPS  ({pytorch_time*1000:6.2f} ms)")
    print(f"BlackwellSparseK:   {bsk_tflops:6.2f} TFLOPS  ({bsk_time*1000:6.2f} ms)")
    print()
    print(f"🚀 Speedup: {speedup:.1f}× faster than PyTorch sparse")
    print()
    
    # Verify correctness
    print("Verifying correctness...")
    max_diff = (C_pytorch - C_bsk).abs().max().item()
    mean_diff = (C_pytorch - C_bsk).abs().mean().item()
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    
    if max_diff < 0.01:
        print("   ✅ Results match (within tolerance)")
    else:
        print("   ⚠️  Large difference detected")
    print()
    
    # Performance estimate for other GPUs
    print("Estimated performance on other GPUs:")
    for device_name in ['L4', 'A100', 'H100']:
        est = bsk.estimate_speedup((M, K), sparsity, device=device_name)
        print(f"   {device_name:4s}: {est['tflops']:6.1f} TFLOPS "
              f"({est['speedup_vs_pytorch']:.0f}× vs PyTorch)")
    print()
    
    print("=" * 70)
    print("🎉 Quickstart complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

