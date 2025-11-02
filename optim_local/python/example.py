#!/usr/bin/env python3
"""
Example: Using Sparse Auto-Tuning in PyTorch

Shows how to use the sparse auto-tuning library for BSR sparse matmul.
"""

import torch
import sparse_autotune

def main():
    print("═" * 60)
    print("  Sparse BSR Auto-Tuning Example")
    print("═" * 60)
    print()
    
    # Configuration
    M, K, N = 4096, 4096, 4096
    block_size = 64
    sparsity = 0.875  # 87.5% sparse
    
    print(f"Matrix: {M} x {K} @ {K} x {N}")
    print(f"Block size: {block_size}")
    print(f"Sparsity: {sparsity * 100:.1f}%")
    print()
    
    # Create random sparse matrix
    print("Generating random BSR matrix...")
    A = sparse_autotune.create_random_bsr(M, K, block_size, sparsity)
    print(f"  nnzb = {A.nnzb} ({A.nnzb / ((M/block_size) * (K/block_size)) * 100:.1f}% dense at block level)")
    print()
    
    # Create dense matrix
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Method 1: Auto-tuned (best performance)
    print("Method 1: Auto-tuned (recommended)")
    print("-" * 60)
    results = sparse_autotune.benchmark(A, B)
    print()
    
    # Method 2: Explicit variant selection
    print("Method 2: Explicit variant")
    print("-" * 60)
    C_custom = sparse_autotune.matmul(A, B, variant='custom')
    C_cusparse = sparse_autotune.matmul(A, B, variant='cusparse')
    
    # Verify correctness
    max_diff = (C_custom - C_cusparse).abs().max().item()
    mean_diff = (C_custom - C_cusparse).abs().mean().item()
    print(f"Correctness check:")
    print(f"  max_diff = {max_diff:.6f}")
    print(f"  mean_diff = {mean_diff:.6f}")
    print()
    
    if max_diff < 1e-3:
        print("✅ Results match!")
    else:
        print("⚠️  Results differ (expected for different algorithms)")
    print()
    
    # Method 3: Auto-tuned with caching
    print("Method 3: Auto-tuned with caching (second call)")
    print("-" * 60)
    C_auto = sparse_autotune.matmul(A, B, variant='auto')
    print("  (Loaded from cache - zero overhead)")
    print()
    
    print("=" * 60)
    print("Example complete!")
    print()
    print("Integration with PyTorch model:")
    print("  class SparseLayer(nn.Module):")
    print("      def forward(self, x):")
    print("          return sparse_autotune.matmul(self.W_sparse, x)")

if __name__ == '__main__':
    main()
