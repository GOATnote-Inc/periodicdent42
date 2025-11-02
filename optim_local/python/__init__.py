"""
Sparse BSR Auto-Tuning for PyTorch

Usage:
    import sparse_autotune
    
    # Auto-tuned sparse matmul
    output = sparse_autotune.matmul(A_bsr, B_dense)
    
    # Benchmark variants
    results = sparse_autotune.benchmark(A_bsr, B_dense)
"""

import torch
import sparse_autotune_cpp
from typing import Tuple, Dict
import os

class BSRMatrix:
    """Block Sparse Row matrix wrapper"""
    def __init__(self, row_ptr: torch.Tensor, col_indices: torch.Tensor, 
                 values: torch.Tensor, shape: Tuple[int, int], block_size: int):
        self.row_ptr = row_ptr
        self.col_indices = col_indices
        self.values = values
        self.shape = shape
        self.block_size = block_size
        
    @property
    def M(self):
        return self.shape[0]
    
    @property
    def K(self):
        return self.shape[1]
    
    @property
    def nnzb(self):
        return self.col_indices.size(0)

def matmul(A: BSRMatrix, B: torch.Tensor, variant: str = 'auto') -> torch.Tensor:
    """
    Sparse matrix multiplication: C = A @ B
    
    Args:
        A: BSR sparse matrix
        B: Dense matrix [K, N]
        variant: 'auto' (auto-tuned), 'custom', or 'cusparse'
    
    Returns:
        Dense output matrix [M, N]
    """
    if variant == 'auto':
        # Check cache
        config_key = f"{A.M}_{B.size(1)}_{A.K}_bs{A.block_size}"
        cache_file = f"/tmp/sparse_cache_{config_key}.txt"
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                variant = f.read().strip()
        else:
            # Run auto-tuning
            results = benchmark(A, B)
            variant = min(results.items(), key=lambda x: x[1])[0]
            
            # Cache result
            with open(cache_file, 'w') as f:
                f.write(variant)
    
    if variant == 'custom':
        return sparse_autotune_cpp.sparse_matmul_auto(
            A.row_ptr, A.col_indices, A.values, B, A.M, A.K, A.block_size
        )
    elif variant == 'cusparse':
        return sparse_autotune_cpp.sparse_matmul_cusparse(
            A.row_ptr, A.col_indices, A.values, B, A.M, A.K, A.block_size
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

def benchmark(A: BSRMatrix, B: torch.Tensor, num_runs: int = 20) -> Dict[str, float]:
    """
    Benchmark all sparse matmul variants
    
    Args:
        A: BSR sparse matrix
        B: Dense matrix [K, N]
        num_runs: Number of timing runs
    
    Returns:
        Dictionary mapping variant name to time in ms
    """
    results = {}
    
    for variant in ['custom', 'cusparse']:
        ms = sparse_autotune_cpp.benchmark_variant(
            variant, A.row_ptr, A.col_indices, A.values, B,
            A.M, A.K, A.block_size, num_runs
        )
        results[variant] = ms
        
        # Calculate TFLOPS
        flops = 2.0 * A.nnzb * A.block_size * A.block_size * B.size(1) * A.block_size
        tflops = (flops / (ms / 1000.0)) / 1e12
        print(f"  {variant:20s}: {ms:6.3f} ms → {tflops:6.1f} TFLOPS")
    
    return results

def create_random_bsr(M: int, K: int, block_size: int, sparsity: float = 0.875,
                      device='cuda') -> BSRMatrix:
    """
    Create a random BSR matrix for testing
    
    Args:
        M, K: Matrix dimensions
        block_size: Block size
        sparsity: Fraction of blocks that are zero
        device: 'cuda' or 'cpu'
    
    Returns:
        Random BSR matrix
    """
    import random
    random.seed(42)
    
    M_blocks = (M + block_size - 1) // block_size
    K_blocks = (K + block_size - 1) // block_size
    
    row_ptr_list = [0]
    col_indices_list = []
    values_list = []
    
    nnzb = 0
    for i in range(M_blocks):
        for j in range(K_blocks):
            if random.random() > sparsity:  # Non-zero block
                col_indices_list.append(j)
                block = torch.randn(block_size, block_size)
                values_list.append(block)
                nnzb += 1
        row_ptr_list.append(nnzb)
    
    row_ptr = torch.tensor(row_ptr_list, dtype=torch.int32, device=device)
    col_indices = torch.tensor(col_indices_list, dtype=torch.int32, device=device)
    values = torch.stack(values_list) if values_list else torch.empty(0, block_size, block_size)
    values = values.to(device)
    
    return BSRMatrix(row_ptr, col_indices, values, (M, K), block_size)
