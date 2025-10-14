#!/usr/bin/env python3
"""
Candidate Kernel Evaluation

Builds, runs, and evaluates candidate FA-S512 kernel configurations.
Returns latencies + metadata for hard gates and optimization loop.

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-13
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cudadent42.bench.fa_s512_tunable import FA_S512_Tunable
from cudadent42.bench.search_space import (
    calculate_smem_usage,
    estimate_occupancy,
    check_coalescing,
    check_bank_conflicts,
    hard_gates
)
from cudadent42.bench.common.env_lock import lock_environment
from cudadent42.bench.common.memory_tracker import MemoryTracker


def candidate_kernel(
    config: Dict[str, int],
    B: int = 32,
    H: int = 8,
    S: int = 512,
    D: int = 64,
    iterations: int = 40,
    warmup: int = 10,
    check_correctness: bool = True
) -> Dict:
    """
    Evaluate candidate kernel configuration
    
    Args:
        config: Dict with kernel tunables (BLOCK_M, BLOCK_N, etc.)
        B, H, S, D: Attention dimensions
        iterations: Number of timing iterations
        warmup: Number of warmup iterations
        check_correctness: Compare output against PyTorch SDPA
    
    Returns:
        Dict with:
            - latencies: List[float] (milliseconds)
            - median_ms: float
            - meta: Dict with validation metadata
                - config: input config
                - build_success: bool
                - run_success: bool
                - coalesced: bool
                - bank_conflicts: int
                - occupancy: float
                - smem_bytes: float
                - peak_mb: float
                - max_rel_err: float (if check_correctness)
                - gate_result: Optional[str] (rejection reason)
                - passes_gates: bool
    """
    # Lock environment
    lock_environment()
    
    # Verify TF32 disabled
    assert torch.backends.cuda.matmul.allow_tf32 == False, "TF32 not disabled!"
    
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create inputs
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    # Calculate expected values
    smem_bytes = calculate_smem_usage(config)
    occupancy = estimate_occupancy(config)
    coalesced = check_coalescing(config)
    bank_conflicts = check_bank_conflicts(config)
    
    # Initialize metadata
    meta = {
        'config': config,
        'build_success': False,
        'run_success': False,
        'coalesced': coalesced,
        'bank_conflicts': bank_conflicts,
        'occupancy': occupancy,
        'smem_bytes': smem_bytes,
        'peak_mb': 0.0,
        'max_rel_err': 0.0,
    }
    
    # Build and run kernel
    kernel = FA_S512_Tunable()
    
    with MemoryTracker() as mem_tracker:
        O_candidate, latencies, run_meta = kernel.run(
            config, Q, K, V, iterations=iterations, warmup=warmup
        )
    
    # Update metadata
    meta['build_success'] = run_meta.get('build_success', False)
    meta['run_success'] = run_meta.get('run_success', False)
    meta['error'] = run_meta.get('error', None)
    
    # Track memory
    mem_stats = mem_tracker.get_stats()
    meta['peak_mb'] = mem_stats.peak_mb
    
    # Check correctness if requested and kernel ran successfully
    if check_correctness and meta['run_success'] and O_candidate is not None:
        try:
            # Compute reference with PyTorch SDPA
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                O_ref = F.scaled_dot_product_attention(Q, K, V)
            
            # Compute max relative error
            abs_diff = torch.abs(O_candidate - O_ref)
            abs_ref = torch.abs(O_ref)
            rel_err = abs_diff / (abs_ref + 1e-8)
            max_rel_err = torch.max(rel_err).item()
            
            meta['max_rel_err'] = max_rel_err
        
        except Exception as e:
            meta['error'] = f"Correctness check failed: {e}"
            meta['max_rel_err'] = float('inf')
    
    # Apply hard gates
    gate_result = hard_gates(meta)
    meta['gate_result'] = gate_result
    meta['passes_gates'] = (gate_result is None)
    
    # Compute median latency
    median_ms = np.median(latencies) if latencies else float('inf')
    
    return {
        'latencies': latencies,
        'median_ms': median_ms,
        'meta': meta
    }


def test_candidate_kernel():
    """Test candidate kernel evaluation"""
    print("Testing Candidate Kernel Evaluation...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Test configuration
    config = {
        'BLOCK_M': 128,
        'BLOCK_N': 64,
        'BLOCK_K': 32,
        'NUM_WARPS': 4,
        'STAGES': 2,
        'UNROLL': 1,
        'CP_ASYNC': 1,
        'SWIZZLE': 1,
        'HALF2': 1,
    }
    
    print(f"\nTest config: {config}")
    print("\nRunning evaluation...")
    
    result = candidate_kernel(config, iterations=20, warmup=5)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Median latency: {result['median_ms']:.4f} ms")
    print(f"Build success:  {result['meta']['build_success']}")
    print(f"Run success:    {result['meta']['run_success']}")
    print(f"Coalesced:      {result['meta']['coalesced']}")
    print(f"Bank conflicts: {result['meta']['bank_conflicts']}")
    print(f"Occupancy:      {result['meta']['occupancy']:.2%}")
    print(f"SMEM usage:     {result['meta']['smem_bytes']/1024:.1f} KB")
    print(f"Peak memory:    {result['meta']['peak_mb']:.1f} MB")
    print(f"Max rel error:  {result['meta']['max_rel_err']:.2e}")
    print(f"Passes gates:   {result['meta']['passes_gates']}")
    
    if not result['meta']['passes_gates']:
        print(f"Rejection:      {result['meta']['gate_result']}")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    test_candidate_kernel()

