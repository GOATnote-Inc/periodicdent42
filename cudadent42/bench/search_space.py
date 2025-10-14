"""
Search Space Definition for FA-S512 Kernel Tuning

Defines the hyperparameter search space and validation gates
from the CUDA optimization doctrine.

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-13
"""

from typing import Dict, List, Optional, Any

# Search space for kernel tunables
SEARCH_SPACE = {
    'BLOCK_M': [64, 128, 256],
    'BLOCK_N': [64, 128, 256],
    'BLOCK_K': [32, 64],
    'NUM_WARPS': [4, 8],
    'STAGES': [2, 3, 4],
    'UNROLL': [1, 2, 4],
    'CP_ASYNC': [0, 1],
    'SWIZZLE': [0, 1],
    'HALF2': [0, 1],
}

# Total configurations: 3×3×2×2×3×3×2×2×2 = 3,888


def calculate_smem_usage(config: Dict[str, int]) -> float:
    """
    Estimate shared memory usage in bytes
    
    Args:
        config: Kernel configuration
    
    Returns:
        Estimated SMEM usage in bytes
    """
    BLOCK_M = config['BLOCK_M']
    BLOCK_N = config['BLOCK_N']
    D = 64  # Fixed head dimension
    STAGES = config['STAGES']
    SWIZZLE = config['SWIZZLE']
    
    # Q_smem: [STAGES][BLOCK_M][D + SMEM_PAD]
    # K_smem: [STAGES][BLOCK_N][D + SMEM_PAD]
    # V_smem: [STAGES][BLOCK_N][D + SMEM_PAD]
    # S_smem: [BLOCK_M][BLOCK_N]
    
    pad = 1 if SWIZZLE else 0
    bytes_per_half = 2
    bytes_per_float = 4
    
    q_smem = STAGES * BLOCK_M * (D + pad) * bytes_per_half
    k_smem = STAGES * BLOCK_N * (D + pad) * bytes_per_half
    v_smem = STAGES * BLOCK_N * (D + pad) * bytes_per_half
    s_smem = BLOCK_M * BLOCK_N * bytes_per_float
    
    total = q_smem + k_smem + v_smem + s_smem
    return total


def estimate_occupancy(config: Dict[str, int]) -> float:
    """
    Estimate theoretical occupancy
    
    Args:
        config: Kernel configuration
    
    Returns:
        Estimated occupancy (0.0 to 1.0)
    """
    NUM_WARPS = config['NUM_WARPS']
    threads_per_block = NUM_WARPS * 32
    smem_bytes = calculate_smem_usage(config)
    
    # L4 limits (SM_89)
    max_threads_per_sm = 1536
    max_blocks_per_sm = 16
    max_smem_per_sm = 49152  # 48 KB
    
    # Calculate limits
    blocks_by_threads = max_threads_per_sm // threads_per_block
    blocks_by_smem = max_smem_per_sm // smem_bytes if smem_bytes > 0 else max_blocks_per_sm
    blocks_by_hardware = max_blocks_per_sm
    
    # Actual blocks per SM
    blocks_per_sm = min(blocks_by_threads, blocks_by_smem, blocks_by_hardware)
    
    # Occupancy = (active warps) / (max warps per SM)
    max_warps_per_sm = max_threads_per_sm // 32
    active_warps = blocks_per_sm * NUM_WARPS
    occupancy = active_warps / max_warps_per_sm
    
    return min(occupancy, 1.0)


def check_coalescing(config: Dict[str, int]) -> bool:
    """
    Check if memory accesses are coalesced
    
    Args:
        config: Kernel configuration
    
    Returns:
        True if coalesced, False otherwise
    """
    HALF2 = config['HALF2']
    D = 64
    
    # With HALF2=1, we load 8 halfs (16 bytes) at a time
    # This is coalesced if consecutive threads access consecutive 16-byte chunks
    # With our layout [BLOCK_M][D], each row is 128 bytes (64 halfs × 2 bytes)
    # Threads in a warp will access consecutive elements → coalesced
    
    if HALF2:
        # 16-byte vectorized loads → coalesced
        return True
    else:
        # Scalar loads may still be coalesced if properly aligned
        # Conservative: require HALF2 for guaranteed coalescing
        return False


def check_bank_conflicts(config: Dict[str, int]) -> int:
    """
    Estimate shared memory bank conflicts
    
    Args:
        config: Kernel configuration
    
    Returns:
        Number of bank conflicts (0 = none)
    """
    SWIZZLE = config['SWIZZLE']
    BLOCK_M = config['BLOCK_M']
    BLOCK_N = config['BLOCK_N']
    D = 64
    
    # With SWIZZLE=1, we add +1 padding → eliminates conflicts
    if SWIZZLE:
        return 0
    
    # Without padding, check if dimensions are multiples of 32 (bank count)
    # If (D % 32 == 0), consecutive threads access the same bank
    if D % 32 == 0:
        return 1  # Conflicts present
    
    return 0  # No conflicts


def hard_gates(meta: Dict[str, Any]) -> Optional[str]:
    """
    Hard validation gates from CUDA optimization doctrine
    
    Rejects configurations that violate fundamental constraints.
    
    Args:
        meta: Dict with keys:
            - config: Kernel configuration
            - coalesced: bool (memory coalescing)
            - bank_conflicts: int (SMEM bank conflict count)
            - occupancy: float (estimated occupancy)
            - peak_mb: float (peak GPU memory in MB)
            - max_rel_err: float (max relative error vs baseline)
            - smem_bytes: float (shared memory usage)
    
    Returns:
        None if passes all gates, else rejection reason string
    """
    config = meta['config']
    
    # Gate 1: Memory coalescing
    if not meta.get('coalesced', True):
        return "bad_coalescing"
    
    # Gate 2: Bank conflicts
    if meta.get('bank_conflicts', 0) > 0:
        return "bank_conflicts"
    
    # Gate 3: Occupancy (allow low if using cp.async for prefetch)
    occupancy = meta.get('occupancy', 0.0)
    cp_async = config.get('CP_ASYNC', 0)
    if occupancy < 0.5 and not cp_async:
        return "low_occupancy_no_prefetch"
    
    # Gate 4: Shared memory budget (L4 has 48KB per SM)
    smem_bytes = meta.get('smem_bytes', 0)
    if smem_bytes > 49152:  # 48 KB
        return "smem_overflow"
    
    # Gate 5: GPU memory (L4 has 23 GB, but leave headroom)
    peak_mb = meta.get('peak_mb', 0.0)
    if peak_mb > 20000:  # 20 GB threshold
        return "oom_risk"
    
    # Gate 6: Numerical correctness (FP16 tolerance)
    max_rel_err = meta.get('max_rel_err', 0.0)
    if max_rel_err > 1e-2:  # 1% relative error
        return "numerics"
    
    # Gate 7: Build success
    if not meta.get('build_success', False):
        return "build_failed"
    
    # Gate 8: Runtime success
    if not meta.get('run_success', False):
        return "runtime_failed"
    
    return None  # Pass all gates


def should_promote(ncu_metrics: Dict[str, float], baseline_ncu: Dict[str, float]) -> bool:
    """
    Decide if candidate should be promoted to full profiling
    
    Based on Nsight metrics showing improved memory behavior
    (memory-first doctrine)
    
    Args:
        ncu_metrics: Nsight metrics for candidate
        baseline_ncu: Nsight metrics for baseline
    
    Returns:
        True if should promote, False otherwise
    """
    # Promote if any of these conditions hold:
    # 1. L2 hit rate improved by ≥8 percentage points
    # 2. DRAM throughput reduced by ≥10 percentage points
    # 3. Tensor core cycles increased by ≥2×
    
    l2_hit_delta = ncu_metrics.get('l2_tex__hit_rate.pct', 0) - \
                   baseline_ncu.get('l2_tex__hit_rate.pct', 0)
    
    dram_delta = baseline_ncu.get('dram__throughput%', 100) - \
                 ncu_metrics.get('dram__throughput%', 100)
    
    tc_ratio = ncu_metrics.get('tensor_cycles%', 0) / \
               max(baseline_ncu.get('tensor_cycles%', 1), 1e-6)
    
    return (
        l2_hit_delta >= 8.0 or
        dram_delta >= 10.0 or
        tc_ratio >= 2.0
    )


def print_search_space_summary():
    """Print summary of search space"""
    total_configs = 1
    for key, values in SEARCH_SPACE.items():
        total_configs *= len(values)
    
    print("=" * 70)
    print("FA-S512 SEARCH SPACE")
    print("=" * 70)
    for key, values in SEARCH_SPACE.items():
        print(f"  {key:12s}: {values}")
    print(f"\n  Total configs: {total_configs:,}")
    print("=" * 70)


if __name__ == "__main__":
    print_search_space_summary()
    
    # Test a config
    test_config = {
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
    
    print("\nTest Configuration:")
    for k, v in test_config.items():
        print(f"  {k}: {v}")
    
    smem = calculate_smem_usage(test_config)
    occ = estimate_occupancy(test_config)
    coal = check_coalescing(test_config)
    banks = check_bank_conflicts(test_config)
    
    print(f"\nEstimates:")
    print(f"  SMEM usage:     {smem:,} bytes ({smem/1024:.1f} KB)")
    print(f"  Occupancy:      {occ:.2%}")
    print(f"  Coalesced:      {coal}")
    print(f"  Bank conflicts: {banks}")
    
    meta = {
        'config': test_config,
        'coalesced': coal,
        'bank_conflicts': banks,
        'occupancy': occ,
        'smem_bytes': smem,
        'peak_mb': 100.0,
        'max_rel_err': 1e-4,
        'build_success': True,
        'run_success': True,
    }
    
    gate_result = hard_gates(meta)
    if gate_result:
        print(f"\n❌ Rejected: {gate_result}")
    else:
        print(f"\n✅ Passes all hard gates")

