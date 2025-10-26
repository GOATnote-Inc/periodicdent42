#!/usr/bin/env python3
# Copyright 2025 GOATnote Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stage Validator: Einstein Framework → Triton Adaptation

Progressive validation for FlashCore Stage 5 development.
Based on Einstein Inversion constraint elimination methodology.

Usage:
    python -m flashcore.validation.stage_validator --stage 1
    python -m flashcore.validation.stage_validator --stage 3
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flashcore.fast.attention_stage5_warpspec import attention_stage5, benchmark_stage5


# ============================================================================
# CONFIGURATION (Einstein Framework Targets)
# ============================================================================

class StageTargets:
    """Performance targets from Einstein performance model"""
    
    # Stage 1: Correctness only
    STAGE1_RTL = 1e-3
    STAGE1_ATOL = 2e-3
    
    # Stage 2: Warp-level sync
    STAGE2_MIN_TFLOPS = 110.0  # 90% of FA2
    
    # Stage 3: Persistent CTAs (batching efficiency)
    STAGE3_MIN_TFLOPS_B1 = 110.0
    STAGE3_MIN_TFLOPS_B8 = 130.0
    STAGE3_MIN_TFLOPS_B32 = 140.0
    STAGE3_MIN_BATCHING_SPEEDUP = 5.0  # B=1 → B=32
    
    # Stage 4: Memory/compute overlap
    STAGE4_MIN_TFLOPS = 180.0  # 95% of FA3
    
    # Stage 5: Beat FA3
    STAGE5_MIN_SPEEDUP_VS_FA3 = 1.05  # 5% faster
    STAGE5_TARGET_TFLOPS = 210.0  # Conservative


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_tflops(time_ms: float, B: int, H: int, S: int, D: int) -> float:
    """Compute TFLOPS from benchmark time"""
    # Attention FLOPs: 4*B*H*S*S*D (Q@K^T + softmax + scores@V)
    flops = 4 * B * H * S * S * D
    tflops = flops / (time_ms / 1000) / 1e12
    return tflops


def get_reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                  is_causal: bool = True) -> torch.Tensor:
    """Get reference output using PyTorch SDPA"""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


# ============================================================================
# STAGE 1: PRODUCER/CONSUMER ARCHITECTURE (CORRECTNESS ONLY)
# ============================================================================

def validate_stage1_correctness(verbose: bool = True) -> bool:
    """
    Stage 1: Producer/Consumer Architecture
    
    Einstein Constraint: None eliminated yet (baseline architecture)
    
    Goal: Correctness only (performance not optimized)
    
    Validates:
    - No crashes
    - Correctness vs PyTorch SDPA (rtol=1e-3, atol=2e-3)
    - Basic functionality
    
    Returns:
        bool: True if stage passes
    """
    if verbose:
        print("\n" + "="*80)
        print("STAGE 1: PRODUCER/CONSUMER ARCHITECTURE (CORRECTNESS)")
        print("="*80)
        print("\nEinstein Constraint Eliminated: None (baseline)")
        print("Expected Gain: N/A (correctness gate)")
        print()
    
    # Test configuration
    B, H, S, D = 16, 16, 2048, 64
    
    if verbose:
        print(f"Test config: B={B}, H={H}, S={S}, D={D}")
        print()
    
    # Create inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Get reference
    if verbose:
        print("[1/3] Computing reference (PyTorch SDPA)...")
    ref = get_reference(Q, K, V, is_causal=True)
    
    # Test our kernel
    if verbose:
        print("[2/3] Testing FlashCore Stage 5...")
    try:
        out = attention_stage5(Q, K, V, is_causal=True, use_warp_spec=False)
    except Exception as e:
        if verbose:
            print(f"❌ FAILED: Kernel crashed with error: {e}")
        return False
    
    # Check correctness
    if verbose:
        print("[3/3] Checking correctness...")
    
    max_diff = (out - ref).abs().max().item()
    mean_diff = (out - ref).abs().mean().item()
    correct = torch.allclose(out, ref, rtol=StageTargets.STAGE1_RTL, atol=StageTargets.STAGE1_ATOL)
    
    # Results
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Max diff:     {max_diff:.6f}")
        print(f"Mean diff:    {mean_diff:.6f}")
        print(f"Tolerance:    rtol={StageTargets.STAGE1_RTL}, atol={StageTargets.STAGE1_ATOL}")
        print(f"Correctness:  {'✅ PASS' if correct else '❌ FAIL'}")
        
        if correct:
            print(f"\n✅ STAGE 1 PASSED")
            print(f"   Architecture: ✅ Correct")
            print(f"   Ready for Stage 2 (warp-level sync)")
        else:
            print(f"\n❌ STAGE 1 FAILED")
            print(f"   Fix correctness before proceeding to Stage 2")
    
    return correct


# ============================================================================
# STAGE 2: WARP-LEVEL SYNCHRONIZATION
# ============================================================================

def validate_stage2_warp_sync(verbose: bool = True) -> bool:
    """
    Stage 2: Warp-Level Synchronization
    
    Einstein Constraint #3: Global sync (__syncthreads) → Warp-level sync
    
    Expected Gain: +2-3% from eliminating 200+ cycle barriers
    
    Validates:
    - Correctness maintained
    - Performance improvement (~110 TFLOPS)
    - Reduced synchronization overhead
    
    Returns:
        bool: True if stage passes
    """
    if verbose:
        print("\n" + "="*80)
        print("STAGE 2: WARP-LEVEL SYNCHRONIZATION")
        print("="*80)
        print("\nEinstein Constraint #3: Global Sync → Warp Sync")
        print("Expected Gain: +2-3% from eliminating __syncthreads")
        print()
    
    # Test configuration
    B, H, S, D = 16, 16, 2048, 64
    
    # Create inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Get reference
    ref = get_reference(Q, K, V, is_causal=True)
    
    # Benchmark our kernel (with warp-spec enabled)
    if verbose:
        print("[1/2] Benchmarking FlashCore (warp-spec enabled)...")
    
    config = {
        'B': B, 'H': H, 'S': S, 'D': D,
        'use_warp_spec': True,  # Enable warp specialization
        'num_producer_warps': 2,
        'use_fast_exp': False,
    }
    
    results = benchmark_stage5(config, warmup=20, iters=100)
    out = attention_stage5(Q, K, V, is_causal=True, use_warp_spec=True)
    
    # Check correctness
    correct = torch.allclose(out, ref, rtol=StageTargets.STAGE1_RTL, atol=StageTargets.STAGE1_ATOL)
    
    # Compute TFLOPS
    tflops = compute_tflops(results['p50'], B, H, S, D)
    
    # Compare to FA2 (if available)
    if verbose:
        print("[2/2] Comparing to PyTorch SDPA...")
    
    # Benchmark PyTorch SDPA
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(20):
        _ = get_reference(Q, K, V)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(100):
        start.record()
        _ = get_reference(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    sdpa_time = np.median(times)
    sdpa_tflops = compute_tflops(sdpa_time, B, H, S, D)
    
    # Results
    passed = correct and tflops >= StageTargets.STAGE2_MIN_TFLOPS
    
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Correctness:        {'✅ PASS' if correct else '❌ FAIL'}")
        print(f"FlashCore TFLOPS:   {tflops:.1f}")
        print(f"PyTorch TFLOPS:     {sdpa_tflops:.1f}")
        print(f"Speedup:            {tflops/sdpa_tflops:.2f}×")
        print(f"Target:             {StageTargets.STAGE2_MIN_TFLOPS} TFLOPS")
        print(f"Target met:         {'✅' if tflops >= StageTargets.STAGE2_MIN_TFLOPS else '❌'}")
        
        if passed:
            print(f"\n✅ STAGE 2 PASSED")
            print(f"   Warp-level sync: ✅")
            print(f"   Performance: {tflops:.1f} TFLOPS ({tflops/sdpa_tflops*100:.1f}% of PyTorch)")
            print(f"   Ready for Stage 3 (persistent CTAs)")
        else:
            print(f"\n❌ STAGE 2 FAILED")
            if not correct:
                print("   - Correctness issue")
            if tflops < StageTargets.STAGE2_MIN_TFLOPS:
                print(f"   - Performance too low ({tflops:.1f} < {StageTargets.STAGE2_MIN_TFLOPS})")
    
    return passed


# ============================================================================
# STAGE 3: PERSISTENT CTAs (BATCHING EFFICIENCY)
# ============================================================================

def validate_stage3_batching(verbose: bool = True) -> bool:
    """
    Stage 3: Persistent CTAs
    
    Einstein Constraint #2: Launch overhead (40%) → Amortized (2%)
    
    Expected Gain: 6× speedup (B=1 → B=32 per-sequence)
    
    Validates:
    - Batching efficiency (5× speedup B=1 → B=32)
    - Performance scales sublinearly with batch size
    - Launch overhead amortization
    
    Returns:
        bool: True if stage passes
    """
    if verbose:
        print("\n" + "="*80)
        print("STAGE 3: PERSISTENT CTA BATCHING")
        print("="*80)
        print("\nEinstein Constraint #2: Launch Overhead → Persistent CTAs")
        print("Expected Gain: 6× speedup (B=1 → B=32)")
        print()
    
    batch_sizes = [1, 8, 32]
    results = {}
    
    H, S, D = 16, 2048, 64
    
    for B in batch_sizes:
        if verbose:
            print(f"[Testing B={B}]")
        
        torch.manual_seed(42)
        Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        
        # Get reference
        ref = get_reference(Q, K, V)
        
        # Benchmark
        config = {'B': B, 'H': H, 'S': S, 'D': D, 'use_warp_spec': True}
        perf = benchmark_stage5(config, warmup=20, iters=100)
        out = attention_stage5(Q, K, V, is_causal=True, use_warp_spec=True)
        
        # Check correctness
        correct = torch.allclose(out, ref, rtol=StageTargets.STAGE1_RTL, atol=StageTargets.STAGE1_ATOL)
        
        # Compute metrics
        tflops = compute_tflops(perf['p50'], B, H, S, D)
        latency_per_seq = perf['p50'] / B
        
        results[B] = {
            'time_ms': perf['p50'],
            'tflops': tflops,
            'latency_per_seq_ms': latency_per_seq,
            'correct': correct
        }
        
        if verbose:
            print(f"  Time:            {perf['p50']:.2f} ms")
            print(f"  TFLOPS:          {tflops:.1f}")
            print(f"  Latency/seq:     {latency_per_seq:.2f} ms")
            print(f"  Correctness:     {'✅' if correct else '❌'}")
    
    # Compute batching efficiency
    speedup_8 = results[1]['latency_per_seq_ms'] / results[8]['latency_per_seq_ms']
    speedup_32 = results[1]['latency_per_seq_ms'] / results[32]['latency_per_seq_ms']
    
    # Check targets
    all_correct = all(r['correct'] for r in results.values())
    batching_ok = speedup_32 >= StageTargets.STAGE3_MIN_BATCHING_SPEEDUP
    tflops_ok = (
        results[1]['tflops'] >= StageTargets.STAGE3_MIN_TFLOPS_B1 and
        results[8]['tflops'] >= StageTargets.STAGE3_MIN_TFLOPS_B8 and
        results[32]['tflops'] >= StageTargets.STAGE3_MIN_TFLOPS_B32
    )
    
    passed = all_correct and batching_ok and tflops_ok
    
    if verbose:
        print(f"\n{'='*80}")
        print("BATCHING EFFICIENCY ANALYSIS")
        print(f"{'='*80}")
        print(f"B=1 → B=8 speedup:     {speedup_8:.2f}× (target: >2.5×)")
        print(f"B=1 → B=32 speedup:    {speedup_32:.2f}× (target: >{StageTargets.STAGE3_MIN_BATCHING_SPEEDUP}×)")
        print(f"B=32 TFLOPS:           {results[32]['tflops']:.1f} (target: {StageTargets.STAGE3_MIN_TFLOPS_B32})")
        
        if passed:
            print(f"\n✅ STAGE 3 PASSED")
            print(f"   Persistent CTAs: ✅")
            print(f"   Batching efficiency: {speedup_32:.1f}×")
            print(f"   Ready for Stage 4 (memory overlap)")
        else:
            print(f"\n❌ STAGE 3 FAILED")
            if not all_correct:
                print("   - Correctness issues detected")
            if not batching_ok:
                print(f"   - Batching efficiency too low ({speedup_32:.1f}× < {StageTargets.STAGE3_MIN_BATCHING_SPEEDUP}×)")
            if not tflops_ok:
                print(f"   - Performance targets not met")
    
    return passed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate FlashCore Stage 5 development (Einstein Framework)'
    )
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                       help='Stage to validate (1-3 implemented so far)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Run appropriate validator
    validators = {
        1: validate_stage1_correctness,
        2: validate_stage2_warp_sync,
        3: validate_stage3_batching,
    }
    
    if args.stage not in validators:
        print(f"❌ Stage {args.stage} validation not yet implemented")
        print(f"   Available: {list(validators.keys())}")
        sys.exit(1)
    
    passed = validators[args.stage](verbose=verbose)
    
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()

