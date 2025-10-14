#!/usr/bin/env python3
"""
Correctness Fuzzing for Custom Attention Kernels

Compares custom kernel output against PyTorch SDPA (oracle) across a range
of jittered shapes to ensure correctness before performance benchmarking.

Test Matrix:
- S ∈ {448, 512, 640}
- B ∈ {16, 32, 48}
- H ∈ {4, 8, 16}
- D = 64 (fixed)

Tolerances (FP16):
- atol = 2e-3 (absolute)
- rtol = 1e-3 (relative)

Exit Codes:
- 0: All tests passed
- 1: At least one test failed
- 2: Custom kernel not found (skipped)

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-14
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cudadent42.bench.common.env_lock import lock_environment


@dataclass
class CorrectnessResult:
    """Result of a single correctness test"""
    config: Dict[str, int]
    passed: bool
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    oracle_norm: float
    custom_norm: float
    time_oracle_ms: float
    time_custom_ms: float
    error_message: Optional[str] = None


def compare_outputs(
    oracle: torch.Tensor,
    custom: torch.Tensor,
    atol: float = 2e-3,
    rtol: float = 1e-3
) -> Tuple[bool, Dict[str, float]]:
    """
    Compare oracle and custom kernel outputs
    
    Args:
        oracle: Reference output from PyTorch SDPA
        custom: Output from custom kernel
        atol: Absolute tolerance
        rtol: Relative tolerance
    
    Returns:
        (passed, metrics) where metrics contains error statistics
    """
    # Compute errors
    abs_error = torch.abs(oracle - custom)
    rel_error = abs_error / (torch.abs(oracle) + 1e-8)
    
    max_abs = float(torch.max(abs_error))
    max_rel = float(torch.max(rel_error))
    mean_abs = float(torch.mean(abs_error))
    
    oracle_norm = float(torch.norm(oracle))
    custom_norm = float(torch.norm(custom))
    
    # Check tolerance
    passed = torch.allclose(oracle, custom, atol=atol, rtol=rtol)
    
    metrics = {
        'max_abs_error': max_abs,
        'max_rel_error': max_rel,
        'mean_abs_error': mean_abs,
        'oracle_norm': oracle_norm,
        'custom_norm': custom_norm
    }
    
    return passed, metrics


def run_sdpa_oracle(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """
    Run PyTorch SDPA (oracle)
    
    Args:
        Q, K, V: Input tensors (B, H, S, D)
    
    Returns:
        (output, time_ms)
    """
    # Warmup
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        for _ in range(5):
            _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        output = F.scaled_dot_product_attention(Q, K, V)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    return output, elapsed_ms


def run_custom_kernel(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    module: torch.nn.Module
) -> Tuple[torch.Tensor, float]:
    """
    Run custom kernel
    
    Args:
        Q, K, V: Input tensors (B, H, S, D)
        module: Custom kernel module
    
    Returns:
        (output, time_ms)
    """
    # Warmup
    for _ in range(5):
        _ = module.forward(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    output = module.forward(Q, K, V)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    return output, elapsed_ms


def test_correctness_single(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    custom_module: Optional[torch.nn.Module] = None,
    atol: float = 2e-3,
    rtol: float = 1e-3,
    seed: int = 42
) -> CorrectnessResult:
    """
    Test correctness for a single configuration
    
    Args:
        batch, heads, seq, dim: Tensor dimensions
        custom_module: Custom kernel module (if None, will try to import)
        atol, rtol: Tolerances
        seed: Random seed
    
    Returns:
        CorrectnessResult with pass/fail and error metrics
    """
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Create inputs
    Q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    
    config = {'batch': batch, 'heads': heads, 'seq': seq, 'dim': dim}
    
    try:
        # Run oracle
        oracle_output, oracle_time = run_sdpa_oracle(Q, K, V)
        
        # Run custom kernel (if available)
        if custom_module is None:
            return CorrectnessResult(
                config=config,
                passed=False,
                max_abs_error=0.0,
                max_rel_error=0.0,
                mean_abs_error=0.0,
                oracle_norm=float(torch.norm(oracle_output)),
                custom_norm=0.0,
                time_oracle_ms=oracle_time,
                time_custom_ms=0.0,
                error_message="Custom module not provided"
            )
        
        custom_output, custom_time = run_custom_kernel(Q, K, V, custom_module)
        
        # Compare outputs
        passed, metrics = compare_outputs(oracle_output, custom_output, atol=atol, rtol=rtol)
        
        return CorrectnessResult(
            config=config,
            passed=passed,
            max_abs_error=metrics['max_abs_error'],
            max_rel_error=metrics['max_rel_error'],
            mean_abs_error=metrics['mean_abs_error'],
            oracle_norm=metrics['oracle_norm'],
            custom_norm=metrics['custom_norm'],
            time_oracle_ms=oracle_time,
            time_custom_ms=custom_time
        )
    
    except Exception as e:
        return CorrectnessResult(
            config=config,
            passed=False,
            max_abs_error=float('inf'),
            max_rel_error=float('inf'),
            mean_abs_error=float('inf'),
            oracle_norm=0.0,
            custom_norm=0.0,
            time_oracle_ms=0.0,
            time_custom_ms=0.0,
            error_message=str(e)
        )


def run_correctness_fuzz(
    custom_module: Optional[torch.nn.Module] = None,
    atol: float = 2e-3,
    rtol: float = 1e-3
) -> Tuple[List[CorrectnessResult], bool]:
    """
    Run correctness fuzzing across test matrix
    
    Args:
        custom_module: Custom kernel module to test
        atol, rtol: Tolerances
    
    Returns:
        (results, all_passed)
    """
    # Test matrix
    seq_lengths = [448, 512, 640]
    batch_sizes = [16, 32, 48]
    head_counts = [4, 8, 16]
    dim = 64  # Fixed
    
    results = []
    
    print("="*70)
    print("CORRECTNESS FUZZING: Custom Kernel vs PyTorch SDPA")
    print("="*70)
    print()
    print(f"Test Matrix:")
    print(f"  S ∈ {seq_lengths}")
    print(f"  B ∈ {batch_sizes}")
    print(f"  H ∈ {head_counts}")
    print(f"  D = {dim}")
    print(f"  Tolerances: atol={atol}, rtol={rtol}")
    print(f"  Total tests: {len(seq_lengths) * len(batch_sizes) * len(head_counts)}")
    print()
    
    if custom_module is None:
        print("⚠️  No custom module provided - testing oracle only")
        print()
    
    total = len(seq_lengths) * len(batch_sizes) * len(head_counts)
    test_num = 0
    
    for seq in seq_lengths:
        for batch in batch_sizes:
            for heads in head_counts:
                test_num += 1
                print(f"[{test_num}/{total}] Testing B={batch}, H={heads}, S={seq}, D={dim}...", end=" ")
                
                result = test_correctness_single(
                    batch=batch,
                    heads=heads,
                    seq=seq,
                    dim=dim,
                    custom_module=custom_module,
                    atol=atol,
                    rtol=rtol
                )
                
                results.append(result)
                
                if result.passed:
                    speedup = result.time_oracle_ms / result.time_custom_ms if result.time_custom_ms > 0 else 0
                    print(f"✅ PASS (max_err={result.max_abs_error:.2e}, speedup={speedup:.2f}×)")
                else:
                    if result.error_message:
                        print(f"❌ FAIL ({result.error_message})")
                    else:
                        print(f"❌ FAIL (max_abs={result.max_abs_error:.2e}, max_rel={result.max_rel_error:.2e})")
    
    all_passed = all(r.passed or r.error_message == "Custom module not provided" for r in results)
    
    return results, all_passed


def print_summary(results: List[CorrectnessResult]):
    """Print summary table of results"""
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    # Count passed/failed
    tested_results = [r for r in results if r.error_message != "Custom module not provided"]
    
    if not tested_results:
        print("⚠️  No tests run (custom module not available)")
        return
    
    passed = sum(1 for r in tested_results if r.passed)
    failed = len(tested_results) - passed
    
    print(f"Tests run: {len(tested_results)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print()
    
    if passed > 0:
        # Error statistics
        max_errors = [r.max_abs_error for r in tested_results if r.passed]
        mean_errors = [r.mean_abs_error for r in tested_results if r.passed]
        
        print(f"Error Statistics (passed tests):")
        print(f"  Max absolute error:")
        print(f"    Min:    {min(max_errors):.2e}")
        print(f"    Median: {np.median(max_errors):.2e}")
        print(f"    Max:    {max(max_errors):.2e}")
        print(f"  Mean absolute error:")
        print(f"    Min:    {min(mean_errors):.2e}")
        print(f"    Median: {np.median(mean_errors):.2e}")
        print(f"    Max:    {max(mean_errors):.2e}")
        print()
        
        # Speedup statistics
        speedups = [r.time_oracle_ms / r.time_custom_ms for r in tested_results if r.passed and r.time_custom_ms > 0]
        if speedups:
            print(f"Speedup vs PyTorch SDPA:")
            print(f"  Min:    {min(speedups):.3f}×")
            print(f"  Median: {np.median(speedups):.3f}×")
            print(f"  Max:    {max(speedups):.3f}×")
            print()
    
    if failed > 0:
        print("Failed Tests:")
        for r in tested_results:
            if not r.passed:
                cfg = r.config
                print(f"  B={cfg['batch']}, H={cfg['heads']}, S={cfg['seq']}, D={cfg['dim']}")
                if r.error_message:
                    print(f"    Error: {r.error_message}")
                else:
                    print(f"    Max abs error: {r.max_abs_error:.2e} (threshold: 2e-3)")
                    print(f"    Max rel error: {r.max_rel_error:.2e} (threshold: 1e-3)")
        print()


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Correctness fuzzing for custom attention kernels")
    parser.add_argument("--module", help="Path to custom kernel module (e.g., fa_s512.so)")
    parser.add_argument("--atol", type=float, default=2e-3, help="Absolute tolerance (default: 2e-3)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance (default: 1e-3)")
    
    args = parser.parse_args()
    
    # Lock environment
    lock_environment()
    assert torch.backends.cuda.matmul.allow_tf32 == False, "TF32 not disabled!"
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Environment: FP16, TF32 off, deterministic")
    print()
    
    # Load custom module (if provided)
    custom_module = None
    if args.module:
        try:
            # TODO: Implement module loading
            print(f"Loading custom module: {args.module}")
            # custom_module = torch.load(args.module)
            print("⚠️  Custom module loading not yet implemented")
        except Exception as e:
            print(f"❌ Failed to load custom module: {e}")
            return 2
    
    # Run fuzzing
    start_time = time.time()
    results, all_passed = run_correctness_fuzz(custom_module=custom_module, atol=args.atol, rtol=args.rtol)
    elapsed = time.time() - start_time
    
    # Print summary
    print_summary(results)
    
    print(f"Total time: {elapsed:.1f} seconds")
    print()
    
    if custom_module is None:
        print("✅ Oracle test complete (no custom kernel to validate)")
        return 2  # Skipped
    elif all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

