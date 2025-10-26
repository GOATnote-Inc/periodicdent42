#!/usr/bin/env python3
"""
Determinism Validator - 1000-trial bitwise reproducibility
Validates: Kernel produces identical outputs across runs (no race conditions)

Based on: MLPerf Inference reproducibility requirements
Used by: NVIDIA kernel QA, FlashAttention validation
"""

import torch
import numpy as np
from typing import Dict, Callable
import json
from pathlib import Path


def validate_determinism(
    kernel_fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_trials: int = 1000,
    device: str = 'cuda'
) -> Dict:
    """
    Validate bitwise determinism across multiple runs
    
    Args:
        kernel_fn: Attention kernel to test
        q, k, v: Input tensors
        n_trials: Number of trials (1000 = production standard)
        device: Device to run on
    
    Returns:
        Dictionary with determinism metrics
    """
    print(f"Running {n_trials}-trial determinism validation...")
    
    # Fix all random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    outputs = []
    timings = []
    
    for trial in range(n_trials):
        torch.cuda.synchronize()
        
        # Device-time measurement
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        out = kernel_fn(q, k, v)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_us = start.elapsed_time(end) * 1000  # Convert ms to μs
        
        outputs.append(out.clone())
        timings.append(elapsed_us)
        
        if (trial + 1) % 100 == 0:
            print(f"  Progress: {trial + 1}/{n_trials} trials")
    
    # Bitwise comparison
    reference = outputs[0]
    deterministic = all(torch.equal(reference, out) for out in outputs[1:])
    
    # Statistical analysis
    timings_array = np.array(timings)
    
    report = {
        'deterministic': deterministic,
        'trials': n_trials,
        'mean_latency_us': float(np.mean(timings_array)),
        'std_latency_us': float(np.std(timings_array)),
        'jitter_percent': float(100 * np.std(timings_array) / np.mean(timings_array)),
        'min_us': float(np.min(timings_array)),
        'max_us': float(np.max(timings_array)),
        'p50_us': float(np.percentile(timings_array, 50)),
        'p95_us': float(np.percentile(timings_array, 95)),
        'p99_us': float(np.percentile(timings_array, 99)),
    }
    
    # Production criteria
    report['PASS_determinism'] = deterministic
    report['PASS_jitter'] = report['jitter_percent'] < 1.0  # < 1% jitter acceptable
    
    return report


def validate_production_kernels():
    """Validate all production FlashCore kernels"""
    from flashcore.fast.attention_production import attention as prod_attention
    from flashcore.fast.attention_multihead import multihead_attention
    
    print("=" * 80)
    print("DETERMINISM VALIDATION - FlashCore Production Kernels")
    print("=" * 80)
    print()
    
    # Test configuration
    B, H, S, D = 16, 8, 512, 64
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = q.clone()
    v = q.clone()
    
    results = {}
    
    # Test 1: Production attention kernel
    print("Testing: attention_production.py")
    results['production'] = validate_determinism(prod_attention, q, k, v, n_trials=1000)
    
    print()
    
    # Test 2: Multi-head attention kernel
    print("Testing: attention_multihead.py")
    results['multihead'] = validate_determinism(multihead_attention, q, k, v, n_trials=1000)
    
    # Generate report
    print()
    print("=" * 80)
    print("DETERMINISM VALIDATION REPORT")
    print("=" * 80)
    
    all_pass = True
    for kernel_name, report in results.items():
        status = "✅ PASS" if (report['PASS_determinism'] and report['PASS_jitter']) else "❌ FAIL"
        print(f"\n{kernel_name}:")
        print(f"  Deterministic: {report['deterministic']}")
        print(f"  Mean Latency: {report['mean_latency_us']:.2f} μs")
        print(f"  Jitter: {report['jitter_percent']:.3f}%")
        print(f"  P99 Latency: {report['p99_us']:.2f} μs")
        print(f"  Status: {status}")
        
        if not (report['PASS_determinism'] and report['PASS_jitter']):
            all_pass = False
    
    # Save report
    output_file = Path('logs/determinism_validation.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    final_report = {
        'overall_status': 'PASS' if all_pass else 'FAIL',
        'kernels': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print()
    print(f"Report saved: {output_file}")
    print()
    
    if all_pass:
        print("✅ ALL KERNELS PASS DETERMINISM VALIDATION")
        return 0
    else:
        print("❌ DETERMINISM VALIDATION FAILED")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = validate_production_kernels()
    sys.exit(exit_code)

