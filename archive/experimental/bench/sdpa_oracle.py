#!/usr/bin/env python3
"""
SDPA Oracle: Hard gate for kernel correctness and performance

Definition of Success:
- Median latency < 0.95× SDPA (5% speedup minimum)
- Correctness: max |Δ| ≤ 2e-3 vs SDPA reference
- Bootstrap CI: 95% confidence interval must be negative (candidate < SDPA)
- Shape: B=1, H=8, S=512, D=64 (L4-specialized)
"""

import torch
import torch.nn.functional as F
import time
import statistics as stats
import numpy as np
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, List

@contextmanager
def sdp_backend(backends: Dict[str, bool]):
    """Context manager for SDPA backend selection"""
    with torch.backends.cuda.sdp_kernel(**backends):
        yield

def sdpa_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
             scale: float, backends: Dict[str, bool]) -> torch.Tensor:
    """Reference SDPA with explicit backend selection"""
    with sdp_backend(backends):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

def bench_ms(fn, iters: int = 120, warmup: int = 20) -> Tuple[float, List[float]]:
    """
    Benchmark function with warmup
    
    Returns:
        median_ms: Median latency in milliseconds
        samples: All timing samples
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Timing
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    
    return stats.median(samples), samples

def bootstrap_ci(samples_a: List[float], samples_b: List[float], 
                 n_bootstrap: int = 10000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for median difference (A - B)
    
    Returns:
        median_diff: Median of (A - B)
        ci_lower: Lower bound of 95% CI
        ci_upper: Upper bound of 95% CI
    """
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    diffs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        a_sample = rng.choice(samples_a, size=len(samples_a), replace=True)
        b_sample = rng.choice(samples_b, size=len(samples_b), replace=True)
        
        # Compute median difference
        diffs.append(np.median(a_sample) - np.median(b_sample))
    
    # Percentile confidence interval
    ci_lower = np.percentile(diffs, alpha * 100 / 2)
    ci_upper = np.percentile(diffs, 100 - alpha * 100 / 2)
    median_diff = np.median(diffs)
    
    return median_diff, ci_lower, ci_upper

def check_correctness(candidate_out: torch.Tensor, 
                     ref_out: torch.Tensor, 
                     tol: float = 2e-3) -> Tuple[bool, float]:
    """
    Check correctness against reference
    
    Returns:
        passed: True if max |Δ| ≤ tol
        max_diff: Maximum absolute difference
    """
    diff = (candidate_out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    passed = max_diff <= tol
    
    return passed, max_diff

def evaluate_candidate(
    candidate_fn,
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    scale: float,
    sdpa_backend: str = "flash",  # "flash" or "math"
    iters: int = 100,
    warmup: int = 20,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    speedup_threshold: float = 0.95,
    correctness_tol: float = 2e-3
) -> Dict:
    """
    Evaluate candidate kernel against SDPA with hard gate
    
    Args:
        candidate_fn: Function that returns output tensor
        q, k, v: Input tensors [B, H, S, D]
        scale: Attention scale factor
        sdpa_backend: "flash" or "math" SDPA backend
        iters: Number of benchmark iterations
        warmup: Number of warmup iterations
        n_bootstrap: Bootstrap samples for CI
        alpha: Significance level (0.05 = 95% CI)
        speedup_threshold: Candidate must be < this × SDPA (0.95 = 5% faster)
        correctness_tol: Max |Δ| tolerance
    
    Returns:
        results: Dict with pass/fail and all metrics
    """
    # Select SDPA backend
    backends = {
        "enable_flash": sdpa_backend == "flash",
        "enable_math": sdpa_backend == "math",
        "enable_mem_efficient": False
    }
    
    # Correctness check
    with torch.no_grad():
        candidate_out = candidate_fn()
        ref_out = sdpa_ref(q, k, v, scale, backends)
    
    correct_passed, max_diff = check_correctness(candidate_out, ref_out, correctness_tol)
    
    # Performance benchmark
    candidate_median, candidate_samples = bench_ms(candidate_fn, iters, warmup)
    sdpa_median, sdpa_samples = bench_ms(
        lambda: sdpa_ref(q, k, v, scale, backends), iters, warmup
    )
    
    # Bootstrap confidence interval
    median_diff, ci_lower, ci_upper = bootstrap_ci(
        candidate_samples, sdpa_samples, n_bootstrap, alpha
    )
    
    # Hard gate checks
    speedup = sdpa_median / candidate_median
    perf_passed = candidate_median < (speedup_threshold * sdpa_median)
    ci_passed = ci_upper < 0  # 95% CI must be entirely negative (candidate < SDPA)
    
    gate_passed = correct_passed and perf_passed and ci_passed
    
    results = {
        "passed": gate_passed,
        "correctness": {
            "passed": correct_passed,
            "max_diff": max_diff,
            "tolerance": correctness_tol
        },
        "performance": {
            "candidate_median_ms": candidate_median,
            "sdpa_median_ms": sdpa_median,
            "speedup": speedup,
            "passed": perf_passed,
            "threshold": speedup_threshold
        },
        "bootstrap": {
            "median_diff_ms": median_diff,
            "ci_lower_ms": ci_lower,
            "ci_upper_ms": ci_upper,
            "alpha": alpha,
            "n_bootstrap": n_bootstrap,
            "ci_passed": ci_passed
        },
        "config": {
            "sdpa_backend": sdpa_backend,
            "shape": list(q.shape),
            "dtype": str(q.dtype),
            "device": str(q.device)
        }
    }
    
    return results

def print_results(results: Dict):
    """Pretty-print evaluation results"""
    print("=" * 70)
    print("SDPA Oracle Evaluation")
    print("=" * 70)
    
    # Overall status
    status = "✅ PASSED" if results["passed"] else "❌ FAILED"
    print(f"\n{status}\n")
    
    # Correctness
    corr = results["correctness"]
    corr_status = "✅" if corr["passed"] else "❌"
    print(f"{corr_status} Correctness: max_diff={corr['max_diff']:.6f} (tol={corr['tolerance']:.6f})")
    
    # Performance
    perf = results["performance"]
    perf_status = "✅" if perf["passed"] else "❌"
    print(f"{perf_status} Performance:")
    print(f"   Candidate: {perf['candidate_median_ms']:.4f} ms")
    print(f"   SDPA:      {perf['sdpa_median_ms']:.4f} ms")
    print(f"   Speedup:   {perf['speedup']:.3f}× {'(FASTER)' if perf['speedup'] > 1 else '(SLOWER)'}")
    print(f"   Target:    >{1.0/perf['threshold']:.3f}× (< {perf['threshold']:.3f}× SDPA)")
    
    # Bootstrap CI
    boot = results["bootstrap"]
    ci_status = "✅" if boot["ci_passed"] else "❌"
    print(f"{ci_status} Bootstrap CI ({100-boot['alpha']*100:.0f}%):")
    print(f"   Median Δ:  {boot['median_diff_ms']:.4f} ms")
    print(f"   CI:        [{boot['ci_lower_ms']:.4f}, {boot['ci_upper_ms']:.4f}] ms")
    print(f"   CI < 0:    {boot['ci_passed']} {'(statistically faster)' if boot['ci_passed'] else '(not significant)'}")
    
    # Config
    cfg = results["config"]
    print(f"\nConfig:")
    print(f"   Backend:   {cfg['sdpa_backend']}")
    print(f"   Shape:     {cfg['shape']}")
    print(f"   Dtype:     {cfg['dtype']}")
    print(f"   Device:    {cfg['device']}")
    
    print("=" * 70)

if __name__ == "__main__":
    # Smoke test
    print("SDPA Oracle Smoke Test")
    print("-" * 70)
    
    # Test shape: B=1, H=8, S=512, D=64
    B, H, S, D = 1, 8, 512, 64
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)
    
    # Test candidate = SDPA (should pass with speedup ~1.0)
    def candidate():
        return sdpa_ref(q, k, v, scale, {"enable_flash": True, "enable_math": False, "enable_mem_efficient": False})
    
    results = evaluate_candidate(
        candidate,
        q, k, v, scale,
        sdpa_backend="flash",
        iters=100,
        warmup=20,
        n_bootstrap=1000,  # Reduced for smoke test
        speedup_threshold=0.95
    )
    
    print_results(results)

