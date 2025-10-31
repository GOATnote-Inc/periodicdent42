"""Validation utilities for BlackwellSparseK kernels."""

import torch
from typing import Tuple, Dict, Any


def validate_correctness(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 2e-3,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate kernel output against reference using FP16-appropriate tolerances.
    
    Args:
        output: Kernel output tensor
        reference: Reference output tensor (e.g., from PyTorch SDPA)
        rtol: Relative tolerance (default: 1e-3 for FP16)
        atol: Absolute tolerance (default: 2e-3 for FP16)
    
    Returns:
        (is_correct, metrics_dict)
    
    Example:
        >>> from blackwell_sparsek import attention_forward
        >>> from blackwell_sparsek.utils import validate_correctness
        >>> 
        >>> Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> 
        >>> ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        >>> out = attention_forward(Q, K, V)
        >>> 
        >>> is_correct, metrics = validate_correctness(out, ref)
        >>> print(f"Correct: {is_correct}, Max diff: {metrics['max_diff']:.6f}")
    """
    # Compute differences
    diff = torch.abs(output - reference)
    
    metrics = {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "median_diff": diff.median().item(),
        "num_elements": diff.numel(),
    }
    
    # Check allclose (standard method for FP16)
    is_correct = torch.allclose(output, reference, rtol=rtol, atol=atol)
    
    # Additional metrics
    if not is_correct:
        # Find worst mismatches
        threshold = atol + rtol * torch.abs(reference)
        mismatches = diff > threshold
        metrics["num_mismatches"] = mismatches.sum().item()
        metrics["mismatch_rate"] = mismatches.float().mean().item()
        
        if mismatches.any():
            worst_idx = diff.argmax()
            metrics["worst_diff_location"] = tuple(
                torch.unravel_index(worst_idx, diff.shape)
            )
            metrics["worst_diff_output"] = output.flatten()[worst_idx].item()
            metrics["worst_diff_reference"] = reference.flatten()[worst_idx].item()
    
    return is_correct, metrics


def compare_to_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kernel_fn,
    scale: float = None,
    rtol: float = 1e-3,
    atol: float = 2e-3,
) -> Dict[str, Any]:
    """
    Compare custom kernel to PyTorch SDPA baseline.
    
    Args:
        Q: Query tensor [B, H, S, D]
        K: Key tensor [B, H, S, D]
        V: Value tensor [B, H, S, D]
        kernel_fn: Custom kernel function
        scale: Softmax scale (default: 1/sqrt(D))
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Dictionary with comparison results
    """
    import time
    
    # Default scale
    if scale is None:
        D = Q.shape[-1]
        scale = 1.0 / (D ** 0.5)
    
    # PyTorch SDPA reference
    torch.cuda.synchronize()
    start = time.perf_counter()
    ref_output = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, scale=scale
    )
    torch.cuda.synchronize()
    sdpa_time = time.perf_counter() - start
    
    # Custom kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    kernel_output = kernel_fn(Q, K, V, scale=scale)
    torch.cuda.synchronize()
    kernel_time = time.perf_counter() - start
    
    # Validate correctness
    is_correct, metrics = validate_correctness(
        kernel_output, ref_output, rtol=rtol, atol=atol
    )
    
    # Compute speedup
    speedup = sdpa_time / kernel_time if kernel_time > 0 else 0.0
    
    return {
        "is_correct": is_correct,
        "correctness_metrics": metrics,
        "sdpa_time_us": sdpa_time * 1e6,
        "kernel_time_us": kernel_time * 1e6,
        "speedup": speedup,
        "input_shape": {
            "B": Q.shape[0],
            "H": Q.shape[1],
            "S": Q.shape[2],
            "D": Q.shape[3],
        },
    }


def print_comparison_summary(comparison: Dict[str, Any]):
    """Pretty-print comparison results."""
    print("=" * 80)
    print("BlackwellSparseK vs PyTorch SDPA Comparison")
    print("=" * 80)
    
    # Input shape
    shape = comparison["input_shape"]
    print(f"\nInput Shape: [B={shape['B']}, H={shape['H']}, S={shape['S']}, D={shape['D']}]")
    
    # Performance
    print("\nPerformance:")
    print(f"  PyTorch SDPA:       {comparison['sdpa_time_us']:8.2f} μs")
    print(f"  BlackwellSparseK:   {comparison['kernel_time_us']:8.2f} μs")
    print(f"  Speedup:            {comparison['speedup']:8.2f}×")
    
    # Correctness
    print("\nCorrectness:")
    print(f"  Status:     {'✅ PASS' if comparison['is_correct'] else '❌ FAIL'}")
    metrics = comparison["correctness_metrics"]
    print(f"  Max Diff:   {metrics['max_diff']:.6f}")
    print(f"  Mean Diff:  {metrics['mean_diff']:.6f}")
    
    if not comparison["is_correct"] and "num_mismatches" in metrics:
        print(f"  Mismatches: {metrics['num_mismatches']} / {metrics['num_elements']}")
        print(f"  Mismatch Rate: {metrics['mismatch_rate']:.2%}")
    
    print("=" * 80)

