#!/usr/bin/env python3
"""
Stage-5 WS Kernel Candidate (NUM_PRODUCER_WARPS=1)
===================================================
Warp Specialization with 1 producer warp.

This is the actual CUDA kernel implementation from feat/stage5-warp-spec-persistent.
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the actual kernel
from tasks.fp8_sdpa_stage_c_wmma.func_forward import forward_kernel, quantize_sim_fp8_per_head
from tasks.fp8_sdpa_stage_c_wmma.build import build_extension


# Build kernel with WS enabled (cached after first build)
_KERNEL_CACHE = None

def _get_kernel():
    """Lazy-load kernel with WS enabled."""
    global _KERNEL_CACHE
    if _KERNEL_CACHE is None:
        # Set environment for WS with 1 producer
        os.environ["USE_CP_ASYNC"] = "1"
        os.environ["USE_WMMA_PV"] = "1"
        os.environ["USE_WARP_SPECIALIZATION"] = "1"
        os.environ["NUM_PRODUCER_WARPS"] = "1"
        os.environ["USE_PERSISTENT_CTA"] = "0"  # Not implemented yet
        os.environ["USE_FAST_EXP"] = "0"  # Keep correctness
        
        print("Building Stage-5 WS kernel (NUM_PRODUCER_WARPS=1)...")
        _KERNEL_CACHE = build_extension()
    return _KERNEL_CACHE


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Run Stage-5 WS kernel.
    
    Args:
        Q, K, V: [B, H, S, D] tensors (FP16)
        scale: softmax scale (typically 1/sqrt(D))
    
    Returns:
        O: [B, H, S, D] output tensor (FP16)
    """
    # Quantize to FP8 (sim)
    Q_q, Q_s = quantize_sim_fp8_per_head(Q)
    K_q, K_s = quantize_sim_fp8_per_head(K)
    V_q, V_s = quantize_sim_fp8_per_head(V)
    
    # Get kernel
    ext = _get_kernel()
    
    # Run kernel
    O = ext.forward(Q_q, K_q, V_q, Q_s, K_s, V_s, scale)
    
    return O


def get_config():
    """Return configuration for this candidate."""
    return {
        "name": "Stage5-WS-P1",
        "description": "Warp Specialization with 1 producer warp (Stage-5)",
        "toggles": {
            "USE_CP_ASYNC": 1,
            "USE_WMMA_PV": 1,
            "USE_WARP_SPECIALIZATION": 1,
            "NUM_PRODUCER_WARPS": 1,
            "USE_PERSISTENT_CTA": 0,
            "USE_FAST_EXP": 0,
        },
        "arch": "sm_89",
        "precision": "FP8 (E4M3 sim)",
    }


if __name__ == "__main__":
    # Quick test
    import math
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  No CUDA available, cannot test kernel")
        sys.exit(1)
    
    # Mission shape
    B, H, S, D = 2, 8, 512, 64
    scale = 1.0 / math.sqrt(D)
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    K = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    V = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    
    print(f"Testing Stage-5 WS kernel on mission shape: {(B, H, S, D)}")
    
    # Warmup
    for _ in range(5):
        O = run(Q, K, V, scale)
    
    # Time
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(50):
        O = run(Q, K, V, scale)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 50
    
    print(f"✅ Kernel executed successfully")
    print(f"   Latency: {elapsed*1e6:.2f} μs")
    print(f"   Output shape: {O.shape}")
    print(f"   Output dtype: {O.dtype}")
