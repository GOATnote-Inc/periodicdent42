#!/usr/bin/env python3
"""
Stage-2 Baseline Kernel (Control)
==================================
Stage-2: cp.async + WMMA P·V (NO warp specialization)

This is the control baseline to compare against Stage-5 WS variants.
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


# Build kernel with Stage-2 config (cached after first build)
_KERNEL_CACHE = None

def _get_kernel():
    """Lazy-load kernel with Stage-2 config."""
    global _KERNEL_CACHE
    if _KERNEL_CACHE is None:
        # Set environment for Stage-2 (no WS)
        os.environ["USE_CP_ASYNC"] = "1"
        os.environ["USE_WMMA_PV"] = "1"
        os.environ["USE_WARP_SPECIALIZATION"] = "0"  # Stage-2 baseline
        os.environ["NUM_PRODUCER_WARPS"] = "1"  # Ignored when WS=0
        os.environ["USE_PERSISTENT_CTA"] = "0"
        os.environ["USE_FAST_EXP"] = "0"
        
        print("Building Stage-2 baseline (cp.async + WMMA P·V, no WS)...")
        _KERNEL_CACHE = build_extension()
    return _KERNEL_CACHE


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Run Stage-2 baseline kernel.
    
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
        "name": "Stage2-Baseline",
        "description": "Stage-2: cp.async + WMMA P·V (control, no WS)",
        "toggles": {
            "USE_CP_ASYNC": 1,
            "USE_WMMA_PV": 1,
            "USE_WARP_SPECIALIZATION": 0,
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
    
    print(f"Testing Stage-2 baseline on mission shape: {(B, H, S, D)}")
    
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
