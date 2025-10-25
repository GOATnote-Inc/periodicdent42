"""
Phase 4 FlashAttention kernel for KernelBench evaluation

Kernel specs:
- Warp-cooperative reductions (NUM_WARPS=8)
- Light barriers (SYNC_POLICY=2, 4 barriers/tile)
- Vectorized loads (VEC_WIDTH=4)
- Block tiling (BLOCK_M=32)
- Target: 839 μs (measured baseline)

Repository: https://github.com/GOATnote-Inc/periodicdent42
Commit: 59f428d
"""

import torch
import torch.nn as nn
import sys
import os

# Add periodicdent42 to path
REPO_ROOT = os.path.expanduser("~/periodicdent42")
sys.path.insert(0, REPO_ROOT)

# Import Phase 4 kernel builder
from bench.build_phase3_variant import build_phase3_variant

# Build Phase 4 configuration (best from Evo sweep)
fa_phase4 = build_phase3_variant(
    BLOCK_M=32,
    NUM_WARPS=8,
    VEC_WIDTH=4,
    SYNC_POLICY=2,
    REDUCE="warp"
)

class Model(nn.Module):
    """Wrapper for Phase 4 kernel to match KernelBench API"""
    
    def __init__(self):
        super().__init__()
        self.scale = 1.0 / (64 ** 0.5)  # HEAD_DIM=64
    
    def forward(self, Q, K, V):
        """
        Forward pass using custom CUDA kernel
        
        Args:
            Q, K, V: [B, H, S, D] tensors in FP16
        
        Returns:
            O: [B, H, S, D] output tensor
        """
        return fa_phase4.forward(Q, K, V, self.scale)

def get_init_inputs():
    """No initialization parameters needed"""
    return []

