#!/usr/bin/env python3
"""
xFormers integration demo for BlackwellSparseK.
"""

import torch

try:
    from blackwell_sparsek.backends import SparseKAttention
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


def main():
    """Run xFormers integration demo."""
    if not BACKEND_AVAILABLE:
        print("❌ BlackwellSparseK xFormers backend not available")
        return 1
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    print("=" * 80)
    print("BlackwellSparseK xFormers Integration Demo")
    print("=" * 80)
    
    # Create attention module
    attention = SparseKAttention()
    
    # Configuration (xFormers uses [B, S, H, D] layout)
    B, S, H, D = 1, 512, 8, 64
    
    print(f"Config: B={B}, S={S}, H={H}, D={D}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Create inputs
    q = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
    
    print(f"\nInput layout: [Batch, Sequence, Heads, Dimension]")
    
    # Forward pass
    print(f"\nComputing attention with xFormers backend...")
    output = attention(q, k, v)
    
    print(f"\nOutput shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    
    print("\n✅ xFormers integration working!")
    return 0


if __name__ == "__main__":
    exit(main())

