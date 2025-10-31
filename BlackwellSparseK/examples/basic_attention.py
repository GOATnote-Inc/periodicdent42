#!/usr/bin/env python3
"""
Basic usage example for BlackwellSparseK attention kernel.
"""

import torch
from blackwell_sparsek import attention_forward

def main():
    """Run basic attention example."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    print("=" * 80)
    print("BlackwellSparseK Basic Attention Example")
    print("=" * 80)
    
    # Configuration
    B, H, S, D = 1, 8, 512, 64  # Batch, Heads, Sequence, Dimension
    
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Create random input tensors
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    print(f"\nInput shapes:")
    print(f"  Q: {tuple(Q.shape)}")
    print(f"  K: {tuple(K.shape)}")
    print(f"  V: {tuple(V.shape)}")
    
    # Compute attention
    print(f"\nComputing attention...")
    output = attention_forward(Q, K, V)
    
    print(f"\nOutput shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    
    # Basic statistics
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std:  {output.std().item():.6f}")
    print(f"  Min:  {output.min().item():.6f}")
    print(f"  Max:  {output.max().item():.6f}")
    
    print("\n✅ Success!")
    return 0


if __name__ == "__main__":
    exit(main())

