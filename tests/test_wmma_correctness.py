#!/usr/bin/env python3
"""
Test WMMA kernel correctness vs PyTorch SDPA
"""

import torch
import torch.nn.functional as F
import subprocess
import sys

def test_correctness():
    print("="*80)
    print("WMMA KERNEL: CORRECTNESS TEST")
    print("="*80)
    print()
    
    # Config (same as C++ test)
    B, H, S, D = 16, 16, 2048, 64
    
    # Create inputs
    torch.manual_seed(42)
    query = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    key = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    value = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Save inputs for C++ kernel
    torch.save({
        'Q': query.cpu(),
        'K': key.cpu(),
        'V': value.cpu(),
    }, '/tmp/attention_inputs.pt')
    
    # Compute reference with PyTorch SDPA
    print("[1/2] Computing reference (PyTorch SDPA)...")
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    
    max_ref = ref.abs().max().item()
    mean_ref = ref.abs().mean().item()
    print(f"  Reference stats:")
    print(f"    Max:  {max_ref:.6f}")
    print(f"    Mean: {mean_ref:.6f}")
    print()
    
    # TODO: Load WMMA output and compare
    # For now, just show that SDPA works
    
    print("[2/2] WMMA kernel test...")
    print("  Run ./build/bin/test_hopper on H100")
    print("  Then compare outputs")
    print()
    
    print("="*80)
    print("Reference computed successfully!")
    print("Next: Compare WMMA output against this reference")
    print("="*80)

if __name__ == "__main__":
    test_correctness()

