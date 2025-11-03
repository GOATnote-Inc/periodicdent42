#!/usr/bin/env python3
"""
Test I4 Correctness (TDD)
=========================

Tests fused softmax+PV kernel against PyTorch reference.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_i4_correctness():
    """Test I4 kernel matches PyTorch SDPA"""
    
    print("="*80)
    print("TEST: I4 Correctness")
    print("="*80)
    print()
    
    # Config
    B, H, S, D = 4, 16, 1024, 64
    S_max = 1024  # Padded size
    
    # Generate inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Compute reference (PyTorch SDPA) - CAUSAL to match I4 kernel
    print("Computing reference (PyTorch SDPA with causal masking)...")
    with torch.no_grad():
        out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    
    # Compute scores Q@K^T for I4 input
    print("Computing Q@K^T scores...")
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Reshape for I4 kernel: [B, H, S, S] -> [B*H, S, S]
    scores_flat = scores.reshape(B*H, S, S)
    V_flat = V.reshape(B*H, S, D)
    
    try:
        # Import compiled kernel
        import dhp_i4_kernel
        
        print("Running I4 kernel...")
        with torch.no_grad():
            out_i4_flat = dhp_i4_kernel.forward(scores_flat, V_flat, S_max, S)
        
        # Reshape back: [B*H, S, d] -> [B, H, S, d]
        out_i4 = out_i4_flat.reshape(B, H, S, D)
        
        # Compare
        max_diff = torch.abs(out_i4 - out_ref).max().item()
        mean_diff = torch.abs(out_i4 - out_ref).mean().item()
        
        print()
        print("Results:")
        print(f"  Max absolute difference:  {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        print()
        
        # Check tolerance (FP16 + online softmax)
        TOLERANCE = 2e-3  # Relaxed for FP16
        
        if max_diff < TOLERANCE:
            print("✅ PASS: I4 matches PyTorch SDPA")
            return True
        else:
            print(f"❌ FAIL: Max diff {max_diff:.6f} exceeds tolerance {TOLERANCE}")
            return False
            
    except ImportError:
        print("⚠️  SKIP: I4 kernel not compiled yet")
        print("   Run: python setup.py install")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    result = test_i4_correctness()
    sys.exit(0 if result else 1)

