#!/usr/bin/env python3
"""
Debug V3 kernel by testing simpler configurations
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from build_v3_release import build_v3_release


def test_tiny_shape():
    """Test with minimal shape to isolate bug"""
    
    B, H, S, D = 1, 1, 512, 64
    device = "cuda"
    dtype = torch.float16
    
    print("\n" + "=" * 80)
    print("Debug: Tiny Shape (B=1, H=1, S=512, D=64, non-causal)")
    print("=" * 80)
    
    # Simple input: all ones
    torch.manual_seed(0)
    Q = torch.ones(B, H, S, D, device=device, dtype=dtype) * 0.1
    K = torch.ones(B, H, S, D, device=device, dtype=dtype) * 0.1
    V = torch.ones(B, H, S, D, device=device, dtype=dtype)
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    # Reference
    print("\nPyTorch SDPA:")
    O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=softmax_scale)
    print(f"  Output shape: {O_ref.shape}")
    print(f"  Output range: [{O_ref.min():.6f}, {O_ref.max():.6f}]")
    print(f"  Output mean: {O_ref.mean():.6f}")
    print(f"  First 8 values: {O_ref[0,0,0,:8].tolist()}")
    
    # V3
    print("\nV3 Kernel (config_id=1):")
    module = build_v3_release()
    O_v3 = module.forward(Q, K, V, softmax_scale, False, 1)
    print(f"  Output shape: {O_v3.shape}")
    print(f"  Output range: [{O_v3.min():.6f}, {O_v3.max():.6f}]")
    print(f"  Output mean: {O_v3.mean():.6f}")
    print(f"  First 8 values: {O_v3[0,0,0,:8].tolist()}")
    
    # Compare
    abs_diff = (O_v3 - O_ref).abs()
    print("\nDifference:")
    print(f"  Max abs diff: {abs_diff.max():.6f}")
    print(f"  Mean abs diff: {abs_diff.mean():.6f}")
    print(f"  First 8 diffs: {abs_diff[0,0,0,:8].tolist()}")
    
    # Manual calculation for validation
    print("\n" + "-" * 80)
    print("Manual Calculation (first query vector):")
    print("-" * 80)
    
    q0 = Q[0,0,0,:].float()  # First query
    scores = torch.matmul(q0.unsqueeze(0), K[0,0].float().T).squeeze() * softmax_scale
    print(f"Scores (Q·K^T * scale): mean={scores.mean():.6f}, range=[{scores.min():.6f}, {scores.max():.6f}]")
    
    attn = F.softmax(scores, dim=-1)
    print(f"Attention weights: sum={attn.sum():.6f}, mean={attn.mean():.6f}")
    
    out_manual = torch.matmul(attn.unsqueeze(0), V[0,0].float()).squeeze().half()
    print(f"Manual output: {out_manual[:8].tolist()}")
    print(f"SDPA output:   {O_ref[0,0,0,:8].tolist()}")
    print(f"V3 output:     {O_v3[0,0,0,:8].tolist()}")
    
    return abs_diff.max().item() < 0.01


def test_config_2():
    """Test config 2 (smaller blocks)"""
    
    B, H, S, D = 1, 1, 512, 64
    device = "cuda"
    dtype = torch.float16
    
    print("\n" + "=" * 80)
    print("Debug: Config 2 (BLOCK_M=32, BLOCK_N=32, non-causal)")
    print("=" * 80)
    
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype) * 0.5
    K = torch.randn(B, H, S, D, device=device, dtype=dtype) * 0.5
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    # Reference
    O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=softmax_scale)
    
    # V3 config 2
    module = build_v3_release()
    O_v3 = module.forward(Q, K, V, softmax_scale, False, 2)
    
    abs_diff = (O_v3 - O_ref).abs()
    print(f"\nMax abs diff: {abs_diff.max():.6f}")
    print(f"Mean abs diff: {abs_diff.mean():.6f}")
    
    return abs_diff.max().item() < 0.01


if __name__ == "__main__":
    print("\n" * 2)
    print("="  * 80)
    print("V3 KERNEL DEBUG SESSION")
    print("=" * 80)
    
    # Test 1: Tiny shape with simple input
    passed1 = test_tiny_shape()
    
    # Test 2: Config 2 (smaller blocks)
    passed2 = test_config_2()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tiny shape test: {'✅ PASSED' if passed1 else '❌ FAILED'}")
    print(f"Config 2 test: {'✅ PASSED' if passed2 else '❌ FAILED'}")
    
    sys.exit(0 if (passed1 and passed2) else 1)

