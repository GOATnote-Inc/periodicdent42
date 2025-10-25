#!/usr/bin/env python3
"""Quick local correctness test for multi-head attention"""
import torch
from attention_multihead import multihead_attention

def test_correctness():
    """Test numerical correctness vs PyTorch SDPA"""
    print("Testing multi-head attention correctness...")
    print()
    
    # Test configurations
    configs = [
        (8, 512, 16, 64),    # Baseline
        (32, 512, 16, 64),   # GPT-3 Small
        (96, 512, 16, 64),   # GPT-4
        (128, 512, 8, 64),   # GPT-4 Max
    ]
    
    for H, S, B, D in configs:
        # Create test tensors
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        
        # Custom kernel
        out_custom = multihead_attention(q, k, v)
        
        # PyTorch reference
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )
        
        # Check correctness
        max_diff = (out_custom - out_ref).abs().max().item()
        mean_diff = (out_custom - out_ref).abs().mean().item()
        
        status = "✅" if max_diff < 2e-3 else "❌"
        print(f"H={H:3}, S={S:4}, B={B:2}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} {status}")
    
    print()
    print("Correctness validation complete!")

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        exit(0)
    
    test_correctness()

