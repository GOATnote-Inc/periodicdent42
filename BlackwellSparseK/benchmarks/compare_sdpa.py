#!/usr/bin/env python3
"""
Compare BlackwellSparseK against PyTorch SDPA across configurations.
"""

import torch
from blackwell_sparsek import attention_forward
from blackwell_sparsek.utils import compare_to_sdpa, print_comparison_summary


def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    print("Comparing BlackwellSparseK vs PyTorch SDPA")
    print("=" * 80)
    
    configs = [
        ("GPT-2", 1, 12, 512, 64),
        ("GPT-3", 1, 96, 2048, 128),
        ("LLaMA-7B", 1, 32, 2048, 128),
        ("LLaMA-70B", 1, 64, 4096, 128),
    ]
    
    for name, B, H, S, D in configs:
        print(f"\n{name} Configuration: B={B}, H={H}, S={S}, D={D}")
        print("-" * 80)
        
        try:
            Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
            K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
            V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
            
            comparison = compare_to_sdpa(Q, K, V, attention_forward)
            print_comparison_summary(comparison)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return 0


if __name__ == "__main__":
    exit(main())

