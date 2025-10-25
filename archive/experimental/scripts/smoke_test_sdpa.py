#!/usr/bin/env python3
"""
SDPA Smoke Test - Warmup and basic functionality check
Phase 0 Pre-flight validation
"""
import torch
import torch.nn.functional as F
import sys

def smoke_test():
    """Quick SDPA warmup to ensure GPU is ready"""
    print("=" * 60)
    print("Phase 0: SDPA Warmup Smoke Test")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)
    
    # Print GPU info
    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"✓ GPU: {device_name}")
    print(f"✓ Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    
    # Verify it's an L4
    if "L4" not in device_name:
        print(f"⚠️  Warning: Expected L4, got {device_name}")
    
    # Quick warmup
    B, H, S, D = 1, 1, 512, 64
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    try:
        # Run SDPA
        out = F.scaled_dot_product_attention(Q, K, V)
        
        # Verify output shape
        assert out.shape == (B, H, S, D), f"Wrong output shape: {out.shape}"
        
        # Verify no NaN/Inf
        assert torch.isfinite(out).all(), "Output contains NaN/Inf"
        
        print(f"✓ SDPA warmup: shape={out.shape}, dtype={out.dtype}")
        print(f"✓ Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        print()
        print("✅ SDPA warmup OK - GPU ready for benchmarking")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"❌ SDPA failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(smoke_test())

