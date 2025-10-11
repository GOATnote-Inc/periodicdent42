#!/usr/bin/env python3
"""
Basic end-to-end tests for CUDAdent42 FlashAttention kernels.
Tests both FP16 and BF16 (if available) on current GPU.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def test_import():
    """Test that the module imports successfully."""
    print("━" * 70)
    print("TEST: Module Import")
    print("━" * 70)
    
    try:
        import flashmoe_science._C as m
        print(f"✅ Module imported: {m}")
        print(f"✅ Has forward: {hasattr(m, 'forward')}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_fp16():
    """Test FP16 forward pass."""
    print("\n━" * 70)
    print("TEST: FP16 Forward Pass")
    print("━" * 70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping")
        return True
    
    try:
        import flashmoe_science._C as m
        
        # Create test tensors
        B, H, S, D = 2, 1, 8, 64
        Q = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        K = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        V = torch.randn(B * H * S, D, dtype=torch.float16, device='cuda')
        
        print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
        print(f"Input dtype: {Q.dtype}")
        
        # Run forward
        O = m.forward(Q, K, V)
        
        print(f"✅ FP16 forward succeeded!")
        print(f"Output shape: {O.shape}, dtype: {O.dtype}")
        print(f"Output range: [{O.min():.4f}, {O.max():.4f}]")
        print(f"Output mean: {O.mean():.4f}, std: {O.std():.4f}")
        
        # Basic sanity checks
        assert O.shape == Q.shape, f"Shape mismatch: {O.shape} != {Q.shape}"
        assert O.dtype == Q.dtype, f"Dtype mismatch: {O.dtype} != {Q.dtype}"
        assert torch.isfinite(O).all(), "Output contains NaN or Inf"
        
        return True
        
    except Exception as e:
        print(f"❌ FP16 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bf16():
    """Test BF16 forward pass (if supported)."""
    print("\n━" * 70)
    print("TEST: BF16 Forward Pass")
    print("━" * 70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping")
        return True
    
    # Check if BF16 is supported (SM80+)
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        print(f"⚠️  BF16 not supported on SM_{major}{minor} (requires SM80+)")
        return True
    
    try:
        import flashmoe_science._C as m
        
        # Create test tensors
        B, H, S, D = 2, 1, 8, 64
        Q = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(B * H * S, D, dtype=torch.bfloat16, device='cuda')
        
        print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
        print(f"Input dtype: {Q.dtype}")
        
        # Run forward
        O = m.forward(Q, K, V)
        
        print(f"✅ BF16 forward succeeded!")
        print(f"Output shape: {O.shape}, dtype: {O.dtype}")
        print(f"Output range: [{O.min():.4f}, {O.max():.4f}]")
        print(f"Output mean: {O.mean():.4f}, std: {O.std():.4f}")
        
        # Basic sanity checks
        assert O.shape == Q.shape, f"Shape mismatch: {O.shape} != {Q.shape}"
        assert O.dtype == Q.dtype, f"Dtype mismatch: {O.dtype} != {Q.dtype}"
        assert torch.isfinite(O).all(), "Output contains NaN or Inf"
        
        return True
        
    except Exception as e:
        print(f"❌ BF16 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  CUDAdent42: Basic End-to-End Tests".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Print system info
    print("System Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        major, minor = torch.cuda.get_device_capability()
        print(f"  Compute capability: SM_{major}{minor}")
    print()
    
    # Run tests
    results = []
    results.append(("Import", test_import()))
    results.append(("FP16", test_fp16()))
    results.append(("BF16", test_bf16()))
    
    # Summary
    print("\n" + "═" * 70)
    print("TEST SUMMARY")
    print("═" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("═" * 70)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

