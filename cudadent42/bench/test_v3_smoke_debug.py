#!/usr/bin/env python3
"""
V3 Smoke Test (Debug Build)
Minimal test for compute-sanitizer debugging
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path
import sys

def build_v3_debug():
    """Build V3 with DEBUG flags enabled"""
    
    kernel_dir = Path(__file__).parent / "kernels"
    kernel_cu = kernel_dir / "fa_s512_v3.cu"
    bindings_cpp = kernel_dir / "fa_s512_v3_bindings.cpp"
    
    print("🔧 Building V3 with DEBUG flags...")
    print(f"   Kernel: {kernel_cu}")
    print(f"   Bindings: {bindings_cpp}")
    
    # Debug build flags
    debug_flags = [
        "-G",                  # Device debug mode
        "-lineinfo",           # Line info for debuggers
        "-Xptxas", "-O0",      # Disable PTX optimization
        "-DDEBUG_V3=1",        # Enable DEBUG guards
        "-UNDEBUG",            # Disable NDEBUG
        "-std=c++17",
        "-arch=sm_89",         # L4
        "--expt-relaxed-constexpr",
    ]
    
    # Compile
    module = torch.utils.cpp_extension.load(
        name="flash_attention_s512_v3_debug",
        sources=[str(bindings_cpp), str(kernel_cu)],
        extra_cuda_cflags=debug_flags,
        verbose=True,
        with_cuda=True,
    )
    
    print("✅ V3 debug build complete")
    return module


def run_smoke_test(module):
    """Run minimal smoke test"""
    
    print("\n🧪 Running smoke test...")
    
    # Minimal test case
    B, H, S, D = 1, 1, 512, 64
    device = "cuda"
    dtype = torch.float16
    
    # Create tensors
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    print(f"   Input: B={B}, H={H}, S={S}, D={D}")
    
    # Run kernel
    try:
        config_id = 1  # Basic config: BLOCK_M=32, BLOCK_N=32, NUM_WARPS=4
        softmax_scale = 1.0 / (D ** 0.5)
        is_causal = False
        
        # Bindings signature: forward(Q, K, V, softmax_scale, is_causal, config_id)
        O = module.forward(Q, K, V, softmax_scale, is_causal, config_id)
        
        print(f"   Output shape: {O.shape}")
        print(f"   Output range: [{O.min():.4f}, {O.max():.4f}]")
        print(f"   Output mean: {O.mean():.4f}")
        
        # Check for NaN/Inf
        if torch.isnan(O).any():
            print("   ❌ Contains NaN")
            return False
        if torch.isinf(O).any():
            print("   ❌ Contains Inf")
            return False
        
        print("   ✅ No NaN/Inf detected")
        return True
        
    except RuntimeError as e:
        print(f"   ❌ Runtime error: {e}")
        return False


def main():
    print("=" * 80)
    print("V3 DEBUG SMOKE TEST")
    print("=" * 80)
    
    # Build
    try:
        module = build_v3_debug()
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)
    
    # Test
    success = run_smoke_test(module)
    
    print("\n" + "=" * 80)
    if success:
        print("✅ SMOKE TEST PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ SMOKE TEST FAILED")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

