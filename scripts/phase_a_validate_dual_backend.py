#!/usr/bin/env python3
"""
Phase A.3: Dual-reference validation (Flash vs Math SDPA)

Tests Phase 4 kernel against both SDPA backends to determine which is correct reference.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.sdpa_oracle import sdpa_ref

def main():
    print("=" * 70)
    print("Phase A.3: Dual-Reference SDPA Validation")
    print("=" * 70)
    print()
    
    # Test shape: B=1, H=8, S=512, D=64
    B, H, S, D = 1, 8, 512, 64
    
    print(f"Shape: B={B}, H={H}, S={S}, D={D}")
    print(f"Dtype: FP16")
    print()
    
    # Generate inputs with fixed seed
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)
    
    print("📊 Testing SDPA backends...")
    print()
    
    # Test Flash backend
    with torch.no_grad():
        ref_flash = sdpa_ref(q, k, v, scale, {
            "enable_flash": True,
            "enable_math": False,
            "enable_mem_efficient": False
        })
    
    # Test Math backend
    with torch.no_grad():
        ref_math = sdpa_ref(q, k, v, scale, {
            "enable_flash": False,
            "enable_math": True,
            "enable_mem_efficient": False
        })
    
    # Compare backends
    diff_backends = (ref_flash - ref_math).abs().max().item()
    print(f"Flash vs Math backend diff: {diff_backends:.6f}")
    
    if diff_backends < 1e-3:
        print("✅ Backends agree (< 1e-3)")
    else:
        print(f"⚠️  Backends differ (> 1e-3) - this is normal for different implementations")
    print()
    
    # Try to load Phase 4 kernel
    print("📦 Loading Phase 4 kernel...")
    try:
        from cudadent42.bench.kernels.build_phase3 import build_phase3_variant
        import os
        
        # Set Phase 4 config
        os.environ['BLOCK_M'] = '32'
        os.environ['NUM_WARPS'] = '8'
        os.environ['VEC_WIDTH'] = '4'
        os.environ['SYNC_POLICY'] = '2'
        os.environ['REDUCE'] = 'warp'
        
        build_phase3_variant()
        import fa_phase3
        
        print("✅ Phase 4 kernel loaded")
        print()
        
        # Test Phase 4 against both backends
        with torch.no_grad():
            phase4_out = fa_phase3.forward(q, k, v, scale)
        
        diff_flash = (phase4_out.float() - ref_flash.float()).abs().max().item()
        diff_math = (phase4_out.float() - ref_math.float()).abs().max().item()
        
        print("📈 Phase 4 Correctness:")
        print(f"   vs Flash: {diff_flash:.6f} {'✅' if diff_flash < 2e-3 else '❌'} (tol=2e-3)")
        print(f"   vs Math:  {diff_math:.6f} {'✅' if diff_math < 2e-3 else '❌'} (tol=2e-3)")
        print()
        
        # Determine best reference
        if diff_flash < diff_math:
            best_backend = "flash"
            best_diff = diff_flash
        else:
            best_backend = "math"
            best_diff = diff_math
        
        print(f"🎯 Best Reference: {best_backend.upper()} backend (diff={best_diff:.6f})")
        print()
        
        # Save result
        result_file = Path(__file__).parent.parent / "evidence" / "phase_a_dual_backend.txt"
        result_file.parent.mkdir(exist_ok=True)
        result_file.write_text(f"""\
Phase A.3: Dual-Reference Validation Results

SDPA Backends:
  Flash vs Math diff: {diff_backends:.6f}

Phase 4 Correctness:
  vs Flash: {diff_flash:.6f} {'✅' if diff_flash < 2e-3 else '❌'}
  vs Math:  {diff_math:.6f} {'✅' if diff_math < 2e-3 else '❌'}

Recommended Backend: {best_backend}
Best Diff: {best_diff:.6f}

PyTorch Version: {torch.__version__}
CUDA Version: {torch.version.cuda}
GPU: {torch.cuda.get_device_name(0)}
""")
        
        print(f"✅ Results saved to {result_file}")
        print()
        
        # Exit codes
        if best_diff < 2e-3:
            print("=" * 70)
            print("✅ PHASE A.3 PASSED: Phase 4 is correct with best reference")
            print("=" * 70)
            sys.exit(0)
        else:
            print("=" * 70)
            print("❌ PHASE A.3 FAILED: Phase 4 has correctness issues")
            print("=" * 70)
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Failed to load Phase 4 kernel: {e}")
        print()
        print("This is expected if Phase 4 hasn't been built yet.")
        print("Use this script after Phase A.1 and A.2 are complete.")
        sys.exit(2)

if __name__ == "__main__":
    main()

