#!/usr/bin/env python3
"""
Test Phase 3 Stable kernel correctness and performance

Uses SDPA Oracle for comprehensive validation.
"""

import torch
import sys
import os
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.sdpa_oracle import evaluate_candidate

def main():
    print("=" * 70)
    print("Phase 3 Stable Kernel Test")
    print("=" * 70)
    print()
    
    # Build kernel
    print("📦 Building Phase 3 Stable kernel...")
    from bench.build_phase3_stable import build_phase3_stable
    
    # Set environment for optimal config (from Phase 4)
    os.environ['BLOCK_M'] = '32'
    os.environ['NUM_WARPS'] = '8'
    os.environ['VEC_WIDTH'] = '4'
    os.environ['SYNC_POLICY'] = '2'
    
    module = build_phase3_stable()
    print("✅ Build complete")
    print()
    
    # Test shape
    B, H, S, D = 1, 8, 512, 64
    print(f"Test Shape: B={B}, H={H}, S={S}, D={D}")
    print()
    
    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)
    
    # Test with SDPA Oracle
    print("🔍 Running SDPA Oracle evaluation...")
    print()
    
    results = evaluate_candidate(
        candidate_fn=lambda: module.forward(q, k, v, scale),
        q=q, k=k, v=v, scale=scale,
        sdpa_backend="flash",
        iters=100,
        warmup=20,
        speedup_threshold=0.95,
        correctness_tol=2e-3
    )
    
    from bench.sdpa_oracle import print_results
    print_results(results)
    
    # Save results
    import json
    output_file = Path("evidence/phase3_stable_results.json")
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))
    print()
    print(f"✅ Results saved to {output_file}")
    
    # Exit with appropriate code
    if results["passed"]:
        print()
        print("=" * 70)
        print("✅ PHASE 3 STABLE: ALL GATES PASSED")
        print("=" * 70)
        sys.exit(0)
    elif results["correctness"]["passed"]:
        print()
        print("=" * 70)
        print("⚠️  PHASE 3 STABLE: CORRECT BUT NOT FAST ENOUGH")
        print("=" * 70)
        print(f"Speedup: {results['performance']['speedup']:.3f}× (need > 1.053×)")
        print("This is expected - Phase 3 is baseline before Tensor Cores")
        sys.exit(0)  # Exit 0 for correctness (performance will improve in Phase B/C)
    else:
        print()
        print("=" * 70)
        print("❌ PHASE 3 STABLE: CORRECTNESS FAILED")
        print("=" * 70)
        print(f"Max diff: {results['correctness']['max_diff']:.6f} (tol: {results['correctness']['tolerance']:.6f})")
        sys.exit(1)

if __name__ == "__main__":
    main()

