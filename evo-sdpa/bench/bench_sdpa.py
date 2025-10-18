#!/usr/bin/env python3
"""
EvoEngineer SDPA Benchmarking Harness
Compile → Correctness → Performance measurement
"""

import os, math, torch, time, sys
from torch.utils.cpp_extension import load

def build_ext():
    """Build CUDA extension from kernels/sdpa_fused.cu"""
    kernel_dir = os.path.join(os.path.dirname(__file__), "..", "kernels")
    srcs = [
        os.path.join(kernel_dir, "sdpa_fused.cu"),
        os.path.join(kernel_dir, "sdpa_fused_v2.cu"),
        os.path.join(kernel_dir, "sdpa_fused_v2b.cu"),
        os.path.join(kernel_dir, "sdpa_fused_v2c.cu"),
        os.path.join(kernel_dir, "sdpa_fused_v2c_v4.cu"),
        os.path.join(kernel_dir, "sdpa_fused_v2c_v5.cu"),
        os.path.join(kernel_dir, "sdpa_fused_bindings.cpp")
    ]
    
    return load(
        name="sdpa_fused_ext",
        sources=srcs,
        extra_cuda_cflags=[
            "-O3",
            "--generate-code=arch=compute_89,code=sm_89",
            "--use_fast_math",
            "-lineinfo",
            "-Xptxas", "-v",
            "-std=c++17"
        ],
        verbose=True
    )

def ref_sdpa(q, k, v, causal, dropout_p=0.0):
    """PyTorch reference SDPA"""
    attn_mask = None
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=causal
    )

def rand_inputs(B, H, L, d, dtype=torch.float16, device="cuda"):
    """Generate random test inputs"""
    q = torch.randn(B, H, L, d, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H, L, d, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, L, d, device=device, dtype=dtype) * 0.1
    return q, k, v

def run_case(mod, B=2, H=8, L=2048, d=64, causal=False, dtype=torch.float16, iters=100, verbose=True):
    """Run single benchmark case"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing: B={B}, H={H}, L={L}, d={d}, causal={causal}, dtype={dtype}")
        print('='*80)
    
    q, k, v = rand_inputs(B, H, L, d, dtype)
    scale = 1.0 / math.sqrt(d)
    
    # Reference
    with torch.no_grad():
        o_ref = ref_sdpa(q, k, v, causal).contiguous()
    
    # Custom kernel
    O = torch.empty_like(o_ref)
    try:
        mod.sdpa_fused_forward(q, k, v, O, causal, scale)
    except Exception as e:
        print(f"❌ KERNEL LAUNCH FAILED: {e}")
        return {"ok": False, "us": float('inf'), "us_ref": 0, "speedup_vs_torch": 0}
    
    # Correctness
    tol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    max_rel = ((O - o_ref).abs() / (o_ref.abs() + 1e-8)).max().item()
    max_abs = (O - o_ref).abs().max().item()
    ok = (max_abs <= 5*tol) or (max_rel <= 5*tol)
    
    if verbose:
        print(f"Correctness: {'✅ PASS' if ok else '❌ FAIL'}")
        print(f"  max_abs_diff: {max_abs:.6f}")
        print(f"  max_rel_diff: {max_rel:.6f}")
    
    if not ok:
        return {"ok": False, "us": float('inf'), "us_ref": 0, "speedup_vs_torch": 0}
    
    # Warmup
    for _ in range(10):
        mod.sdpa_fused_forward(q, k, v, O, causal, scale)
    torch.cuda.synchronize()
    
    # Timing - Custom kernel
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mod.sdpa_fused_forward(q, k, v, O, causal, scale)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iters
    
    # Timing - PyTorch reference
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            ref_sdpa(q, k, v, causal)
    torch.cuda.synchronize()
    us_ref = (time.perf_counter() - t1) * 1e6 / iters
    
    speedup = us_ref / us if us > 0 else 0
    
    if verbose:
        print(f"\nPerformance:")
        print(f"  Custom kernel:  {us:7.2f} μs")
        print(f"  PyTorch SDPA:   {us_ref:7.2f} μs")
        print(f"  Speedup:        {speedup:.2f}×")
    else:
        # Compact output for automated runs
        status = "✅" if ok else "❌"
        print(f"{status} B={B} H={H} L={L:4d} d={d:3d} causal={int(causal)} | "
              f"custom={us:7.2f}μs torch={us_ref:7.2f}μs speedup={speedup:.2f}× "
              f"max_diff={max_abs:.6f}")
    
    return {
        "ok": ok,
        "us": us,
        "us_ref": us_ref,
        "speedup_vs_torch": speedup,
        "max_abs": max_abs,
        "max_rel": max_rel
    }

def main():
    print("\n" + "="*80)
    print("🧬 EvoEngineer SDPA Benchmark")
    print("="*80)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    print("\n📦 Building CUDA extension...")
    try:
        mod = build_ext()
        print("✅ Build successful")
    except Exception as e:
        print(f"❌ Build failed: {e}")
        return
    
    # Test cases
    test_cases = [
        # Mission-critical shape
        {"B": 1, "H": 8, "L": 512, "d": 64, "causal": False},
        # EvoEngineer standard shapes
        {"B": 2, "H": 8, "L": 512, "d": 64, "causal": True},
        {"B": 2, "H": 8, "L": 2048, "d": 64, "causal": True},
        {"B": 2, "H": 8, "L": 2048, "d": 128, "causal": True},
        {"B": 1, "H": 8, "L": 8192, "d": 64, "causal": False},
    ]
    
    results = []
    for case in test_cases:
        res = run_case(mod, iters=100, **case)
        results.append({**case, **res})
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    
    print(f"\nCorrectness: {passed}/{total} passed")
    
    if passed > 0:
        valid_results = [r for r in results if r["ok"]]
        avg_speedup = sum(r["speedup_vs_torch"] for r in valid_results) / len(valid_results)
        best_speedup = max(r["speedup_vs_torch"] for r in valid_results)
        
        print(f"Average speedup: {avg_speedup:.2f}×")
        print(f"Best speedup:    {best_speedup:.2f}×")
        
        # Mission-critical case
        mission = results[0]
        if mission["ok"]:
            print(f"\n🎯 MISSION-CRITICAL (B=1, H=8, L=512, d=64):")
            print(f"   Latency: {mission['us']:.2f} μs")
            print(f"   Target:  < 5.00 μs (5.2× faster than SDPA @ 25.94 μs)")
            if mission['us'] < 5.0:
                print(f"   ✅ TARGET ACHIEVED!")
            else:
                print(f"   ⚠️  Need {mission['us'] / 5.0:.2f}× more speedup")
    
    print()

if __name__ == "__main__":
    main()

