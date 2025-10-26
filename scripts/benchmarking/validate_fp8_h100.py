#!/usr/bin/env python3
"""
FP8 Kernel Validation on H100
Uses CORRECT test methodology: torch.allclose(rtol, atol)

Learned from multi-head bug:
- WRONG: max_diff < threshold (absolute)
- RIGHT: torch.allclose(rtol, atol) (relative + absolute)
"""
import torch
import sys

def validate_fp8():
    """Systematic FP8 validation with correct methodology"""
    
    print("=" * 80)
    print("FP8 KERNEL VALIDATION - H100")
    print("=" * 80)
    print()
    
    # Check Hopper
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]})")
    
    if cap[0] < 9:
        print(f"❌ FAIL: FP8 requires Hopper (sm_90+)")
        return False
    
    print(f"✅ Hopper detected")
    print()
    
    # Import kernels
    sys.path.insert(0, "/workspace/flashcore_validation")
    from attention_fp8 import attention_fp8
    from attention_production import attention as attention_fp16
    
    print("Test Methodology:")
    print("- Correctness: torch.allclose(rtol=5e-3, atol=5e-3)")
    print("- FP8 tolerance relaxed vs FP16 due to lower precision")
    print("- Criterion: |diff| <= 0.005 + 0.005 * |ref_value|")
    print()
    
    configs = [
        (128, 16, "Small batch"),
        (256, 16, "Medium batch"),
        (512, 16, "Large batch"),
        (512, 32, "Large batch + high throughput"),
    ]
    
    print(f"{'Seq':>4} {'Batch':>5} {'Description':>25} {'FP8 (μs)':>10} {'Correct':>8} {'Status':>8}")
    print("-" * 80)
    
    all_pass = True
    
    for S, B, desc in configs:
        # Create FP16 reference tensors
        torch.manual_seed(42)
        q_fp16 = torch.randn(B, 8, S, 64, device='cuda', dtype=torch.float16)
        k_fp16, v_fp16 = q_fp16.clone(), q_fp16.clone()
        
        # Reference (FP16)
        ref = attention_fp16(q_fp16, k_fp16, v_fp16)
        
        # Convert to FP8
        q_fp8 = q_fp16.to(torch.float8_e4m3fn)
        k_fp8 = k_fp16.to(torch.float8_e4m3fn)
        v_fp8 = v_fp16.to(torch.float8_e4m3fn)
        
        # Warmup
        for _ in range(10):
            _ = attention_fp8(q_fp8, k_fp8, v_fp8)
        torch.cuda.synchronize()
        
        # Benchmark (100 trials)
        times = []
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out_fp8 = attention_fp8(q_fp8, k_fp8, v_fp8)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # μs
        
        times.sort()
        median_us = times[len(times) // 2] / B  # per-sequence
        
        # CORRECT test: torch.allclose with rtol/atol
        # FP8 tolerance: more relaxed than FP16
        correct = torch.allclose(out_fp8, ref, rtol=5e-3, atol=5e-3)
        
        status = "✅ PASS" if (correct and median_us < 1.0) else "⚠️ FAIL"
        if not correct or median_us >= 1.0:
            all_pass = False
        
        print(f"{S:4} {B:5} {desc:>25} {median_us:9.2f} {str(correct):>8} {status:>8}")
    
    print("-" * 80)
    print()
    
    if all_pass:
        print("✅ FP8 VALIDATION: PASS")
        print("   - Correctness: torch.allclose(rtol=5e-3, atol=5e-3)")
        print("   - Performance: <1 μs per sequence")
        print("   - Status: PRODUCTION READY")
    else:
        print("❌ FP8 VALIDATION: FAIL")
        print("   - Check correctness or performance issues")
    
    print()
    print("=" * 80)
    
    return all_pass


if __name__ == '__main__':
    success = validate_fp8()
    sys.exit(0 if success else 1)

