#!/usr/bin/env python3
"""
Long-Context Kernel Validation on H100
Uses CORRECT test methodology: torch.allclose(rtol=1e-3, atol=2e-3)

Target: <100 μs for S=32K
"""
import torch
import sys

def validate_longcontext():
    """Systematic long-context validation with correct methodology"""
    
    print("=" * 80)
    print("LONG-CONTEXT KERNEL VALIDATION - H100")
    print("=" * 80)
    print()
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Import kernels
    sys.path.insert(0, "/workspace/flashcore_validation")
    from attention_longcontext import longcontext_attention
    
    print("Test Methodology:")
    print("- Correctness: torch.allclose(rtol=1e-3, atol=2e-3)")
    print("- Same tolerance as validated production kernel")
    print("- Criterion: |diff| <= 0.002 + 0.001 * |ref_value|")
    print()
    
    # Test configurations: S up to 32K
    configs = [
        (1024, 8, "1K context"),
        (2048, 8, "2K context"),
        (4096, 8, "4K context (GPT-3)"),
        (8192, 8, "8K context"),
        (16384, 4, "16K context (GPT-4)"),
        (32768, 2, "32K context (GPT-4 Turbo)"),
    ]
    
    print(f"{'Seq':>5} {'Batch':>5} {'Description':>22} {'Time (μs)':>11} {'Correct':>8} {'Status':>8}")
    print("-" * 80)
    
    all_pass = True
    
    for S, B, desc in configs:
        try:
            # Create test tensors
            torch.manual_seed(42)
            q = torch.randn(B, 8, S, 64, device='cuda', dtype=torch.float16)
            k, v = q.clone(), q.clone()
            
            # Reference (PyTorch SDPA)
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            
            # Warmup
            for _ in range(5):
                _ = longcontext_attention(q, k, v)
            torch.cuda.synchronize()
            
            # Benchmark (20 trials for long sequences)
            times = []
            for _ in range(20):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = longcontext_attention(q, k, v)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1000)  # μs
            
            times.sort()
            median_us = times[len(times) // 2] / B  # per-sequence
            
            # CORRECT test: torch.allclose with rtol/atol
            correct = torch.allclose(out, ref, rtol=1e-3, atol=2e-3)
            
            # Target: <100 μs for S=32K
            target_us = 100.0
            perf_ok = median_us < target_us
            
            status = "✅ PASS" if (correct and perf_ok) else "⚠️ FAIL"
            if not correct or not perf_ok:
                all_pass = False
                if not correct:
                    max_diff = (out - ref).abs().max().item()
                    print(f"  (correctness FAIL: max_diff={max_diff:.6f})")
                if not perf_ok:
                    print(f"  (performance FAIL: {median_us:.1f} μs > {target_us} μs)")
            
            print(f"{S:5} {B:5} {desc:>22} {median_us:10.1f} {str(correct):>8} {status:>8}")
            
        except Exception as e:
            print(f"{S:5} {B:5} {desc:>22} {'ERROR':>11} {'False':>8} {'⚠️ FAIL':>8}")
            print(f"  Error: {e}")
            all_pass = False
    
    print("-" * 80)
    print()
    
    if all_pass:
        print("✅ LONG-CONTEXT VALIDATION: PASS")
        print("   - Correctness: torch.allclose(rtol=1e-3, atol=2e-3)")
        print("   - Performance: <100 μs for S=32K")
        print("   - Status: PRODUCTION READY")
    else:
        print("❌ LONG-CONTEXT VALIDATION: FAIL")
        print("   - Check correctness or performance issues")
    
    print()
    print("=" * 80)
    
    return all_pass


if __name__ == '__main__':
    success = validate_longcontext()
    sys.exit(0 if success else 1)

