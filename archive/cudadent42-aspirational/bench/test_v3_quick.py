"""
Quick V3 Validation & Benchmark
Test all 3 pre-compiled configs for correctness and performance
"""

import torch
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fa_s512_v3 import flash_attention_s512_v3_forward, CONFIGS

def test_correctness(B, H, S, D, config_id, is_causal=False):
    """Test correctness vs PyTorch SDPA"""
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    K = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    V = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    
    # V3
    O_v3 = flash_attention_s512_v3_forward(Q, K, V, config_id=config_id, is_causal=is_causal)
    
    # SDPA
    Q_sdpa = Q.permute(0, 2, 1, 3)
    K_sdpa = K.permute(0, 2, 1, 3)
    V_sdpa = V.permute(0, 2, 1, 3)
    O_sdpa = torch.nn.functional.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa, is_causal=is_causal)
    O_sdpa = O_sdpa.permute(0, 2, 1, 3)
    
    # Compare
    max_diff = (O_v3 - O_sdpa).abs().max().item()
    mean_diff = (O_v3 - O_sdpa).abs().mean().item()
    
    passed = max_diff < 0.01  # FP16 tolerance
    return passed, max_diff, mean_diff


def benchmark_config(B, H, S, D, config_id, num_warmup=20, num_iters=100):
    """Benchmark a single config"""
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    K = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    V = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        _ = flash_attention_s512_v3_forward(Q, K, V, config_id=config_id)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        O = flash_attention_s512_v3_forward(Q, K, V, config_id=config_id)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters
    
    return elapsed * 1000  # ms


def main():
    print("="*70)
    print(" V3 QUICK VALIDATION & BENCHMARK")
    print("="*70)
    
    B, H, S, D = 32, 8, 512, 64  # Target shape
    
    print(f"\nShape: B={B}, H={H}, S={S}, D={D}")
    print("\n" + "="*70)
    print("CORRECTNESS TESTS")
    print("="*70)
    
    # Test all configs
    correctness_results = {}
    for config_id in [1, 2, 3]:
        cfg = CONFIGS[config_id]
        print(f"\nConfig {config_id}: BLOCK_M={cfg['BLOCK_M']}, BLOCK_N={cfg['BLOCK_N']}, WARPS={cfg['WARPS']}")
        
        # Non-causal
        passed, max_diff, mean_diff = test_correctness(B, H, S, D, config_id, is_causal=False)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Non-causal: {status} (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
        
        # Causal
        passed_causal, max_diff_c, mean_diff_c = test_correctness(B, H, S, D, config_id, is_causal=True)
        status_c = "✅ PASS" if passed_causal else "❌ FAIL"
        print(f"  Causal:     {status_c} (max_diff={max_diff_c:.6f}, mean_diff={mean_diff_c:.6f})")
        
        correctness_results[config_id] = passed and passed_causal
    
    # Check if any passed
    if not any(correctness_results.values()):
        print("\n❌ All configs failed correctness - stopping")
        return 1
    
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)
    
    # Benchmark passing configs
    perf_results = {}
    for config_id in [1, 2, 3]:
        if not correctness_results[config_id]:
            print(f"\nConfig {config_id}: SKIPPED (correctness failed)")
            continue
        
        cfg = CONFIGS[config_id]
        print(f"\nConfig {config_id}: BLOCK_M={cfg['BLOCK_M']}, BLOCK_N={cfg['BLOCK_N']}, WARPS={cfg['WARPS']}")
        print("  Warming up (20 iterations)...", end="", flush=True)
        
        try:
            latency = benchmark_config(B, H, S, D, config_id)
            perf_results[config_id] = latency
            print(f"\r  Latency: {latency:.4f} ms")
        except Exception as e:
            print(f"\r  ❌ Benchmark failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if perf_results:
        best_config = min(perf_results, key=perf_results.get)
        best_latency = perf_results[best_config]
        
        print(f"\n✅ Best Config: {best_config}")
        print(f"   Latency: {best_latency:.4f} ms")
        print(f"   Config: {CONFIGS[best_config]}")
        
        # Compare to V2 baseline (0.3184 ms)
        v2_baseline = 0.3184
        speedup = v2_baseline / best_latency
        improvement = (1 - best_latency / v2_baseline) * 100
        
        print(f"\nComparison to V2 (0.3184 ms):")
        print(f"   Speedup: {speedup:.2f}×")
        print(f"   Improvement: {improvement:+.1f}%")
        
        # Check target
        target = 0.255
        if best_latency <= target:
            print(f"\n🎯 ✅ TARGET ACHIEVED: {best_latency:.4f} ms ≤ {target:.4f} ms")
            return 0
        else:
            gap = (best_latency / target - 1) * 100
            print(f"\n🎯 ⚠️  Target: {target:.4f} ms (currently {gap:+.1f}% away)")
            return 0
    else:
        print("\n❌ No configs passed both correctness and performance")
        return 1


if __name__ == "__main__":
    sys.exit(main())
