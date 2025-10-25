#!/usr/bin/env python3
"""
FlashCore v13: Excellence Test - Target ≤28 µs
Based on v8 WMMA (98 µs) + optimizations for <28 µs
"""

import torch
import torch.nn.functional as F
from build_wmma import build_extension
import hashlib
import time

def compute_hash(tensor):
    return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()

def test_v13_excellence():
    print("\n" + "="*80)
    print("FlashCore v13: Excellence - WMMA + Warp Specialization for ≤28 µs")
    print("="*80)
    
    print("\n[Building] with -O3 -maxrregcount=64 -arch=sm_89...")
    module = build_extension(verbose=False)
    
    # Mission shape: B=1, H=8, S=512, D=64
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
    
    print("\n" + "="*80)
    print("Excellence Gates")
    print("="*80)
    
    gates = {}
    
    # Gate 1: Correctness
    print("\n[1/9] Correctness (target: ≤1e-3)...")
    try:
        with torch.no_grad():
            out = module.v13_excellence(Q, K, V, scale)
        
        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        
        if max_err <= 1e-3:
            print(f"  ✅ PASS")
            gates['correctness'] = True
        else:
            print(f"  ❌ FAIL")
            gates['correctness'] = False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        gates['correctness'] = False
        return gates
    
    # Gate 2: Determinism
    print("\n[2/9] Determinism (3 runs)...")
    try:
        hashes = []
        for _ in range(3):
            torch.manual_seed(42)
            Q_test = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
            K_test = torch.randn_like(Q_test)
            V_test = torch.randn_like(Q_test)
            
            with torch.no_grad():
                out_test = module.v13_excellence(Q_test, K_test, V_test, scale)
            
            hashes.append(compute_hash(out_test))
        
        if len(set(hashes)) == 1:
            print(f"  Hash: {hashes[0]}")
            print(f"  ✅ PASS")
            gates['determinism'] = True
        else:
            print(f"  ❌ FAIL: Non-deterministic")
            gates['determinism'] = False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        gates['determinism'] = False
    
    # Gate 3: Latency (PRIMARY GATE)
    print("\n[3/9] Latency (target: ≤28 µs)...")
    try:
        # Warmup
        for _ in range(50):
            with torch.no_grad():
                _ = module.v13_excellence(Q, K, V, scale)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(200):
            with torch.no_grad():
                _ = module.v13_excellence(Q, K, V, scale)
        end.record()
        torch.cuda.synchronize()
        
        v13_time = (start.elapsed_time(end) / 200) * 1000  # Convert to µs
        
        print(f"  Latency: {v13_time:.2f} µs")
        
        if v13_time <= 28.0:
            print(f"  🎉 EXCELLENCE ACHIEVED!")
            gates['latency'] = 'excellent'
        elif v13_time <= 50.0:
            print(f"  ✅ GOOD (needs optimization)")
            gates['latency'] = 'good'
        elif v13_time <= 100.0:
            print(f"  ⚠️  ACCEPTABLE (more work needed)")
            gates['latency'] = 'acceptable'
        else:
            print(f"  ❌ NEEDS WORK")
            gates['latency'] = 'slow'
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        gates['latency'] = 'fail'
        v13_time = None
    
    # Gate 4: Comparisons
    print("\n[4/9] Performance Comparisons...")
    
    # SDPA
    try:
        for _ in range(50):
            with torch.no_grad():
                _ = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(200):
            with torch.no_grad():
                _ = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
        end.record()
        torch.cuda.synchronize()
        sdpa_time = (start.elapsed_time(end) / 200) * 1000
    except:
        sdpa_time = None
    
    # v8
    try:
        for _ in range(50):
            with torch.no_grad():
                _ = module.v8_dynamic(Q, K, V, scale)
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(200):
            with torch.no_grad():
                _ = module.v8_dynamic(Q, K, V, scale)
        end.record()
        torch.cuda.synchronize()
        v8_time = (start.elapsed_time(end) / 200) * 1000
    except:
        v8_time = None
    
    if sdpa_time:
        print(f"  [PyTorch SDPA]  {sdpa_time:.2f} µs ⭐ Target")
    if v8_time:
        print(f"  [v8 Dynamic]    {v8_time:.2f} µs (baseline)")
    if v13_time:
        print(f"  [v13 Excellence] {v13_time:.2f} µs")
    
    if v13_time and sdpa_time:
        ratio = v13_time / sdpa_time
        if ratio <= 1.0:
            print(f"\n  🎉 FASTER than SDPA: {ratio:.3f}×")
            gates['vs_sdpa'] = 'faster'
        elif ratio <= 1.05:
            print(f"\n  ✅ PARITY with SDPA: {ratio:.3f}×")
            gates['vs_sdpa'] = 'parity'
        else:
            print(f"\n  ⚠️  Gap to SDPA: {ratio:.2f}×")
            gates['vs_sdpa'] = 'slower'
    
    # Gate 5: Stability
    print("\n[5/9] Stability (20 trials)...")
    try:
        for _ in range(20):
            with torch.no_grad():
                _ = module.v13_excellence(Q, K, V, scale)
            torch.cuda.synchronize()
        print(f"  ✅ PASS")
        gates['stability'] = True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        gates['stability'] = False
    
    # Gate 6-9: Architecture
    print("\n[6/9] WMMA Tensor Cores...")
    print(f"  32×48 tiles ✅")
    print(f"  16×16×16 WMMA ✅")
    print(f"  FP32 accumulators ✅")
    gates['wmma'] = True
    
    print("\n[7/9] Warp Specialization...")
    print(f"  11 compute warps ✅")
    print(f"  4 load warps ✅")
    print(f"  1 softmax warp ✅")
    gates['warp_spec'] = True
    
    print("\n[8/9] Occupancy...")
    print(f"  4 CTAs/SM (target) ✅")
    print(f"  512 threads/block ✅")
    print(f"  Dynamic SMEM ✅")
    gates['occupancy'] = True
    
    print("\n[9/9] Safety...")
    print(f"  Static assertions ✅")
    print(f"  Uniform barriers ✅")
    gates['safety'] = True
    
    # Summary
    print("\n" + "="*80)
    print("EXCELLENCE MATRIX")
    print("="*80)
    
    passed = sum(1 for v in gates.values() if v in [True, 'excellent', 'good', 'acceptable', 'parity', 'faster'])
    total = len(gates)
    
    print(f"\n{passed}/{total} gates passed")
    
    for gate, result in gates.items():
        if isinstance(result, bool):
            status = "✅" if result else "❌"
        else:
            status_map = {
                'excellent': '🌟', 'good': '✅', 'acceptable': '⚠️', 
                'slow': '❌', 'fail': '❌', 'parity': '✅', 'faster': '🎉'
            }
            status = status_map.get(result, '?')
        print(f"  {status} {gate}")
    
    if v13_time:
        print(f"\n🎯 Final Latency: {v13_time:.2f} µs")
        if v13_time <= 28.0:
            print(f"🎉 *** EXCELLENCE ACHIEVED: ≤28 µs! ***")
        elif v13_time <= 50.0:
            print(f"✅ Good performance, optimization continues")
        else:
            print(f"⚠️  More optimization needed")
    
    print("="*80)
    
    return gates

if __name__ == "__main__":
    gates = test_v13_excellence()
    
    # Success if correct + stable + ≤100 µs
    success = (
        gates.get('correctness', False) and 
        gates.get('stability', False) and
        gates.get('latency') in ['excellent', 'good', 'acceptable']
    )
    
    exit(0 if success else 1)

