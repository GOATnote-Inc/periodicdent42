#!/usr/bin/env python3
"""
Test KV Cache Correctness

Validates FlashCore KV cache implementation against PyTorch SDPA reference
"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add flashcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashcore.fast.attention_production import attention_with_kv_cache


def test_kv_cache_vs_pytorch_prefill_decode():
    """
    Test KV cache correctness: prefill + multiple decode steps
    
    Compares incremental inference (with cache) to full attention (without cache)
    """
    print("=" * 70)
    print("TEST 1: KV Cache Correctness (Prefill + Decode)")
    print("=" * 70)
    
    B, H, S_prefill, S_decode, D = 2, 8, 64, 10, 64
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Create full sequence
    q_full = torch.randn(B, H, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    k_full = torch.randn(B, H, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    v_full = torch.randn(B, H, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    
    # PyTorch reference (MUST be causal to match incremental cache behavior)
    # Incremental decoding is implicitly causal: token i only sees tokens [0:i]
    print(f"\nComputing reference with PyTorch SDPA (causal)...")
    expected = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)
    
    # Our implementation with cache
    print(f"Computing with FlashCore KV cache...")
    
    # Step 1: Prefill (causal - each token in prefill only sees previous tokens)
    q_prefill = q_full[:, :, :S_prefill, :]
    k_prefill = k_full[:, :, :S_prefill, :]
    v_prefill = v_full[:, :, :S_prefill, :]
    
    output_prefill, cache = attention_with_kv_cache(
        q_prefill, k_prefill, v_prefill, 
        update_cache=True,
        is_causal=True  # Causal masking in prefill phase
    )
    
    print(f"  Prefill: S={S_prefill}, cache initialized")
    
    # Step 2: Decode (one token at a time, causal - new token sees cache + itself)
    outputs = [output_prefill]
    for t in range(S_decode):
        q_t = q_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        k_t = k_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        v_t = v_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        
        output_t, cache = attention_with_kv_cache(
            q_t, k_t, v_t, 
            past_key_value=cache, 
            update_cache=True,
            is_causal=True  # Causal masking in decode phase
        )
        outputs.append(output_t)
        
        if (t + 1) % 5 == 0:
            print(f"  Decode step {t+1}/{S_decode}")
    
    # Concatenate results
    result = torch.cat(outputs, dim=2)
    
    # Compare
    max_diff = (result - expected).abs().max().item()
    mean_diff = (result - expected).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Target:    rtol=1e-3, atol=2e-3 (FP16 tolerance)")
    
    # Check correctness (use proper FP16 tolerance like other tests)
    passed = torch.allclose(result, expected, atol=2e-3, rtol=1e-3)
    
    if passed:
        print("\n✅ PASS: KV cache matches PyTorch reference")
        return True
    else:
        print("\n❌ FAIL: KV cache differs from reference")
        print(f"   Max difference {max_diff} exceeds threshold 1e-3")
        return False


def test_kv_cache_first_call_no_cache():
    """
    Test that first call (no cache) works correctly
    """
    print("\n" + "=" * 70)
    print("TEST 2: First Call (No Cache)")
    print("=" * 70)
    
    B, H, S, D = 4, 8, 128, 64
    
    torch.manual_seed(43)
    
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # PyTorch reference
    expected = F.scaled_dot_product_attention(q, k, v)
    
    # FlashCore (first call, no cache)
    result, cache = attention_with_kv_cache(q, k, v, past_key_value=None, update_cache=True)
    
    # Compare
    max_diff = (result - expected).abs().max().item()
    print(f"\nMax diff: {max_diff:.6f}")
    
    # Check cache was created
    assert cache is not None, "Cache should be returned when update_cache=True"
    K_cache, V_cache, seq_lens = cache
    print(f"Cache shape: K={K_cache.shape}, V={V_cache.shape}, seq_lens={seq_lens.shape}")
    
    passed = torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
    
    if passed:
        print("✅ PASS: First call correctness")
        return True
    else:
        print(f"❌ FAIL: First call differs (max_diff={max_diff})")
        return False


def test_kv_cache_single_decode_step():
    """
    Test single decode step with cache
    """
    print("\n" + "=" * 70)
    print("TEST 3: Single Decode Step")
    print("=" * 70)
    
    B, H, S_cache, D = 2, 8, 256, 64
    
    torch.manual_seed(44)
    
    # Create cache (must be 3-tuple: K, V, seq_lens)
    # Note: Cache is pre-allocated to max size, but only S_cache tokens are filled
    cache_max_len = 4096
    K_cache = torch.zeros(B, H, cache_max_len, D, device='cuda', dtype=torch.float16)
    V_cache = torch.zeros(B, H, cache_max_len, D, device='cuda', dtype=torch.float16)
    
    # Fill first S_cache positions with actual data
    K_cache[:, :, :S_cache, :] = torch.randn(B, H, S_cache, D, device='cuda', dtype=torch.float16)
    V_cache[:, :, :S_cache, :] = torch.randn(B, H, S_cache, D, device='cuda', dtype=torch.float16)
    
    # seq_lens tracks actual fill (not max size!)
    seq_lens = torch.full((B,), S_cache, dtype=torch.int32, device='cuda')
    cache = (K_cache, V_cache, seq_lens)
    
    # New token
    q_new = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
    k_new = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
    v_new = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
    
    # Create full sequence for reference (decode phase: new token attends to cache + itself)
    q_full = q_new
    k_full = torch.cat([K_cache[:, :, :S_cache, :], k_new], dim=2)
    v_full = torch.cat([V_cache[:, :, :S_cache, :], v_new], dim=2)
    
    # PyTorch reference (causal: new token only sees cache + itself, not future)
    expected = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)
    
    # FlashCore with cache (causal)
    result, _ = attention_with_kv_cache(
        q_new, k_new, v_new,
        past_key_value=cache,
        update_cache=False,  # Don't modify cache for this test
        is_causal=True  # Causal masking in decode
    )
    
    # Compare
    max_diff = (result - expected).abs().max().item()
    print(f"\nMax diff: {max_diff:.6f}")
    print(f"Cache size: {S_cache}, New tokens: 1")
    
    passed = torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
    
    if passed:
        print("✅ PASS: Single decode step correctness")
        return True
    else:
        print(f"❌ FAIL: Decode step differs (max_diff={max_diff})")
        return False


def test_kv_cache_various_configs():
    """
    Test various configurations (different S, B, H)
    """
    print("\n" + "=" * 70)
    print("TEST 4: Various Configurations")
    print("=" * 70)
    
    configs = [
        (1, 8, 32, 64),    # Small
        (4, 8, 128, 64),   # Medium
        (8, 16, 256, 64),  # Larger
    ]
    
    all_passed = True
    
    for i, (B, H, S, D) in enumerate(configs):
        print(f"\n  Config {i+1}: B={B}, H={H}, S={S}, D={D}")
        
        torch.manual_seed(45 + i)
        
        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
        
        # Reference
        expected = F.scaled_dot_product_attention(q, k, v)
        
        # FlashCore
        result, _ = attention_with_kv_cache(q, k, v, update_cache=False)
        
        # Compare
        max_diff = (result - expected).abs().max().item()
        passed = torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
        
        status = "✅" if passed else "❌"
        print(f"    Max diff: {max_diff:.6f} {status}")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ PASS: All configurations")
        return True
    else:
        print("\n❌ FAIL: Some configurations failed")
        return False


def run_all_tests():
    """Run all KV cache correctness tests"""
    print("\n" + "=" * 70)
    print("FLASHCORE KV CACHE - CORRECTNESS TESTS")
    print("=" * 70)
    print()
    
    results = []
    
    # Run tests
    results.append(("Prefill + Decode", test_kv_cache_vs_pytorch_prefill_decode()))
    results.append(("First Call (No Cache)", test_kv_cache_first_call_no_cache()))
    results.append(("Single Decode Step", test_kv_cache_single_decode_step()))
    results.append(("Various Configurations", test_kv_cache_various_configs()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(run_all_tests())

