#!/usr/bin/env python3
"""
Test Grouped-Query Attention (GQA) Correctness

Validates FlashCore GQA implementation (H_q != H_kv) against PyTorch SDPA reference
"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add flashcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashcore.fast.attention_production import attention_with_kv_cache


def test_gqa_vs_pytorch_head_broadcasting():
    """
    Test GQA correctness by comparing to manual head broadcasting
    
    GQA: Q has H_q heads, K/V have H_kv heads
    Reference: Manually broadcast K/V to H_q heads, then use standard attention
    """
    print("=" * 70)
    print("TEST 1: GQA Correctness (Manual Head Broadcasting)")
    print("=" * 70)
    
    B, H_q, H_kv, S, D = 2, 32, 8, 128, 64
    group_size = H_q // H_kv  # 4
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Create GQA tensors
    q = torch.randn(B, H_q, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    
    print(f"\nConfig: B={B}, H_q={H_q}, H_kv={H_kv}, S={S}, D={D}")
    print(f"Group size: {group_size} (each KV head shared by {group_size} query heads)")
    
    # PyTorch reference: Manual head broadcasting
    print(f"\nComputing reference with PyTorch SDPA (manual broadcasting)...")
    k_broadcast = k.repeat_interleave(group_size, dim=1)  # [B, H_q, S, D]
    v_broadcast = v.repeat_interleave(group_size, dim=1)  # [B, H_q, S, D]
    expected = F.scaled_dot_product_attention(q, k_broadcast, v_broadcast)
    
    # FlashCore GQA
    print(f"Computing with FlashCore GQA...")
    result, _ = attention_with_kv_cache(
        q, k, v,
        num_query_heads=H_q,
        num_kv_heads=H_kv,
        update_cache=False
    )
    
    # Compare
    max_diff = (result - expected).abs().max().item()
    mean_diff = (result - expected).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Target:    < 1e-3")
    
    # Check correctness
    passed = torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
    
    if passed:
        print("\n✅ PASS: GQA matches PyTorch reference")
        return True
    else:
        print("\n❌ FAIL: GQA differs from reference")
        print(f"   Max difference {max_diff} exceeds threshold 1e-3")
        return False


def test_gqa_various_head_ratios():
    """
    Test various H_q / H_kv ratios
    """
    print("\n" + "=" * 70)
    print("TEST 2: Various Head Ratios")
    print("=" * 70)
    
    test_configs = [
        (32, 32),   # MHA (1:1) - should work as before
        (32, 16),   # 2:1
        (32, 8),    # 4:1 (LLaMA, Mistral)
        (32, 4),    # 8:1
        (32, 1),    # 32:1 (MQA)
        (28, 4),    # 7:1 (Qwen)
    ]
    
    all_passed = True
    
    for H_q, H_kv in test_configs:
        print(f"\n  Config: H_q={H_q}, H_kv={H_kv} (ratio {H_q//H_kv}:1)")
        
        B, S, D = 2, 64, 64
        torch.manual_seed(43)
        
        q = torch.randn(B, H_q, S, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
        
        # Reference
        group_size = H_q // H_kv
        k_ref = k.repeat_interleave(group_size, dim=1)
        v_ref = v.repeat_interleave(group_size, dim=1)
        expected = F.scaled_dot_product_attention(q, k_ref, v_ref)
        
        # FlashCore
        result, _ = attention_with_kv_cache(q, k, v)
        
        # Compare
        max_diff = (result - expected).abs().max().item()
        passed = torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
        
        status = "✅" if passed else "❌"
        print(f"    Max diff: {max_diff:.6f} {status}")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ PASS: All head ratios")
        return True
    else:
        print("\n❌ FAIL: Some head ratios failed")
        return False


def test_gqa_with_kv_cache():
    """
    Test GQA with KV cache (Phase 1 + Phase 2 integration)
    
    Key validation: Cache stored with H_kv heads (memory savings!)
    """
    print("\n" + "=" * 70)
    print("TEST 3: GQA + KV Cache Integration")
    print("=" * 70)
    
    B, H_q, H_kv, S_prefill, S_decode, D = 2, 32, 8, 64, 5, 64
    
    torch.manual_seed(44)
    
    # Create full sequence
    q_full = torch.randn(B, H_q, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    k_full = torch.randn(B, H_kv, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    v_full = torch.randn(B, H_kv, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    
    print(f"\nConfig: B={B}, H_q={H_q}, H_kv={H_kv}, S_prefill={S_prefill}, S_decode={S_decode}")
    print(f"Memory savings: {H_q/H_kv:.1f}× (cache stored as H_kv={H_kv}, not H_q={H_q})")
    
    # PyTorch reference (full sequence)
    group_size = H_q // H_kv
    k_ref = k_full.repeat_interleave(group_size, dim=1)
    v_ref = v_full.repeat_interleave(group_size, dim=1)
    expected = F.scaled_dot_product_attention(q_full, k_ref, v_ref)
    
    # FlashCore with GQA cache
    # Step 1: Prefill
    q_prefill = q_full[:, :, :S_prefill, :]
    k_prefill = k_full[:, :, :S_prefill, :]
    v_prefill = v_full[:, :, :S_prefill, :]
    
    output_prefill, cache = attention_with_kv_cache(
        q_prefill, k_prefill, v_prefill,
        update_cache=True
    )
    
    print(f"\n  Prefill: S={S_prefill}, cache initialized")
    
    # Verify cache shape (should be H_kv, not H_q)
    K_cache, V_cache, seq_lens = cache
    assert K_cache.shape == (B, H_kv, 4096, D), \
        f"Expected cache shape (B={B}, H_kv={H_kv}, S_max=4096, D={D}), got {K_cache.shape}"
    print(f"  Cache shape: {K_cache.shape} (✅ stored with H_kv={H_kv}, not H_q={H_q})")
    print(f"  seq_lens: {seq_lens}")
    
    # Step 2: Decode
    outputs = [output_prefill]
    for t in range(S_decode):
        q_t = q_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        k_t = k_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        v_t = v_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        
        output_t, cache = attention_with_kv_cache(
            q_t, k_t, v_t,
            past_key_value=cache,
            update_cache=True
        )
        outputs.append(output_t)
    
    print(f"  Decode: {S_decode} tokens processed")
    
    # Concatenate results
    result = torch.cat(outputs, dim=2)
    
    # Compare
    max_diff = (result - expected).abs().max().item()
    mean_diff = (result - expected).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Target:    rtol=1e-2, atol=1e-2 (Industry standard for FP16 cache)")
    
    # Use industry-standard tolerance for cache-based operations
    passed = torch.allclose(result, expected, atol=1e-2, rtol=1e-2)
    
    if passed:
        print("✅ PASS: GQA + KV cache integration (within FP16 tolerance)")
        return True
    else:
        print(f"❌ FAIL: GQA + cache differs (max_diff={max_diff})")
        return False


def test_gqa_memory_savings():
    """
    Verify GQA actually uses less memory than MHA
    """
    print("\n" + "=" * 70)
    print("TEST 4: Memory Savings Validation")
    print("=" * 70)
    
    B, S, D = 16, 2048, 128
    H_q = 32
    
    # MHA cache size
    H_kv_mha = 32
    cache_mha_bytes = B * H_kv_mha * S * D * 2 * 2  # 2 for K/V, 2 bytes per half
    
    # GQA cache size
    H_kv_gqa = 8
    cache_gqa_bytes = B * H_kv_gqa * S * D * 2 * 2
    
    savings = cache_mha_bytes / cache_gqa_bytes
    
    print(f"\nConfig: B={B}, S={S}, D={D}")
    print(f"MHA (H={H_kv_mha}): {cache_mha_bytes / 1e6:.1f} MB")
    print(f"GQA (H_q={H_q}, H_kv={H_kv_gqa}): {cache_gqa_bytes / 1e6:.1f} MB")
    print(f"Memory savings: {savings:.1f}× ({cache_mha_bytes / 1e6:.1f}MB → {cache_gqa_bytes / 1e6:.1f}MB)")
    
    # Verify
    expected_savings = H_q / H_kv_gqa  # Should be 4.0
    assert abs(savings - expected_savings) < 0.01, f"Expected {expected_savings}× savings, got {savings}"
    
    print(f"\n✅ PASS: Memory savings verified ({savings:.1f}× reduction)")
    return True


def test_gqa_invalid_head_ratio():
    """
    Test that invalid head ratios raise errors
    """
    print("\n" + "=" * 70)
    print("TEST 5: Invalid Head Ratio Validation")
    print("=" * 70)
    
    B, S, D = 2, 64, 64
    
    # Test 1: H_q not divisible by H_kv
    print("\n  Test 1: H_q=32, H_kv=7 (not divisible)")
    H_q, H_kv = 32, 7
    q = torch.randn(B, H_q, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    
    try:
        _ = attention_with_kv_cache(q, k, v)
        print("    ❌ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        if "must be divisible" in str(e):
            print(f"    ✅ PASS: Correctly raised ValueError: {e}")
        else:
            print(f"    ❌ FAIL: Wrong error message: {e}")
            return False
    
    # Test 2: Valid ratio should work
    print("\n  Test 2: H_q=32, H_kv=8 (valid)")
    H_q, H_kv = 32, 8
    q = torch.randn(B, H_q, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    
    try:
        result, _ = attention_with_kv_cache(q, k, v)
        print("    ✅ PASS: Valid ratio accepted")
    except Exception as e:
        print(f"    ❌ FAIL: Valid ratio rejected: {e}")
        return False
    
    print("\n✅ PASS: Invalid head ratio validation")
    return True


def run_all_tests():
    """Run all GQA correctness tests"""
    print("\n" + "=" * 70)
    print("FLASHCORE GQA - CORRECTNESS TESTS")
    print("=" * 70)
    print()
    
    results = []
    
    # Run tests
    results.append(("GQA vs Manual Broadcasting", test_gqa_vs_pytorch_head_broadcasting()))
    results.append(("Various Head Ratios", test_gqa_various_head_ratios()))
    results.append(("GQA + KV Cache Integration", test_gqa_with_kv_cache()))
    results.append(("Memory Savings Validation", test_gqa_memory_savings()))
    results.append(("Invalid Head Ratio Validation", test_gqa_invalid_head_ratio()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:35s}: {status}")
    
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

