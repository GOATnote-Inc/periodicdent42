# PHASE 2: Grouped-Query Attention (Triton Adaptation)

**Priority**: 🔴 CRITICAL (Unlocks LLaMA 3.1, Mistral)  
**Effort**: 35-40 hours  
**Implementation**: **Triton** (extending Phase 1)  
**Complexity**: Medium-High (head broadcasting + memory layout)

---

## 🎯 **OBJECTIVE**

Support modern efficient architectures where query heads > KV heads:

```python
# FROM (current - Multi-Head Attention):
Q: [B, H=32, S, D]
K: [B, H=32, S, D]  # Same number of heads

# TO (target - Grouped-Query Attention):
Q: [B, H_q=32, S, D]
K: [B, H_kv=8, S, D]   # Fewer heads (4× reduction)
V: [B, H_kv=8, S, D]   # Each KV head shared by 4 query heads
```

**Impact**: 
- LLaMA 3.1: 32 query heads, 8 KV heads (75% KV memory reduction)
- Mistral 7B: 32 query heads, 8 KV heads
- Qwen 2.5: 28 query heads, 4 KV heads

---

## 📐 **DESIGN OVERVIEW**

### **Head Group Mapping**

```python
# Each KV head serves multiple query heads
group_size = H_q // H_kv  # e.g., 32 // 8 = 4

# Query head i uses KV head j where:
kv_head_idx = q_head_idx // group_size

# Example (H_q=32, H_kv=8, group_size=4):
Q heads [0,1,2,3]    → K/V head 0
Q heads [4,5,6,7]    → K/V head 1
Q heads [8,9,10,11]  → K/V head 2
...
Q heads [28,29,30,31] → K/V head 7
```

### **Integration with Phase 1 (KV Cache)**

**Key Insight**: Cache is stored with **H_kv** heads, not H_q (memory savings!)

```python
# Cache shapes:
K_cache: [B, H_kv=8, S_max, D]   # NOT [B, H_q=32, S_max, D]
V_cache: [B, H_kv=8, S_max, D]

# Memory savings: 4× reduction (32 heads → 8 heads)
```

---

## 🔧 **TRITON KERNEL MODIFICATIONS**

### **Phase 1 Kernel (MHA only)**
```python
@triton.jit
def _attention_kv_cache_fwd(
    Q, K_new, V_new,
    K_cache, V_cache, seq_lens, O,
    # ... strides ...
    B: tl.constexpr, H: tl.constexpr,  # H_q = H_kv = H
    S_q: tl.constexpr, S_max: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)  # Query head
    q_block_idx = tl.program_id(2)
    
    # For MHA: kv_head_idx = head_idx (same)
    kv_head_idx = head_idx
```

### **Phase 2 Kernel (GQA support)**
```python
@triton.jit
def _attention_gqa_kv_cache_fwd(
    Q, K_new, V_new,          # Q: [B, H_q, S_q, D], K/V: [B, H_kv, S_q, D]
    K_cache, V_cache,         # [B, H_kv, S_max, D]
    seq_lens, O,              # [B], [B, H_q, S_q, D]
    # ... strides ...
    B: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr,  # NEW: separate
    S_q: tl.constexpr, S_max: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)  # Query head [0, H_q)
    q_block_idx = tl.program_id(2)
    
    # NEW: Compute which KV head to use (GQA mapping)
    group_size = H_q // H_kv
    kv_head_idx = q_head_idx // group_size  # e.g., heads 0-3 → 0, 4-7 → 1
    
    # Get sequence length for this batch
    seq_len = tl.load(seq_lens + batch_idx)
    
    # Query block positions
    offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S_q
    offs_d = tl.arange(0, D)
    
    # Load query block (from Q_head)
    q_ptrs = (Q + batch_idx * stride_qb + q_head_idx * stride_qh +
              offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Attend to CACHED keys/values (from kv_head_idx, NOT q_head_idx)
    for k_block_start in range(0, seq_len, BLOCK_N):
        offs_n = k_block_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        # Load cached K block (using kv_head_idx for GQA)
        k_ptrs = (K_cache + batch_idx * stride_kcb + kv_head_idx * stride_kch +
                  offs_n[:, None] * stride_kcm + offs_d[None, :] * stride_kcd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute scores
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk *= SCALE
        
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(p, axis=1)
        
        # Load cached V block (using kv_head_idx)
        v_ptrs = (V_cache + batch_idx * stride_vcb + kv_head_idx * stride_vch +
                  offs_n[:, None] * stride_vcm + offs_d[None, :] * stride_vcd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Update accumulator
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_ij
        l_i = l_ij
    
    # Attend to NEW keys/values (also from kv_head_idx)
    for k_block_start in range(0, S_q, BLOCK_N):
        offs_n = k_block_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S_q
        
        # Load new K block (using kv_head_idx for GQA)
        k_ptrs = (K_new + batch_idx * stride_knb + kv_head_idx * stride_knh +
                  offs_n[:, None] * stride_knm + offs_d[None, :] * stride_knd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute scores
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk *= SCALE
        
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(p, axis=1)
        
        # Load new V block (using kv_head_idx)
        v_ptrs = (V_new + batch_idx * stride_vnb + kv_head_idx * stride_vnh +
                  offs_n[:, None] * stride_vnm + offs_d[None, :] * stride_vnd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Update accumulator
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_ij
        l_i = l_ij
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output (to q_head_idx position)
    o_ptrs = (O + batch_idx * stride_ob + q_head_idx * stride_oh +
              offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])
```

---

## 🐍 **PYTHON WRAPPER UPDATES**

### **Updated Function Signature**
```python
def attention_with_kv_cache(
    query: torch.Tensor,                    # [B, H_q, S_q, D]
    key: torch.Tensor,                      # [B, H_kv, S_q, D]  # NOTE: H_kv
    value: torch.Tensor,                    # [B, H_kv, S_q, D]
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    seq_lens: Optional[torch.Tensor] = None,
    cache_max_len: int = 4096,
    update_cache: bool = True,
    is_causal: bool = False,              # Phase 3
    num_query_heads: Optional[int] = None,   # NEW: explicit H_q
    num_kv_heads: Optional[int] = None       # NEW: explicit H_kv
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Triton-based attention with GQA and KV cache support.
    
    Args:
        query: Query tensor [B, H_q, S_q, D]
        key: Key tensor [B, H_kv, S_q, D] where H_kv <= H_q
        value: Value tensor [B, H_kv, S_q, D]
        num_query_heads: Number of query heads (H_q)
        num_kv_heads: Number of KV heads (H_kv)
    
    Constraints:
        - H_q % H_kv == 0 (group_size must be integer)
    
    Returns:
        output: [B, H_q, S_q, D]
        cache: (K_cache [B, H_kv, ...], V_cache [B, H_kv, ...])
    """
    B, H_q, S_q, D = query.shape
    _, H_kv, _, _ = key.shape
    
    # Validate GQA constraint
    if H_q % H_kv != 0:
        raise ValueError(
            f"num_query_heads ({H_q}) must be divisible by num_kv_heads ({H_kv}). "
            f"Got group_size = {H_q / H_kv:.2f} (must be integer)."
        )
    
    group_size = H_q // H_kv
    
    # Handle cache (stored with H_kv heads, not H_q)
    if past_key_value is None:
        K_cache = torch.empty(B, H_kv, cache_max_len, D, device=query.device, dtype=query.dtype)
        V_cache = torch.empty(B, H_kv, cache_max_len, D, device=query.device, dtype=query.dtype)
        if seq_lens is None:
            seq_lens = torch.zeros(B, dtype=torch.int32, device=query.device)
    else:
        K_cache, V_cache = past_key_value
        # Validate cache has H_kv heads
        assert K_cache.shape[1] == H_kv, f"Cache has {K_cache.shape[1]} heads, expected {H_kv}"
        if seq_lens is None:
            seq_lens = torch.full((B,), K_cache.shape[2], dtype=torch.int32, device=query.device)
    
    # Allocate output (H_q heads)
    output = torch.empty_like(query)
    
    # Launch kernel with GQA support
    grid = lambda META: (B, H_q, triton.cdiv(S_q, META['BLOCK_M']))
    
    _attention_gqa_kv_cache_fwd[grid](
        query, key, value,
        K_cache, V_cache,
        seq_lens,
        output,
        # Strides
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        K_cache.stride(0), K_cache.stride(1), K_cache.stride(2), K_cache.stride(3),
        V_cache.stride(0), V_cache.stride(1), V_cache.stride(2), V_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Dimensions (now separate H_q and H_kv)
        B, H_q, H_kv, S_q, cache_max_len, D,
        # Tuning parameters
        BLOCK_M=64,
        BLOCK_N=64,
        SCALE=1.0 / (D ** 0.5)
    )
    
    # Update cache if requested (append to H_kv-shaped cache)
    if update_cache:
        for b in range(B):
            start_idx = seq_lens[b].item()
            end_idx = start_idx + S_q
            K_cache[b, :, start_idx:end_idx, :] = key[b]
            V_cache[b, :, start_idx:end_idx, :] = value[b]
            seq_lens[b] += S_q
    
    return output, (K_cache, V_cache) if update_cache else None
```

---

## ✅ **TESTING STRATEGY**

### **Test 1: GQA Correctness vs Reference**
```python
def test_gqa_correctness():
    """Test GQA against manual head broadcasting."""
    B, S, D = 4, 128, 128
    H_q, H_kv = 32, 8
    group_size = H_q // H_kv  # 4
    
    q = torch.randn(B, H_q, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    
    # Reference: Manual head broadcasting
    k_broadcast = k.repeat_interleave(group_size, dim=1)  # [B, H_q, S, D]
    v_broadcast = v.repeat_interleave(group_size, dim=1)
    
    expected = F.scaled_dot_product_attention(q, k_broadcast, v_broadcast)
    
    # Our GQA implementation
    result, _ = attention_with_kv_cache(q, k, v, num_query_heads=H_q, num_kv_heads=H_kv)
    
    assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
    print("✅ GQA correctness test passed")
```

### **Test 2: Different Head Ratios**
```python
def test_head_ratios():
    """Test various H_q / H_kv ratios."""
    test_configs = [
        (32, 32),   # MHA (1:1)
        (32, 16),   # 2:1
        (32, 8),    # 4:1 (LLaMA, Mistral)
        (32, 4),    # 8:1
        (32, 1),    # 32:1 (MQA)
        (28, 4),    # 7:1 (Qwen)
    ]
    
    for H_q, H_kv in test_configs:
        print(f"Testing H_q={H_q}, H_kv={H_kv} (ratio {H_q//H_kv}:1)")
        
        q = torch.randn(2, H_q, 64, 128, device='cuda', dtype=torch.float16)
        k = torch.randn(2, H_kv, 64, 128, device='cuda', dtype=torch.float16)
        v = torch.randn(2, H_kv, 64, 128, device='cuda', dtype=torch.float16)
        
        # Our implementation
        output, _ = attention_with_kv_cache(q, k, v)
        
        # Reference
        group_size = H_q // H_kv
        k_ref = k.repeat_interleave(group_size, dim=1)
        v_ref = v.repeat_interleave(group_size, dim=1)
        expected = F.scaled_dot_product_attention(q, k_ref, v_ref)
        
        assert torch.allclose(output, expected, atol=1e-3, rtol=1e-3)
        print(f"  ✅ Passed")
```

### **Test 3: GQA + KV Cache Integration**
```python
def test_gqa_with_cache():
    """Test GQA with KV cache from Phase 1."""
    B, H_q, H_kv, S, D = 4, 32, 8, 128, 128
    
    # Prefill
    q = torch.randn(B, H_q, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    
    output_prefill, cache = attention_with_kv_cache(q, k, v, update_cache=True)
    
    # Verify cache shape (should be H_kv, not H_q)
    K_cache, V_cache = cache
    assert K_cache.shape == (B, H_kv, S, D), f"Expected {(B, H_kv, S, D)}, got {K_cache.shape}"
    assert V_cache.shape == (B, H_kv, S, D)
    print(f"✅ Cache shape correct: [B={B}, H_kv={H_kv}, S={S}, D={D}]")
    print(f"   Memory saved: {(H_q / H_kv):.1f}× vs MHA")
    
    # Decode
    q_new = torch.randn(B, H_q, 1, D, device='cuda', dtype=torch.float16)
    k_new = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
    v_new = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
    
    output_decode, cache_updated = attention_with_kv_cache(
        q_new, k_new, v_new, past_key_value=cache, update_cache=True
    )
    
    # Verify updated cache shape
    K_cache_new, V_cache_new = cache_updated
    assert K_cache_new.shape == (B, H_kv, S + 1, D)
    print("✅ GQA + KV cache integration test passed")
```

### **Test 4: Memory Savings Validation**
```python
def test_memory_savings():
    """Verify GQA uses less memory than MHA."""
    B, S, D = 16, 2048, 128
    H_q = 32
    
    # MHA cache
    H_kv_mha = 32
    cache_mha_size = B * H_kv_mha * S * D * 2 * 2  # 2 for K/V, 2 bytes per half
    
    # GQA cache
    H_kv_gqa = 8
    cache_gqa_size = B * H_kv_gqa * S * D * 2 * 2
    
    savings = cache_mha_size / cache_gqa_size
    print(f"Memory savings: {savings:.1f}× ({cache_mha_size / 1e6:.1f}MB → {cache_gqa_size / 1e6:.1f}MB)")
    
    assert savings == 4.0  # 32/8 = 4×
    print("✅ Memory savings test passed")
```

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Step 1: Kernel Modifications (10-12 hours)**
- [ ] Add H_q and H_kv parameters to kernel
- [ ] Implement kv_head_idx = q_head_idx // group_size mapping
- [ ] Update all K/V loads to use kv_head_idx
- [ ] Test with single config (H_q=32, H_kv=8)

### **Step 2: Python Wrapper Updates (6-8 hours)**
- [ ] Add num_query_heads and num_kv_heads parameters
- [ ] Add validation (H_q % H_kv == 0)
- [ ] Update cache handling (H_kv shapes)
- [ ] Update docstrings with GQA examples

### **Step 3: Testing (12-15 hours)**
- [ ] Implement Test 1 (GQA correctness)
- [ ] Implement Test 2 (head ratios)
- [ ] Implement Test 3 (GQA + cache)
- [ ] Implement Test 4 (memory savings)
- [ ] Verify all Phase 1 tests still pass

### **Step 4: Optimization (6-8 hours)**
- [ ] Profile memory access patterns
- [ ] Verify no regression when H_q = H_kv (MHA case)
- [ ] Benchmark LLaMA config (H_q=32, H_kv=8)

---

## 🎯 **ACCEPTANCE CRITERIA**

### **Functional**
- ✅ Supports H_q / H_kv ratios: 1:1, 2:1, 4:1, 8:1, 32:1
- ✅ Correctness vs reference (error < 1e-3)
- ✅ Works with KV cache from Phase 1
- ✅ Validates H_q % H_kv == 0

### **Performance**
- ✅ MHA case (H_q = H_kv): <5% regression vs Phase 1
- ✅ GQA case: 4× memory savings (H_kv = H_q / 4)
- ✅ LLaMA config: Decode <10μs (same as Phase 1)

### **Integration**
- ✅ All Phase 1 tests pass with GQA
- ✅ Cache memory reduced by factor of (H_q / H_kv)

---

## 🚀 **READY TO IMPLEMENT**

**Prerequisites**: Phase 1 (KV Cache) complete and tested

**Next Step**: Modify `_attention_kv_cache_fwd` to support separate H_q and H_kv

**Command for Cursor**:
```
Implement Grouped-Query Attention (GQA) in Triton.
Extend Phase 1 kernel with H_q != H_kv support.
Target: LLaMA 3.1 config (H_q=32, H_kv=8, D=128)
Correctness: <1e-3 error vs PyTorch
Performance: No regression when H_q = H_kv
```

---

**Status**: ⏳ Awaiting Phase 1 completion  
**Estimated Time**: 35-40 hours  
**Priority**: 🔴 CRITICAL (Unlocks modern LLMs)

