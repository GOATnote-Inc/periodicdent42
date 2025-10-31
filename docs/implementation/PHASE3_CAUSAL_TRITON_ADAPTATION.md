# PHASE 3: Causal Masking (Triton Adaptation)

**Priority**: 🔴 CRITICAL (Required for all autoregressive models)  
**Effort**: 10-15 hours  
**Implementation**: **Triton** (extending Phase 2)  
**Complexity**: Low-Medium (well-understood problem)

---

## 🎯 **OBJECTIVE**

Enable autoregressive generation by preventing attention to future tokens:

```python
# FROM (current - bidirectional attention):
# Token i can attend to all tokens [0, S)
attention_scores[i] = softmax(Q[i] @ K.T)  # Full matrix

# TO (target - causal attention):
# Token i can ONLY attend to tokens [0, i]
attention_scores[i, j] = softmax(Q[i] @ K[j]) if j <= i else 0
```

**Visual**:
```
Bidirectional (current):        Causal (target):
  0 1 2 3 4                       0 1 2 3 4
0 ✓ ✓ ✓ ✓ ✓                     0 ✓ ✗ ✗ ✗ ✗
1 ✓ ✓ ✓ ✓ ✓                     1 ✓ ✓ ✗ ✗ ✗
2 ✓ ✓ ✓ ✓ ✓                     2 ✓ ✓ ✓ ✗ ✗
3 ✓ ✓ ✓ ✓ ✓                     3 ✓ ✓ ✓ ✓ ✗
4 ✓ ✓ ✓ ✓ ✓                     4 ✓ ✓ ✓ ✓ ✓
```

**Impact**: Required by ALL modern LLMs (GPT, LLaMA, Mistral, Qwen, etc.)

---

## 📐 **DESIGN OVERVIEW**

### **Masking Strategy for Triton**

**Triton Advantage**: Can efficiently mask during QK computation using element-wise operations

```python
# Compute causal mask
q_pos = seq_len_cache + offs_m  # Absolute position of query
k_pos = offs_n                   # Position of key
causal_mask = q_pos[:, None] >= k_pos[None, :]  # [BLOCK_M, BLOCK_N]

# Apply mask to scores
qk = tl.where(causal_mask, qk, float('-inf'))  # Mask future tokens
```

**Decision**: Use `tl.where` masking (Triton-native, efficient)

---

## 🔧 **TRITON KERNEL MODIFICATIONS**

### **Phase 2 Kernel (No Causal Masking)**
```python
@triton.jit
def _attention_gqa_kv_cache_fwd(...):
    # ...
    
    # Compute scores (no masking)
    qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    qk *= SCALE
    
    # Online softmax
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    p = tl.exp(qk - m_ij[:, None])
```

### **Phase 3 Kernel (With Causal Masking)**
```python
@triton.jit
def _attention_gqa_kv_cache_causal_fwd(
    Q, K_new, V_new,
    K_cache, V_cache, seq_lens, O,
    # ... strides ...
    B: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr,
    S_q: tl.constexpr, S_max: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr  # NEW: Causal flag
):
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)
    
    # Compute KV head for GQA
    group_size = H_q // H_kv
    kv_head_idx = q_head_idx // group_size
    
    # Get sequence length for this batch
    seq_len_cache = tl.load(seq_lens + batch_idx)
    
    # Query block positions
    offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S_q
    offs_d = tl.arange(0, D)
    
    # Compute ABSOLUTE position of queries (for causal masking)
    q_pos = seq_len_cache + offs_m  # Absolute position in full sequence
    
    # Load query block
    q_ptrs = (Q + batch_idx * stride_qb + q_head_idx * stride_qh +
              offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # ========== ATTEND TO CACHED KEYS/VALUES ==========
    for k_block_start in range(0, seq_len_cache, BLOCK_N):
        offs_n = k_block_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len_cache
        
        # Load cached K block
        k_ptrs = (K_cache + batch_idx * stride_kcb + kv_head_idx * stride_kch +
                  offs_n[:, None] * stride_kcm + offs_d[None, :] * stride_kcd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute scores
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk *= SCALE
        
        # ========== NEW: CAUSAL MASKING ==========
        if IS_CAUSAL:
            # Cache positions are [0, seq_len_cache)
            k_pos = offs_n
            # Causal mask: q_pos >= k_pos (can attend to past)
            causal_mask = q_pos[:, None] >= k_pos[None, :]
            # Apply mask: set future tokens to -inf
            qk = tl.where(causal_mask, qk, float('-inf'))
        # ==========================================
        
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(p, axis=1)
        
        # Load cached V block
        v_ptrs = (V_cache + batch_idx * stride_vcb + kv_head_idx * stride_vch +
                  offs_n[:, None] * stride_vcm + offs_d[None, :] * stride_vcd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Update accumulator
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_ij
        l_i = l_ij
    
    # ========== ATTEND TO NEW KEYS/VALUES ==========
    for k_block_start in range(0, S_q, BLOCK_N):
        offs_n = k_block_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S_q
        
        # Load new K block
        k_ptrs = (K_new + batch_idx * stride_knb + kv_head_idx * stride_knh +
                  offs_n[:, None] * stride_knm + offs_d[None, :] * stride_knd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute scores
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk *= SCALE
        
        # ========== NEW: CAUSAL MASKING ==========
        if IS_CAUSAL:
            # New token positions are [seq_len_cache, seq_len_cache + S_q)
            k_pos = seq_len_cache + offs_n
            # Causal mask: q_pos >= k_pos
            causal_mask = q_pos[:, None] >= k_pos[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))
        # ==========================================
        
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(p, axis=1)
        
        # Load new V block
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
    
    # Store output
    o_ptrs = (O + batch_idx * stride_ob + q_head_idx * stride_oh +
              offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])
```

---

## 🐍 **PYTHON WRAPPER UPDATES**

### **Updated Function Signature**
```python
def attention_with_kv_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    seq_lens: Optional[torch.Tensor] = None,
    cache_max_len: int = 4096,
    update_cache: bool = True,
    is_causal: bool = False,  # NEW PARAMETER
    num_query_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        is_causal: If True, apply causal masking (prevent attention to future).
                   Required for autoregressive generation (GPT, LLaMA, etc.)
    """
    B, H_q, S_q, D = query.shape
    _, H_kv, _, _ = key.shape
    
    # ... (cache handling same as Phase 2) ...
    
    # Launch kernel with causal support
    grid = lambda META: (B, H_q, triton.cdiv(S_q, META['BLOCK_M']))
    
    _attention_gqa_kv_cache_causal_fwd[grid](
        query, key, value,
        K_cache, V_cache,
        seq_lens,
        output,
        # ... strides ...
        B, H_q, H_kv, S_q, cache_max_len, D,
        BLOCK_M=64,
        BLOCK_N=64,
        SCALE=1.0 / (D ** 0.5),
        IS_CAUSAL=is_causal  # Pass as constexpr for compile-time optimization
    )
    
    # ... (cache update same as Phase 2) ...
    
    return output, (K_cache, V_cache) if update_cache else None
```

### **Usage Example: GPT-Style Generation**
```python
# Initial prompt: "Once upon a time"
tokens = tokenizer.encode("Once upon a time")  # [512, 2402, 247, 673]
B = 1

# Prefill with causal masking
q = model.get_queries(tokens)  # [1, 32, 4, 128]
k = model.get_keys(tokens)     # [1, 8, 4, 128] (GQA)
v = model.get_values(tokens)

output, cache = attention_with_kv_cache(
    q, k, v,
    is_causal=True,  # Enable causal masking
    update_cache=True
)
# Each position only attends to previous positions

# Generate next 100 tokens
for step in range(100):
    # Sample next token
    logits = model.lm_head(output[:, :, -1, :])
    next_token = torch.argmax(logits, dim=-1)
    
    # Get Q/K/V for new token
    q_new = model.get_queries([next_token])  # [1, 32, 1, 128]
    k_new = model.get_keys([next_token])     # [1, 8, 1, 128]
    v_new = model.get_values([next_token])
    
    # Decode step (new token attends to all cache)
    output, cache = attention_with_kv_cache(
        q_new, k_new, v_new,
        past_key_value=cache,
        is_causal=True,  # Still set (though effectively no-op in decode)
        update_cache=True
    )
```

---

## ✅ **TESTING STRATEGY**

### **Test 1: Correctness vs PyTorch**
```python
def test_causal_correctness():
    """Compare to PyTorch causal attention."""
    B, H_q, H_kv, S, D = 4, 32, 8, 128, 128
    
    q = torch.randn(B, H_q, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    
    # PyTorch reference (with GQA)
    group_size = H_q // H_kv
    k_ref = k.repeat_interleave(group_size, dim=1)
    v_ref = v.repeat_interleave(group_size, dim=1)
    expected = F.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=True)
    
    # Our implementation
    result, _ = attention_with_kv_cache(q, k, v, is_causal=True)
    
    assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
    print("✅ Causal correctness test passed")
```

### **Test 2: Manual Masking Verification**
```python
def test_causal_mask_structure():
    """Verify upper triangle is actually masked."""
    B, H_q, H_kv, S, D = 1, 8, 8, 8, 64
    
    # Simple input to see attention pattern clearly
    q = torch.ones(B, H_q, S, D, device='cuda', dtype=torch.float16)
    k = torch.ones(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    v = torch.arange(S, device='cuda', dtype=torch.float16).view(1, 1, S, 1).repeat(1, H_kv, 1, D)
    
    output, _ = attention_with_kv_cache(q, k, v, is_causal=True)
    
    # With causal masking and uniform Q/K:
    # Position i attends to [0:i+1] uniformly
    # So output[i] ≈ mean(v[0:i+1])
    for i in range(S):
        expected_avg = sum(range(i + 1)) / (i + 1)
        actual_avg = output[0, 0, i, 0].item()
        assert abs(actual_avg - expected_avg) < 0.5, f"Position {i}: expected {expected_avg}, got {actual_avg}"
    
    print("✅ Causal mask structure test passed")
```

### **Test 3: Causal with KV Cache**
```python
def test_causal_with_cache():
    """Test causal masking with KV cache."""
    B, H_q, H_kv, S_prefill, D = 2, 32, 8, 64, 128
    
    torch.manual_seed(42)
    
    # Prefill phase
    q_prefill = torch.randn(B, H_q, S_prefill, D, device='cuda', dtype=torch.float16)
    k_prefill = torch.randn(B, H_kv, S_prefill, D, device='cuda', dtype=torch.float16)
    v_prefill = torch.randn(B, H_kv, S_prefill, D, device='cuda', dtype=torch.float16)
    
    output_prefill, cache = attention_with_kv_cache(
        q_prefill, k_prefill, v_prefill,
        is_causal=True,
        update_cache=True
    )
    
    # Decode phase (10 steps)
    torch.manual_seed(43)
    outputs = [output_prefill]
    for step in range(10):
        q_new = torch.randn(B, H_q, 1, D, device='cuda', dtype=torch.float16)
        k_new = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
        v_new = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
        
        output_new, cache = attention_with_kv_cache(
            q_new, k_new, v_new,
            past_key_value=cache,
            is_causal=True,
            update_cache=True
        )
        outputs.append(output_new)
    
    result = torch.cat(outputs, dim=2)  # [B, H_q, S_prefill + 10, D]
    
    # Verify shape
    assert result.shape == (B, H_q, S_prefill + 10, D)
    print("✅ Causal + cache integration test passed")
```

### **Test 4: Performance Overhead**
```python
def test_causal_performance():
    """Measure overhead of causal masking."""
    B, H_q, H_kv, S, D = 16, 32, 8, 512, 128
    
    q = torch.randn(B, H_q, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = attention_with_kv_cache(q, k, v, is_causal=False)
        _ = attention_with_kv_cache(q, k, v, is_causal=True)
    
    # Measure non-causal
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = attention_with_kv_cache(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    time_non_causal = time.perf_counter() - start
    
    # Measure causal
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = attention_with_kv_cache(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    time_causal = time.perf_counter() - start
    
    overhead = (time_causal / time_non_causal - 1.0) * 100
    print(f"Causal overhead: {overhead:.2f}%")
    
    # Should be <5%
    assert overhead < 5.0, f"Overhead {overhead:.2f}% exceeds 5% threshold"
    print("✅ Performance overhead test passed")
```

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Step 1: Kernel Modification (3-4 hours)**
- [ ] Add IS_CAUSAL parameter to kernel
- [ ] Implement position tracking (q_pos = seq_len_cache + offs_m)
- [ ] Add causal masking in both cache and new token loops
- [ ] Test with simple config (S=8, verify mask pattern)

### **Step 2: Integration Testing (2-3 hours)**
- [ ] Test causal + KV cache
- [ ] Test causal + GQA
- [ ] Test causal + GQA + KV cache (full integration)
- [ ] Handle edge cases (S_q=1, decode phase)

### **Step 3: Python API (1-2 hours)**
- [ ] Add is_causal parameter with default False
- [ ] Update docstrings with examples
- [ ] Add usage example in docs

### **Step 4: Comprehensive Testing (5-7 hours)**
- [ ] Implement Test 1 (correctness vs PyTorch)
- [ ] Implement Test 2 (manual mask verification)
- [ ] Implement Test 3 (causal + cache)
- [ ] Implement Test 4 (performance overhead)
- [ ] Verify all Phase 1+2 tests pass with is_causal=False

---

## 🎯 **ACCEPTANCE CRITERIA**

### **Functional**
- ✅ Passes correctness test vs PyTorch (error < 1e-3)
- ✅ Works with KV cache (Phase 1)
- ✅ Works with GQA (Phase 2)
- ✅ Works with all features combined
- ✅ Handles edge cases (S=1, very long sequences)

### **Performance**
- ✅ Overhead < 5% vs non-causal (prefill phase)
- ✅ Decode phase: No overhead (all cache in past)

### **Integration**
- ✅ All existing tests pass with is_causal=False (no regression)
- ✅ Ready for LLaMA 3.1 integration (Phase 4)

---

## 🚀 **READY TO IMPLEMENT**

**Prerequisites**: Phases 1 (KV Cache) and 2 (GQA) complete

**Next Step**: Add IS_CAUSAL parameter to existing GQA kernel

**Command for Cursor**:
```
Implement causal masking in Triton.
Add IS_CAUSAL flag to Phase 2 kernel.
Use tl.where for efficient masking.
Target: <5% overhead, exact match to PyTorch causal attention.
```

---

**Status**: ⏳ Awaiting Phase 1+2 completion  
**Estimated Time**: 10-15 hours  
**Priority**: 🔴 CRITICAL (Required by all LLMs)

