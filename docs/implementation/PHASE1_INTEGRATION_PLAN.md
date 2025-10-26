# Phase 1: KV Cache Integration Plan

**Current Code Analysis**: `flashcore/fast/attention_production.py`  
**Status**: Clean baseline, ready for extension  
**Approach**: Extend, don't replace (preserve existing functionality)

---

## 📐 **CURRENT CODE STRUCTURE**

### **What We Have** ✅
```python
# Current kernel signature
@triton.jit
def _attention_fwd_kernel(
    Q, K, V, Out,
    # Strides (all tensors same shape [B, H, N, D])
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Dimensions (all same)
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    # Current logic:
    # - Process BLOCK_M queries at a time
    # - Iterate through all N keys/values
    # - Online softmax accumulation
    # - No cache support
```

### **Key Features** ✅
- Online softmax (FlashAttention-style)
- Proven performance (<5μs with batching)
- Clean Triton code
- Auto-tuning support

### **Limitations** ❌
- No KV cache (recomputes full attention every time)
- No GQA support (H_q must equal H_k)
- No causal masking
- Fixed D=64 (hardcoded assertion)

---

## 🎯 **PHASE 1 GOAL**

**Add KV cache support while preserving existing functionality**

### **New Functionality**:
```python
# NEW: Cache-aware kernel
@triton.jit
def _attention_kv_cache_fwd_kernel(
    Q, K_new, V_new,          # New tokens [B, H, S_q, D]
    K_cache, V_cache,         # Cached tokens [B, H, S_max, D]
    seq_lens,                 # Valid cache length per batch [B]
    Out,                      # Output [B, H, S_q, D]
    # ... strides ...
    # ... dimensions ...
    # Key changes:
    # - S_q: new token count (typically 1 for decode, variable for prefill)
    # - S_max: maximum cache capacity
    # - Iterate: cache [0:seq_lens[b]] + new [0:S_q]
):
    pass  # To implement
```

### **Backward Compatibility**:
```python
# Keep existing function
def attention(q, k, v, ...):
    # Original API - no breaking changes
    pass

# NEW: Extended function
def attention_with_kv_cache(
    q, k, v,
    past_key_value=None,  # NEW: optional cache
    seq_lens=None,         # NEW: per-batch cache lengths
    update_cache=True,     # NEW: whether to return updated cache
    ...
):
    if past_key_value is None:
        # First call: use original kernel
        return attention(q, k, v, ...), None
    else:
        # Subsequent calls: use cache-aware kernel
        return ..., updated_cache
```

---

## 📋 **IMPLEMENTATION STEPS**

### **Step 1: Create New Kernel (15-20 hours)**

**File**: `flashcore/fast/attention_production.py` (extend existing file)

**Task 1.1: Add new kernel function** (2h)
```python
@triton.jit
def _attention_kv_cache_fwd_kernel(
    Q, K_new, V_new,
    K_cache, V_cache, seq_lens, Out,
    # Strides for Q, K_new, V_new
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_knb, stride_knh, stride_knm, stride_knd,
    stride_vnb, stride_vnh, stride_vnm, stride_vnd,
    # Strides for K_cache, V_cache
    stride_kcb, stride_kch, stride_kcm, stride_kcd,
    stride_vcb, stride_vch, stride_vcm, stride_vcd,
    # Strides for Out
    stride_ob, stride_oh, stride_om, stride_od,
    # Dimensions
    B: tl.constexpr, H: tl.constexpr,
    S_q: tl.constexpr, S_max: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    """Extended kernel with KV cache support"""
    # Program IDs
    pid_m = tl.program_id(0)  # Query block index
    pid_bh = tl.program_id(1)  # Batch × Head index
    
    # Decode batch and head
    b = pid_bh // H
    h = pid_bh % H
    
    # Load sequence length for this batch
    seq_len_cache = tl.load(seq_lens + b)
    
    # Query offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    
    # Load Q block [BLOCK_M, D]
    q_ptrs = (Q + b * stride_qb + h * stride_qh +
              offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S_q, other=0.0)
    
    # Initialize accumulators (same as original)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # STEP A: Process cached K/V [0:seq_len_cache)
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(0, seq_len_cache, BLOCK_N):
        offs_n_cur = start_n + offs_n
        mask_n = offs_n_cur < seq_len_cache
        
        # Load K_cache block [D, BLOCK_N]
        k_ptrs = (K_cache + b * stride_kcb + h * stride_kch +
                  offs_n_cur[None, :] * stride_kcm + offs_d[:, None] * stride_kcd)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Load V_cache block [BLOCK_N, D]
        v_ptrs = (V_cache + b * stride_vcb + h * stride_vch +
                  offs_n_cur[:, None] * stride_vcm + offs_d[None, :] * stride_vcd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute scores (same as original)
        qk = tl.dot(q, k)
        qk *= SCALE
        
        # Online softmax update (same as original)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        l_i = alpha * l_i + tl.sum(p, 1)
        m_i = m_ij
    
    # STEP B: Process new K/V [0:S_q)
    for start_n in range(0, S_q, BLOCK_N):
        offs_n_cur = start_n + offs_n
        mask_n = offs_n_cur < S_q
        
        # Load K_new block [D, BLOCK_N]
        k_ptrs = (K_new + b * stride_knb + h * stride_knh +
                  offs_n_cur[None, :] * stride_knm + offs_d[:, None] * stride_knd)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Load V_new block [BLOCK_N, D]
        v_ptrs = (V_new + b * stride_vnb + h * stride_vnh +
                  offs_n_cur[:, None] * stride_vnm + offs_d[None, :] * stride_vnd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute scores (same logic)
        qk = tl.dot(q, k)
        qk *= SCALE
        
        # Online softmax update (same logic)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        l_i = alpha * l_i + tl.sum(p, 1)
        m_i = m_ij
    
    # Final normalization (same as original)
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = (Out + b * stride_ob + h * stride_oh +
              offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < S_q)
```

**Task 1.2: Add Python wrapper** (4h)
```python
def attention_with_kv_cache(
    query: torch.Tensor,                    # [B, H, S_q, D]
    key: torch.Tensor,                      # [B, H, S_q, D]
    value: torch.Tensor,                    # [B, H, S_q, D]
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    seq_lens: Optional[torch.Tensor] = None,
    cache_max_len: int = 4096,
    update_cache: bool = True,
    block_m: int = 64,
    block_n: int = 64,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Attention with KV cache support for incremental inference
    
    Args:
        query: Query tensor [B, H, S_q, D]
        key: Key tensor [B, H, S_q, D] (new keys to cache)
        value: Value tensor [B, H, S_q, D] (new values to cache)
        past_key_value: Optional (K_cache, V_cache) from previous step
        seq_lens: Optional [B] tensor with cache length per batch
        cache_max_len: Maximum cache capacity
        update_cache: Whether to return updated cache
        block_m, block_n: Block sizes for tuning
        scale: Attention scale (default: 1/sqrt(D))
    
    Returns:
        output: [B, H, S_q, D]
        cache: (K_cache, V_cache) if update_cache else None
    """
    B, H, S_q, D = query.shape
    
    # Input validation
    assert query.is_cuda and key.is_cuda and value.is_cuda
    assert query.shape == key.shape == value.shape
    
    # Default scale
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    
    # Handle cache initialization
    if past_key_value is None:
        # First call: no cache yet
        K_cache = torch.empty(B, H, cache_max_len, D, device=query.device, dtype=query.dtype)
        V_cache = torch.empty(B, H, cache_max_len, D, device=query.device, dtype=query.dtype)
        if seq_lens is None:
            seq_lens = torch.zeros(B, dtype=torch.int32, device=query.device)
    else:
        K_cache, V_cache = past_key_value
        if seq_lens is None:
            # Infer from cache shape (assume full)
            seq_lens = torch.full((B,), K_cache.shape[2], dtype=torch.int32, device=query.device)
    
    # Allocate output
    output = torch.empty_like(query)
    
    # Launch kernel
    grid = (triton.cdiv(S_q, block_m), B * H)
    
    _attention_kv_cache_fwd_kernel[grid](
        query, key, value,
        K_cache, V_cache, seq_lens, output,
        # Query strides
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        # Key strides
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        # Value strides
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        # Cache strides
        K_cache.stride(0), K_cache.stride(1), K_cache.stride(2), K_cache.stride(3),
        V_cache.stride(0), V_cache.stride(1), V_cache.stride(2), V_cache.stride(3),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Dimensions
        B, H, S_q, cache_max_len, D,
        # Block sizes
        BLOCK_M=block_m, BLOCK_N=block_n,
        # Scale
        SCALE=scale
    )
    
    # Update cache if requested
    if update_cache:
        # Append new K/V to cache
        for b in range(B):
            start_idx = seq_lens[b].item()
            end_idx = start_idx + S_q
            if end_idx <= cache_max_len:
                K_cache[b, :, start_idx:end_idx, :] = key[b]
                V_cache[b, :, start_idx:end_idx, :] = value[b]
                seq_lens[b] += S_q
            else:
                # Cache overflow - need eviction strategy
                # For MVP: just raise error
                raise RuntimeError(f"Cache overflow: {end_idx} > {cache_max_len}")
        
        return output, (K_cache, V_cache)
    else:
        return output, None
```

---

### **Step 2: Add Tests** (12-15 hours)

**File**: `tests/test_kv_cache_correctness.py` (new file)

**Test 1: Basic correctness** (3h)
```python
import torch
import torch.nn.functional as F
from flashcore.fast.attention_production import attention_with_kv_cache

def test_kv_cache_vs_pytorch():
    """Compare KV cache implementation to PyTorch SDPA"""
    B, H, S_prefill, S_decode, D = 2, 8, 128, 10, 64
    
    torch.manual_seed(42)
    
    # Create full sequence
    q_full = torch.randn(B, H, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    k_full = torch.randn(B, H, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    v_full = torch.randn(B, H, S_prefill + S_decode, D, device='cuda', dtype=torch.float16)
    
    # PyTorch reference (full attention)
    expected = F.scaled_dot_product_attention(q_full, k_full, v_full)
    
    # Our implementation with cache
    # Step 1: Prefill
    q_prefill = q_full[:, :, :S_prefill, :]
    k_prefill = k_full[:, :, :S_prefill, :]
    v_prefill = v_full[:, :, :S_prefill, :]
    
    output_prefill, cache = attention_with_kv_cache(
        q_prefill, k_prefill, v_prefill, update_cache=True
    )
    
    # Step 2: Decode (one token at a time)
    outputs = [output_prefill]
    for t in range(S_decode):
        q_t = q_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        k_t = k_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        v_t = v_full[:, :, S_prefill + t:S_prefill + t + 1, :]
        
        output_t, cache = attention_with_kv_cache(
            q_t, k_t, v_t, past_key_value=cache, update_cache=True
        )
        outputs.append(output_t)
    
    # Concatenate results
    result = torch.cat(outputs, dim=2)
    
    # Compare
    max_diff = (result - expected).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3), \
        f"KV cache output differs from reference: max_diff={max_diff}"
    
    print("✅ KV cache correctness test passed")
```

**Test 2: Variable sequence lengths** (3h)
**Test 3: Memory leak check** (2h)
**Test 4: Performance benchmark** (4h)

---

### **Step 3: Optimize & Validate** (8-10 hours)

**Task 3.1: Profile performance** (3h)
- Measure decode latency (target: <10μs for B=16, S_cache=2048)
- Compare to PyTorch SDPA with cache
- Identify bottlenecks with Triton profiler

**Task 3.2: Optimize if needed** (4h)
- Tune block sizes for cache + new token case
- Optimize memory access patterns
- Reduce register pressure if needed

**Task 3.3: Final validation** (3h)
- Run all tests
- Verify no memory leaks (1000+ decode steps)
- Confirm <10μs target met

---

## 🎯 **EXPECTED RESULTS**

### **After Phase 1**:

✅ **Functionality**:
- KV cache working correctly
- Incremental inference enabled
- Backward compatible (original `attention()` still works)

✅ **Performance**:
- Decode <10μs (B=16, S_cache=2048, H=8, D=64)
- Competitive with PyTorch SDPA
- Memory efficient (cache overhead only)

✅ **Quality**:
- Error <1e-3 vs PyTorch SDPA
- No memory leaks
- Comprehensive tests (>90% coverage)

---

## 📋 **NEXT STEPS AFTER PHASE 1**

Once Phase 1 is complete and validated:

1. **Proceed to Phase 2** (GQA support)
   - Extend kernel to support H_q != H_kv
   - Modify cache to use H_kv dimensions
   - 4× memory savings

2. **Or pause for validation**
   - Deploy Phase 1 to real use case
   - Gather user feedback
   - Iterate if needed

---

## ✅ **READY TO BEGIN**

**Status**: Integration plan complete  
**Next Action**: Implement Task 1.1 (new kernel function)  
**Estimated Time**: 40-50 hours total  
**Target**: <10μs decode latency, <1e-3 error

**GO BUILD! 🚀**

