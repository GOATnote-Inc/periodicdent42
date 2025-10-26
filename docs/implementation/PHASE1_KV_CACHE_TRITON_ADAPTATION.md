# PHASE 1: KV Cache Implementation (Triton Adaptation)

**Priority**: 🔴 CRITICAL (Highest ROI)  
**Effort**: 40-50 hours  
**Implementation**: **Triton** (not CUDA)

---

## 🎯 **WHY TRITON?**

**Current FlashCore**: Already Triton-based (`flashcore/fast/attention_production.py`)

**Benefits**:
- ✅ Proven fast (<5μs already achieved)
- ✅ Faster iteration (Python DSL vs CUDA C++)
- ✅ Easier debugging
- ✅ Auto-tuning built-in
- ✅ Maintains existing codebase consistency

**Trade-off**: Less low-level control vs hand-tuned CUDA

**Decision**: **Use Triton for Phases 1-4, evaluate CUDA port later if needed**

---

## 🔧 **TRITON KERNEL SIGNATURE**

### **From CUDA (Original Spec)**:
```cuda
__global__ void dhp_attention_with_cache_kernel(
    const float* Q,           // [B, H_q, S_q, D]
    const float* K_new,       // [B, H_kv, S_q, D]
    const float* V_new,       // [B, H_kv, S_q, D]
    const float* K_cache,     // [B, H_kv, S_max, D]
    const float* V_cache,     // [B, H_kv, S_max, D]
    const int* seq_lens,      // [B]
    float* O,                 // [B, H_q, S_q, D]
    ...
)
```

### **To Triton (Adapted)**:
```python
@triton.jit
def _attention_kv_cache_fwd(
    Q, K_new, V_new,          # [B, H, S_q, D]
    K_cache, V_cache,         # [B, H_kv, S_max, D]
    seq_lens,                 # [B] - cache length per sample
    O,                        # [B, H, S_q, D]
    # Strides (Triton requires explicit)
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_knb, stride_knh, stride_knm, stride_knd,
    stride_vnb, stride_vnh, stride_vnm, stride_vnd,
    stride_kcb, stride_kch, stride_kcm, stride_kcd,
    stride_vcb, stride_vch, stride_vcm, stride_vcd,
    stride_ob, stride_oh, stride_om, stride_od,
    # Dimensions
    B: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr,
    S_q: tl.constexpr, S_max: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    # Attention parameters
    SCALE: tl.constexpr
):
    """
    Triton kernel for attention with KV cache support.
    
    Logic follows PHASE1_KV_CACHE_SPEC.md but implemented in Triton.
    """
    # Get batch, head, and position indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)
    
    # Compute KV head for GQA (Phase 2 prep)
    group_size = H_q // H_kv
    kv_head_idx = head_idx // group_size
    
    # Get sequence length for this sample
    seq_len = tl.load(seq_lens + batch_idx)
    
    # Query block positions
    offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S_q
    
    # Head dimension
    offs_d = tl.arange(0, D)
    
    # Load query block
    q_ptrs = (Q + batch_idx * stride_qb + head_idx * stride_qh +
              offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize output accumulator
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Attend to CACHED keys/values [0:seq_len)
    for k_block_start in range(0, seq_len, BLOCK_N):
        offs_n = k_block_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        # Load cached K block
        k_ptrs = (K_cache + batch_idx * stride_kcb + kv_head_idx * stride_kch +
                  offs_n[:, None] * stride_kcm + offs_d[None, :] * stride_kcd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute scores: Q @ K^T
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk *= SCALE
        
        # Online softmax update
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
        
        # Update statistics
        m_i = m_ij
        l_i = l_ij
    
    # Attend to NEW keys/values [0:S_q)
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
        
        # Online softmax update
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
        
        # Update statistics
        m_i = m_ij
        l_i = l_ij
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = (O + batch_idx * stride_ob + head_idx * stride_oh +
              offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])
```

---

## 🐍 **PYTHON WRAPPER** (Same as Original Spec)

```python
def attention_with_kv_cache(
    query: torch.Tensor,                    # [B, H_q, S_q, D]
    key: torch.Tensor,                      # [B, H_kv, S_q, D]
    value: torch.Tensor,                    # [B, H_kv, S_q, D]
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    seq_lens: Optional[torch.Tensor] = None,
    cache_max_len: int = 4096,
    update_cache: bool = True,
    is_causal: bool = False,
    num_query_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Triton-based attention with KV cache.
    
    Implementation follows PHASE1_KV_CACHE_SPEC.md logic.
    """
    B, H_q, S_q, D = query.shape
    _, H_kv, _, _ = key.shape
    
    # Handle cache
    if past_key_value is None:
        K_cache = torch.empty(B, H_kv, cache_max_len, D, device=query.device, dtype=query.dtype)
        V_cache = torch.empty(B, H_kv, cache_max_len, D, device=query.device, dtype=query.dtype)
        if seq_lens is None:
            seq_lens = torch.zeros(B, dtype=torch.int32, device=query.device)
    else:
        K_cache, V_cache = past_key_value
        if seq_lens is None:
            # Infer from cache (assume all filled)
            seq_lens = torch.full((B,), K_cache.shape[2], dtype=torch.int32, device=query.device)
    
    # Allocate output
    output = torch.empty_like(query)
    
    # Launch kernel
    grid = lambda META: (B, H_q, triton.cdiv(S_q, META['BLOCK_M']))
    
    _attention_kv_cache_fwd[grid](
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
        # Dimensions
        B, H_q, H_kv, S_q, cache_max_len, D,
        # Tuning parameters
        BLOCK_M=64,  # Auto-tune later
        BLOCK_N=64,
        SCALE=1.0 / (D ** 0.5)
    )
    
    # Update cache if requested
    if update_cache:
        # Append new K/V to cache
        for b in range(B):
            start_idx = seq_lens[b].item()
            end_idx = start_idx + S_q
            K_cache[b, :, start_idx:end_idx, :] = key[b]
            V_cache[b, :, start_idx:end_idx, :] = value[b]
            seq_lens[b] += S_q
    
    return output, (K_cache, V_cache) if update_cache else None
```

---

## ✅ **KEY DIFFERENCES FROM CUDA SPEC**

### **1. No Manual Thread Management**
- CUDA: Explicit `threadIdx`, `blockIdx`, warp-level ops
- Triton: Automatic parallelization via `tl.program_id()`

### **2. Automatic Shared Memory**
- CUDA: Manual `__shared__` allocation
- Triton: Compiler manages SMEM for register spills

### **3. Block-Level Programming**
- CUDA: Thread-level with manual synchronization
- Triton: Block-level with automatic sync

### **4. Auto-Tuning**
- CUDA: Manual tuning of block sizes
- Triton: `@triton.autotune` decorator for automatic tuning

---

## 📋 **IMPLEMENTATION CHECKLIST** (Adapted)

### **Step 1: Core Kernel (20-25 hours)**
- [ ] Implement `_attention_kv_cache_fwd` kernel in Triton
- [ ] Handle cache read (concat K_cache + K_new)
- [ ] Implement online softmax across cache + new
- [ ] Test with single config (B=1, H=8, S_q=1, S_cache=128)

### **Step 2: Python Wrapper (6-8 hours)**
- [ ] Implement `attention_with_kv_cache()` function
- [ ] Handle cache initialization
- [ ] Handle cache updates
- [ ] Add input validation
- [ ] Test with various configs

### **Step 3: Testing (12-15 hours)**
- [ ] Test correctness vs PyTorch SDPA
- [ ] Test variable sequence lengths
- [ ] Test memory leak check (1000 steps)
- [ ] Performance benchmark

### **Step 4: Optimization (8-10 hours)**
- [ ] Add `@triton.autotune` for block sizes
- [ ] Profile with Triton profiler
- [ ] Validate <10μs decode target met

---

## 🎯 **ACCEPTANCE CRITERIA** (Same as Original)

### **Functional**
- ✅ Error < 1e-4 vs PyTorch SDPA
- ✅ Variable seq_lens supported
- ✅ No memory leaks
- ✅ Cache updates correctly

### **Performance**
- ✅ Decode < 10μs (B=16, S_cache=2048, H=32, D=128)
- ✅ Memory: Cache overhead only (~2×)
- ✅ Prefill: Within 10% of baseline

---

## 🚀 **READY TO IMPLEMENT**

**Advantage Over CUDA**: Faster iteration, proven fast with existing Triton codebase

**Command for Cursor**:
```
Implement KV cache support in Triton following this adapted specification.
Extend flashcore/fast/attention_production.py with cache support.
Target: Correctness first (<1e-4 error), then optimize for <10μs decode.
```

---

**Status**: ⏳ Ready for Triton implementation  
**Estimated Time**: 40-50 hours  
**Priority**: 🔴 CRITICAL

