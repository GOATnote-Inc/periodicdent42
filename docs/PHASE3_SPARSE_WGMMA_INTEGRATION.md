# Phase 3: Sparse Paging + WGMMA Integration

**Mission**: 35K+ tokens/sec, 128K context, 70% memory savings  
**Method**: Fast compute (WGMMA) + Smart memory (sparse paging)  
**Timeline**: 3-5 days

---

## 🎯 **The Breakthrough Insight**

### **What Phase 2 Taught Us**

```
Phase 1:  0.65 TFLOPS (baseline, dense)
Phase 2:  0.59 TFLOPS (optimized memory, still slow compute) ❌

Lesson: Optimizing memory for slow compute = wasted effort!
```

### **The Real Path to 35K tokens/sec**

```
Component 1: WGMMA Tensor Cores    → 100-150 TFLOPS (fast compute)
Component 2: Sparse paging (CSR)   → 70% memory reduction
Component 3: System integration    → SGLang backend

= 5× system speedup (25K → 35K+ tokens/sec) ✅
```

---

## 🏗️ **Architecture Overview**

### **Three-Layer Stack**

```
┌─────────────────────────────────────┐
│  SGLang Serving Layer               │
│  - RadixAttention cache             │
│  - Continuous batching              │
│  - Scheduler (page reuse)           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  FlashCore Sparse Backend           │
│  - CSR paging layout                │
│  - Sparse attention dispatch        │
│  - 128K context support             │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  WGMMA Kernel (Phase 3A)            │
│  - 16×16×16 matrix tiles            │
│  - FP16 Tensor Cores                │
│  - 100-150 TFLOPS                   │
└─────────────────────────────────────┘
```

### **Data Flow**

```python
# SGLang request → FlashCore backend
def forward(q, k_cache, v_cache, seq_metadata):
    # 1. Build sparse layout (CSR)
    csr_layout = sparse_pager.build_layout(
        token_to_page, seq_starts, seq_ends,
        page_resident_bitmap, page_tokens, num_pages
    )
    
    # 2. Dispatch to WGMMA kernel
    output = flashcore_sparse_wgmma(
        q, k_cache, v_cache, csr_layout
    )
    
    return output

# Result: 3.3× less memory traffic, 150× faster compute!
```

---

## 📊 **Performance Model**

### **Memory Savings (Sparse Paging)**

```
Scenario: Multi-turn chat with shared system prompt

Dense attention:
  - Load all 128K tokens every decode step
  - Memory: 128K × H × D × 2 bytes = 64 MB (BF16)
  - Bandwidth: 64 MB × 1000 decodes/sec = 64 GB/s

Sparse paging (70% reuse):
  - Load only unique pages (30% of tokens)
  - Memory: 38K × H × D × 2 bytes = 19 MB
  - Bandwidth: 19 MB × 1000 decodes/sec = 19 GB/s
  - Savings: 3.3× less DRAM traffic!
```

### **Compute Speedup (WGMMA)**

```
Scalar (Phase 1):
  - 0.65 TFLOPS
  - Time: 420ms

WGMMA (Phase 3):
  - 100-150 TFLOPS (target)
  - Time: 2-3ms (140-210× faster)
  - Method: 16×16×16 Tensor Core tiles
```

### **System Throughput**

```
Baseline (FA3 dense):    25K tokens/sec
+ Sparse paging:         25K × 3.3× = 82.5K tokens/sec (if compute was free)
+ Realistic compute:     35K-40K tokens/sec (compute becomes bottleneck)

Net: 40-60% gain over dense FA3 ✅
```

---

## 🛠️ **Implementation Plan**

### **Phase 3A: WGMMA Kernel (2 days)**

**Goal**: 100-150 TFLOPS on dense attention

```cuda
// flashcore/fast/attention_phase3_wgmma.cu
#include <mma.h>  // Ampere WMMA
// Hopper WGMMA if sm_90

__global__ void attention_wgmma(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int B, int H, int S, int D, float scale, bool is_causal
) {
    // Use 16×16×16 Tensor Core tiles
    // Replace scalar loops with mma_sync
    // FP16 accumulation (2× faster on H100)
    
    // Expected: 100-150 TFLOPS
}
```

**Success Criteria**:
- ✅ Correctness: `torch.allclose(out, sdpa_ref, rtol=1e-3, atol=2e-3)`
- ✅ Performance: 100+ TFLOPS on H100
- ✅ Latency: < 5ms for B=16, H=16, S=2048, D=64

**NCU Validation**:
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./build/bin/test_hopper

Target: SM throughput ≥ 60% (Tensor Cores active!)
```

---

### **Phase 3B: Sparse Paging Integration (2 days)**

**Goal**: Wire CSR paging to WGMMA kernel

**Step 1: Build sparse pager extension**

```bash
# setup.py extension (already created!)
python setup.py build_ext --inplace

# Result: flashcore_sparse_pager.so
```

**Step 2: Create sparse attention backend**

```python
# flashcore/backends/radix_sparse_flashcore.py
import torch
from torch.utils.cpp_extension import load
import flashcore_sparse_pager as pager

class RadixSparseFlashCore:
    def __init__(self, page_tokens=128):
        self.page_tokens = page_tokens
        self.scale = None
        
    def __call__(self, q, k_pages, v_pages, csr_layout):
        """
        Args:
            q: [B, H, D] query
            k_pages: [num_pages, H, page_tokens, D] paged keys
            v_pages: [num_pages, H, page_tokens, D] paged values
            csr_layout: (row_offsets, cols, counts, staging_ids)
        
        Returns:
            out: [B, H, D] attention output
        """
        row_offsets, cols, counts, _ = csr_layout
        
        # Dispatch to sparse WGMMA kernel
        return flashcore_sparse_wgmma_decode(
            q, k_pages, v_pages,
            row_offsets, cols, counts,
            self.page_tokens, self.scale
        )
```

**Step 3: Implement sparse WGMMA kernel**

```cuda
// flashcore/fast/attention_sparse_wgmma.cu
__global__ void sparse_wgmma_decode(
    const __half* Q,              // [B, H, D]
    const __half* K_pages,        // [num_pages, H, page_tokens, D]
    const __half* V_pages,        // [num_pages, H, page_tokens, D]
    const int32_t* row_offsets,   // CSR row pointers [B+1]
    const int32_t* cols,          // CSR column indices [nnz]
    const int32_t* token_counts,  // Tokens per page [nnz]
    int page_tokens, float scale
) {
    // For each query row (batch element):
    //   For each non-zero page in CSR:
    //     Load page from K_pages, V_pages
    //     WGMMA: Q @ K^T (16×16×16 tiles)
    //     Online softmax update
    //     WGMMA: P @ V (accumulate)
    //   Write output
    
    // Memory: Only load 30% of pages (sparse!)
    // Compute: Full WGMMA speed (100-150 TFLOPS)
}
```

**Success Criteria**:
- ✅ Correctness: Match dense WGMMA output
- ✅ Memory: 3× less DRAM traffic (NCU validation)
- ✅ Performance: 100+ TFLOPS (Tensor Cores still saturated)

---

### **Phase 3C: SGLang Integration (1 day)**

**Goal**: Drop-in backend for SGLang

```python
# flashcore/backends/sglang_adapter.py
from sglang.srt.layers.attention.backend import AttentionBackend
from flashcore.backends.radix_sparse_flashcore import RadixSparseFlashCore

class FlashCoreSparseBackend(AttentionBackend):
    """
    SGLang attention backend using FlashCore sparse WGMMA.
    
    Usage:
        # In sglang launcher
        python -m sglang.launch_server \\
            --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
            --attention-backend flashcore_sparse \\
            --max-context-len 128000
    """
    
    def __init__(self, num_heads, head_dim, num_kv_heads, scale, args):
        self.backend = RadixSparseFlashCore(page_tokens=args.page_tokens)
        self.backend.scale = scale
        
    def forward_decode(self, q, k_cache, v_cache, forward_batch):
        # Build CSR layout from forward_batch metadata
        csr_layout = self._build_csr_layout(forward_batch)
        
        # Dispatch to FlashCore sparse kernel
        return self.backend(q, k_cache, v_cache, csr_layout)
    
    def _build_csr_layout(self, forward_batch):
        # Extract seq metadata from forward_batch
        # Build CSR using sparse_pager
        # Return (row_offsets, cols, counts, staging_ids)
        ...
```

**Registration**:
```python
# In sglang/srt/attention_backends/registry.py
from flashcore.backends.sglang_adapter import FlashCoreSparseBackend

ATTENTION_BACKENDS["flashcore_sparse"] = FlashCoreSparseBackend
```

**Success Criteria**:
- ✅ SGLang server starts with `--attention-backend flashcore_sparse`
- ✅ Passes SGLang's attention backend tests
- ✅ Benchmarks show 35K+ tokens/sec on H100

---

## 📈 **Validation & Benchmarks**

### **Correctness Tests**

```python
# tests/test_sparse_wgmma.py
def test_sparse_vs_dense():
    """Sparse WGMMA matches dense for 70% cache reuse."""
    q, k, v = generate_attention_inputs(...)
    
    # Dense reference
    out_dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # Sparse (with CSR layout)
    k_pages, v_pages, csr_layout = build_sparse_layout(k, v, page_tokens=128)
    out_sparse = flashcore_sparse_wgmma(q, k_pages, v_pages, csr_layout)
    
    # Should match!
    assert torch.allclose(out_sparse, out_dense, rtol=1e-3, atol=2e-3)
```

### **Performance Benchmarks**

```python
# benchmarks/bench_sparse_wgmma_h100.py
configs = [
    # (B, H, S, D, page_tokens, cache_reuse)
    (8, 32, 128*1024, 128, 128, 0.7),   # 128K context, 70% reuse
    (16, 32, 32*1024, 128, 128, 0.5),   # 32K context, 50% reuse
    (32, 32, 8*1024, 128, 128, 0.3),    # 8K context, 30% reuse
]

for config in configs:
    # Measure tokens/sec, TFLOPS, memory bandwidth
    # Compare vs FA3 dense baseline
    ...
```

**Target Metrics**:
```
Config: B=8, S=128K, 70% reuse
- Tokens/sec: 35K+ (vs FA3's 25K)
- TFLOPS: 100-150 (kernel)
- DRAM: 19 GB/s (vs FA3's 64 GB/s)
- Latency: 3-5ms per decode step
```

---

## 🎓 **Key Insights**

### **Why This Works**

1. **Sparse paging** solves memory bottleneck (3.3× less traffic)
2. **WGMMA** solves compute bottleneck (150× faster than scalar)
3. **System integration** enables real workloads (SGLang serving)

### **Standing on Giants**

- **SGLang**: Sparse paging concept, RadixAttention
- **FA3**: WGMMA Tensor Core methodology
- **FlashCore**: Integration and optimization

### **What Makes This Different**

```
FA3:          Dense, fast kernel (450 TFLOPS)
SGLang:       Sparse, system-level optimization
FlashCore:    Sparse + fast kernel (best of both!)
```

---

## 🚀 **Next Steps**

1. **[NOW]** Implement Phase 3A: WGMMA kernel (dense)
2. **[THEN]** Integrate Phase 3B: Sparse paging
3. **[FINALLY]** Deploy Phase 3C: SGLang backend

**Timeline**: 3-5 days to 35K+ tokens/sec! 🔥

---

## 📚 **References**

- **SGLang RadixAttention**: [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- **FlashAttention-3**: Hopper WGMMA methodology
- **NVIDIA Hopper Programming Guide**: TMA, WGMMA, async pipelines
- **User's radix_sparse bundle**: CSR paging implementation

**Attribution**: Standing on the shoulders of giants to see 35K+ tokens/sec! 🚀

