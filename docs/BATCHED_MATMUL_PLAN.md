# Strided-Batched Matmul Implementation Plan

## 🎯 **Objective**

Eliminate 8k+ kernel launches → **O(1) launches** via cuBLASLt strided-batched matmul.

---

## 📊 **Current Bottleneck**

```cpp
// CURRENT: Nested loops = 8,192 cuBLASLt calls!
for (int b = 0; b < B; ++b) {          // 16 batches
    for (int h = 0; h < H; ++h) {      // 16 heads
        for (int p = 0; p < num_pages; ++p) {  // 32 pages
            // Q@K^T: Launch #1
            cublasLtMatmul(...);
            
            // Softmax: Launch #2
            kernel_online_softmax<<<>>>();
            
            // P@V: Launch #3
            cublasLtMatmul(...);
        }
    }
}

Total launches: 16 × 16 × 32 × 3 = 24,576 kernel launches!
```

**Launch overhead dominates** (~80% of execution time).

---

## ✅ **Target Architecture**

```cpp
// TARGET: Single cuBLASLt call for all batches!
// Batch = (b, h, p) tuple
const int total_batches = B * H * num_pages;

// Q@K^T: ONE call for ALL batches
cublasLtMatmul(..., total_batches);  // Launch #1

// Softmax: Still per-batch (can't batch stateful operations easily)
for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
    kernel_online_softmax<<<>>>();  // Launches #2..total_batches+1
}

// P@V: ONE call for ALL batches
cublasLtMatmul(..., total_batches);  // Launch #total_batches+2

Total launches: 2 (GEMM) + total_batches (softmax) = ~8k → ~8k
```

**Wait, that doesn't help!** Softmax is still per-batch!

---

## 🔧 **Revised Strategy: Batched GEMM Only**

Actually, the **expert was right** - we need to batch the GEMMs first, then optimize softmax separately.

### **Phase 1: Batch Q@K^T and P@V**
```
Current: 16×16×32 = 8,192 Q@K^T calls + 8,192 P@V calls = 16,384 GEMM launches
Target:  1 Q@K^T call + 1 P@V call = 2 GEMM launches
Speedup: 8,192× reduction in GEMM launch overhead!
```

### **Phase 2: Keep Softmax Per-Page** (for now)
```
Softmax is stateful (online algorithm with m/l/r state).
Keep per-page for now: 8,192 softmax launches
Future: Fuse softmax into GEMM epilogue or use batched TMA
```

### **Expected Speedup**
```
GEMM overhead: ~40% of total time → eliminated!
Softmax overhead: ~10% of total time → kept (for now)
Remaining: ~50% actual compute

Result: ~1.8-2× overall speedup from batching GEMMs alone!
```

---

## 📐 **cuBLASLt Strided-Batched API**

### **Key Attributes**
```cpp
// On each matrix layout (A, B, C/D):
CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT        // Number of matrices in batch
CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET  // Bytes between consecutive matrices
```

### **Example: Batched Q@K^T**
```cpp
// Q layout: [total_batches, M, D]
//   Stride: M * D * sizeof(__half) bytes between Q[i] and Q[i+1]
int32_t batch_count = B * H * num_pages;
int64_t stride_Q = M * D * sizeof(__half);

cublasLtMatrixLayoutSetAttribute(layout_Q,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
    &batch_count, sizeof(batch_count));
    
cublasLtMatrixLayoutSetAttribute(layout_Q,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
    &stride_Q, sizeof(stride_Q));
```

---

## 🚧 **Implementation Challenges**

### **Challenge 1: Variable Page Sizes**
```
Pages have different sizes (last page might be smaller).
Solution: Pad all pages to max_page_size or use separate calls per page size.
```

### **Challenge 2: Per-Page K/V Slicing**
```
K and V are sliced per-page: K_block = K[page_start:page_end, :]
Solution: Pre-build strided K/V arrays or use gather operations.
```

### **Challenge 3: Online Softmax State**
```
Softmax state (m/l/r) is per-row, accumulated across pages.
Solution: Keep per-batch state arrays, index by (b,h,p).
```

---

## 🎯 **Simplified Implementation (MVP)**

### **Step 1: Batch All Heads (Same Page)**
Instead of batching across pages (complex), batch across **heads only** first:

```cpp
// For each page p:
//   Batch ALL B×H heads into one Q@K^T call
//   Then batch ALL B×H heads into one P@V call

const int heads_batch = B * H;  // 16 × 16 = 256

// Q layout: [heads_batch, M, D]
stride_Q = M * D * sizeof(__half);
batch_count = heads_batch;

// ONE Q@K^T call for all 256 heads
cublasLtMatmul(..., heads_batch);

// 256 softmax calls (one per head) - still fast!
for (int i = 0; i < heads_batch; ++i) {
    kernel_online_softmax<<<>>>();
}

// ONE P@V call for all 256 heads
cublasLtMatmul(..., heads_batch);

// Repeat for each page (32 iterations)
```

**Result:**
- Q@K^T: 8,192 calls → 32 calls (256× reduction!)
- P@V: 8,192 calls → 32 calls (256× reduction!)
- Softmax: 8,192 calls → 8,192 calls (unchanged)

**Expected speedup: 2-3×** (GEMM overhead eliminated, softmax kept)

---

## 🔢 **Memory Layout**

### **Current (Per-Head)**
```
Q: [B, H, S, D] → Access Q[b][h][:] for each (b,h)
K: [B, H, S, D] → Access K[b][h][:] for each (b,h)
V: [B, H, S, D] → Access V[b][h][:] for each (b,h)
O: [B, H, S, D] → Write O[b][h][:] for each (b,h)
```

**Stride between heads:** `S * D * sizeof(__half)`

### **Batched (All Heads)**
```
Q_batch: pointer to Q[0][0], stride = S*D*sizeof(half)
K_batch: pointer to K[0][0], stride = S*D*sizeof(half)
O_batch: pointer to O[0][0], stride = S*D*sizeof(half)

batch_count = B * H
```

cuBLASLt will handle: `Q_batch[i] = Q_batch + i * stride`

---

## 📝 **Implementation Code (MVP)**

```cpp
// BATCH ALL HEADS for current page
const int batch_count = B * H;
const int64_t stride_Q = S * D * sizeof(__half);
const int64_t stride_K = S * D * sizeof(__half);
const int64_t stride_V = S * D * sizeof(__half);
const int64_t stride_O = S * D * sizeof(__half);

// Set batch attributes on layouts (once per page)
cublasLtMatrixLayoutSetAttribute(layout_Q,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
cublasLtMatrixLayoutSetAttribute(layout_Q,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_Q, sizeof(stride_Q));

// Same for K, V, O layouts...

// Process pages
for (int p = 0; p < num_pages; ++p) {
    // Slice K/V for this page
    const __half* K_page = K_ptr + p * page_size * D;
    const __half* V_page = V_ptr + p * page_size * D;
    
    // ONE Q@K^T call for ALL B×H heads!
    cublasLtMatmul(g_cublaslt_handle, g_desc_qk,
                   &alpha, Q_ptr, layout_Q,      // Batched Q
                   K_page, layout_Kb,            // Single K_page (broadcast)
                   &beta, S_batch, layout_Sb,    // Batched S
                   S_batch, layout_Sb,
                   &algo, workspace, ws_size, stream);
    
    // Softmax per-head (still 256 calls per page)
    for (int bh = 0; bh < batch_count; ++bh) {
        kernel_online_softmax<<<>>>(S_batch + bh * M * page_cols, ...);
    }
    
    // ONE P@V call for ALL B×H heads!
    cublasLtMatmul(g_cublaslt_handle, g_desc_pv,
                   &alpha, P_batch, layout_Pb,   // Batched P
                   V_page, layout_Vb,            // Single V_page (broadcast)
                   &beta, O_ptr, layout_O,       // Batched O (accumulate)
                   O_ptr, layout_O,
                   &algo, workspace, ws_size, stream);
}
```

---

## 🎯 **Next Steps**

1. **Implement MVP** (batch heads only): 2-3 hours
2. **Test & validate**: 1 hour
3. **Profile with NCU**: 1 hour
4. **Optimize further** (batch pages too): 2-3 hours

**Total: 6-9 hours to full implementation**

**Expected final speedup: 5-10× with proper batching!**

