# CUTLASS FMHA Baseline - H100 Performance (Oct 31, 2025)

## ✅ Established: CUTLASS Example 88 Working

**Hardware:** H100 80GB HBM3 (sm_90a)  
**Version:** CUTLASS 4.3.0 (main branch)  
**Build:** Release, `-arch=sm_90a`

---

## 📊 Performance Results

### CUTLASS FMHA (Dense Attention)

| Sequence | Batch | Heads | Dim | TFLOPS (cooperative) | TFLOPS (ping-pong) |
| :------- | :---- | :---- | :-- | :------------------- | :----------------- |
| 1024     | 1     | 8     | 64  | 560.4                | 605.6              |
| 2048     | 1     | 8     | 64  | 560.2                | 602.9              |
| 4096     | 1     | 8     | 64  | 558.3                | 603.0              |
| 8192     | 1     | 8     | 64  | 559.2                | 604.3              |
| 16384    | 1     | 8     | 64  | 559.6                | 602.7              |

**Key observations:**
- ✅ Consistent ~600 TFLOPS across sequence lengths
- ✅ Ping-pong schedule slightly faster than cooperative
- ✅ No degradation up to S=16K

### Our BSR GEMM Baseline

| Config          | TFLOPS | vs. CUTLASS |
| :-------------- | :----- | :---------- |
| M=8K, topk=16   | 111.0  | **5.4× slower** |

---

## 🎯 Gap Analysis

### What CUTLASS Has (That We Don't)

```
✅ Full attention pipeline (Q@K^T + softmax + P@V fused)
✅ TMA (Tensor Memory Accelerator)
✅ Warp specialization (producer/consumer)
✅ Ping-pong vs. cooperative schedules
✅ Online softmax (memory-efficient)
✅ Optimized for Hopper architecture
✅ 600 TFLOPS sustained performance
```

### What We Have

```
✅ BSR sparse GEMM (C = A @ B)
✅ WMMA Tensor Cores
✅ Cooperative loads
✅ Working baseline (111 TFLOPS)
❌ No attention semantics
❌ No TMA
❌ No warp specialization
❌ No softmax
```

---

## 🚀 Path Forward: Adapt CUTLASS for Sparse

### Option 1: Extend Example 88 with Sparse Support (4-6 weeks)

**Approach:** Modify CUTLASS FMHA collective for BSR sparse patterns

```cpp
// CUTLASS uses CollectiveBuilder
// Located in: cutlass/gemm/collective/collective_builder.hpp

// Current (dense):
for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
    load_K_tile(k_tile);
    load_V_tile(k_tile);
    compute_attention();
}

// Sparse adaptation:
for (int sparse_idx = 0; sparse_idx < topk; sparse_idx++) {
    int k_tile = block_col_idx[q_tile][sparse_idx];
    load_K_tile(k_tile);  // Sparse indexed load
    load_V_tile(k_tile);
    compute_attention();
}
```

**Implementation:**
1. Week 1-2: Study CUTLASS collective builder
2. Week 3-4: Add sparse indexing to TMA loads
3. Week 5-6: Validate correctness + optimize

**Expected result:** Sparse attention at 400-500 TFLOPS

### Option 2: Build Attention Layer on Our GEMM (2-3 weeks)

**Approach:** Add softmax + fusion to our existing kernel

```cpp
// Phase 1: Add softmax kernel (separate)
__global__ void online_softmax(float* QK, float* P, int S);

// Phase 2: Fuse Q@K^T + softmax
__global__ void qk_softmax_fused(...);

// Phase 3: Full pipeline Q@K^T + softmax + P@V
__global__ void sparse_attention_fused(...);
```

**Timeline:**
- Week 1: Softmax kernel + validation
- Week 2: Fuse Q@K^T + softmax
- Week 3: Complete pipeline + optimize

**Expected result:** 200-300 TFLOPS (without TMA yet)

### Option 3: Hybrid - Use CUTLASS Primitives (Recommended, 2-4 weeks)

**Approach:** Use CUTLASS CuTe for TMA, add our sparse logic

```cpp
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>

// Use CUTLASS TMA for loads
using TMA_Q = SM90_TMA_LOAD;
using TMA_K = SM90_TMA_LOAD_MULTICAST;

// Our sparse indexing
for (int i = 0; i < topk; i++) {
    int kv_block = sparse_idx[i];
    cute::copy(TMA_K{}, K_gmem[kv_block], K_smem);
    // ... compute ...
}
```

**Timeline:**
- Week 1: Integrate CuTe TMA (learn CUTLASS APIs)
- Week 2: Add softmax + online reduction
- Week 3: Fuse full pipeline
- Week 4: Optimize + validate

**Expected result:** 400-500 TFLOPS with sparse patterns

---

## 📊 Performance Targets

### Realistic (Option 3, 4 weeks)

```
Sparse Attention (topk=32):
  TFLOPS:      400-500 (vs. CUTLASS dense 600)
  Latency:     ~40 μs @ S=8K
  vs. Baseline: 3.6-4.5× improvement
  vs. CUTLASS:  Still slower, but sparse benefit at S>32K
```

### Stretch (Option 1, 6 weeks)

```
Full CUTLASS Integration:
  TFLOPS:      500-550 (sparse patterns)
  Latency:     ~30 μs @ S=8K, ~200 μs @ S=131K sparse
  vs. Dense:    500× faster at S=131K (due to sparsity)
```

---

## 🎯 Immediate Actions

### Today (2 hours)

1. **Study CUTLASS builder patterns**
   ```bash
   cd /opt/cutlass/include/cutlass/gemm/collective
   grep -r "CollectiveBuilder" .
   ```

2. **Test CUTLASS at longer sequences**
   ```bash
   ./88_hopper_fmha -b 1 -h 8 -q 32768 -k 32768 -d 64
   # Measure latency at S=32K
   ```

3. **Extract key CuTe patterns**
   - How does TMA work?
   - How is softmax implemented?
   - How are Q@K^T and P@V fused?

### Tomorrow (8 hours)

1. **Create sparse attention skeleton**
   - Copy our BSR GEMM kernel
   - Add softmax stub
   - Add pipeline structure

2. **Integrate first CuTe primitive**
   - Use SM90_TMA_LOAD for Q
   - Validate correctness
   - Measure improvement

3. **Document learning**
   - What works
   - What's blocked
   - Performance deltas

---

## 📦 Deliverables Status

✅ **CUTLASS Example 88 built and benchmarked**  
✅ **Baseline: 600 TFLOPS dense attention**  
✅ **Source code downloaded for analysis**  
✅ **Performance gap quantified: 5.4×**  
✅ **Three implementation paths defined**  
⬜ Sparse attention prototype (Option 3 target)  
⬜ CuTe TMA integration  
⬜ Softmax + fusion  

---

## 🚀 Recommendation

**Execute Option 3 (Hybrid approach):**

- **Week 1:** Integrate CuTe TMA into our kernel (learn by doing)
- **Week 2:** Add online softmax (reference: Example 88)
- **Week 3:** Fuse Q@K^T + softmax + P@V
- **Week 4:** Optimize + validate correctness

**Target:** 400 TFLOPS sparse attention (3.6× our baseline, competitive for sparse patterns)

**Advantage over dense:** 100-500× faster at S>32K due to sparse structure

---

**Status:** CUTLASS baseline established. Ready to adapt for sparse patterns using their tools.

