# Phase 1: WMMA Tensor Core Implementation Guide

**Goal**: Implement WMMA for Q·K^T and P·V to achieve 10× speedup  
**Target**: ~150 µs (from 1500 µs baseline)  
**Estimated Time**: 30-40 hours  
**Prerequisites**: Phase 0 complete, GPU access (L4 or similar)

---

## 📐 Algorithm Overview

### Current Baseline (Scalar)
```
For each query row q_i:
    For each K/V tile:
        s = q_i @ k_tile^T (scalar dot products)  ← Replace with WMMA
        p = softmax(s) (FP32)
        o += p @ v_tile (scalar accumulation)     ← Replace with WMMA
```

### Phase 1 Target (WMMA)
```
For each Q tile (32×64):
    For each K/V tile:
        S = Q_tile @ K_tile^T (WMMA 16×16×16)  ✅ Tensor Cores
        P = softmax(S) (FP32, keep as-is)
        O += P @ V_tile (WMMA 16×16×16)        ✅ Tensor Cores
```

**Key Insight**: Use Tensor Cores for matrix multiplications, keep FP32 for softmax (numerical stability).

---

## 🔧 Implementation Steps

### Step 1: Copy Baseline and Add WMMA Includes

**Create** `kernels/flashcore_wmma.cu`:

```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>  // ← Add this
#include <float.h>

using namespace nvcuda;  // ← Add this

// Rest of file...
```

### Step 2: Define WMMA Tile Sizes

```cuda
// WMMA configuration
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block sizes (must be multiples of WMMA sizes)
#define TILE_M 32    // Q rows per block (2×WMMA_M)
#define TILE_N 32    // K rows per tile (2×WMMA_N)
#define HEAD_DIM 64  // D (4×WMMA_K)

// Thread block size
#define NUM_WARPS 4
#define THREADS_PER_BLOCK (NUM_WARPS * 32)
```

### Step 3: Implement Q @ K^T with WMMA

**Pseudocode**:
```cuda
__global__ void flash_attention_wmma_kernel(
    const half* Q, const half* K, const half* V, half* O,
    float scale, int B, int H, int S
) {
    // Shared memory for tiles
    __shared__ half Q_smem[TILE_M][HEAD_DIM];
    __shared__ half K_smem[TILE_N][HEAD_DIM];
    __shared__ half V_smem[TILE_N][HEAD_DIM];
    __shared__ float S_smem[TILE_M][TILE_N];  // Attention scores (FP32)
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Load Q tile (collaborative)
    // ...
    
    // Loop over K/V tiles
    for (int tile_n = 0; tile_n < (S + TILE_N - 1) / TILE_N; tile_n++) {
        // Load K tile
        // ...
        __syncthreads();
        
        // ========================================
        // WMMA: Q @ K^T
        // ========================================
        
        // Each warp handles one 16×16 output tile
        // Warp 0: S[0:16, 0:16]
        // Warp 1: S[0:16, 16:32]
        // Warp 2: S[16:32, 0:16]
        // Warp 3: S[16:32, 16:32]
        
        int warp_m = warp_id / 2;  // 0 or 1
        int warp_n = warp_id % 2;  // 0 or 1
        
        // WMMA fragments
        wmma::fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        wmma::fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
        wmma::fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        
        // Initialize accumulator
        wmma::fill_fragment(acc_frag, 0.0f);
        
        // Loop over K dimension (D=64 → 4 tiles of WMMA_K=16)
        #pragma unroll
        for (int k = 0; k < HEAD_DIM / WMMA_K; k++) {
            // Load Q fragment [warp_m*16 : (warp_m+1)*16, k*16 : (k+1)*16]
            wmma::load_matrix_sync(a_frag,
                &Q_smem[warp_m * WMMA_M][k * WMMA_K],
                HEAD_DIM);
            
            // Load K^T fragment [warp_n*16 : (warp_n+1)*16, k*16 : (k+1)*16]
            // Note: K is row-major, we want K^T, so use col_major layout
            wmma::load_matrix_sync(b_frag,
                &K_smem[warp_n * WMMA_N][k * WMMA_K],
                HEAD_DIM);
            
            // Multiply-accumulate
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        // Apply scale and store to shared memory
        #pragma unroll
        for (int t = 0; t < acc_frag.num_elements; t++) {
            acc_frag.x[t] *= scale;
        }
        
        wmma::store_matrix_sync(
            &S_smem[warp_m * WMMA_M][warp_n * WMMA_N],
            acc_frag,
            TILE_N,
            wmma::mem_row_major
        );
        
        __syncthreads();
        
        // ========================================
        // Softmax (FP32, keep as-is)
        // ========================================
        // ... (use baseline code)
        
        // ========================================
        // WMMA: P @ V
        // ========================================
        // Similar to Q @ K^T, but:
        // - Input is P (FP32 in shared), convert to FP16
        // - Output is O (accumulate in FP16)
        
        // ... (similar structure to Q @ K^T)
    }
}
```

**Key Points**:
1. **Fragment Types**:
   - `matrix_a`: Q tile (row-major)
   - `matrix_b`: K tile (col-major for K^T)
   - `accumulator`: Output (FP32 for precision)

2. **Warp Mapping**:
   - 4 warps → 2×2 grid of 16×16 tiles
   - Each warp computes one tile independently

3. **K Dimension Loop**:
   - D=64 → 4 iterations (64/16)
   - Accumulate across K dimension

### Step 4: Implement P @ V with WMMA

**Pseudocode** (similar to Q @ K^T):
```cuda
// Load V tile
// ...

// Convert P from FP32 to FP16 for WMMA input
__shared__ half P_smem_fp16[TILE_M][TILE_N];
// ... convert S_smem (FP32) → P_smem_fp16 (FP16)

// WMMA: P @ V
wmma::fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> p_frag;
wmma::fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> v_frag;
wmma::fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> o_frag;  // FP16 output

// ... (similar to Q @ K^T)
```

---

## 📚 Reference Implementations

### periodicdent42 WMMA Kernels

Study these for examples:

1. **`cudadent42/bench/kernels/fa_phase5_wmma.cu`**
   - Full WMMA implementation for Q @ K^T and P @ V
   - Has bugs in P @ V (likely accumulator type mismatch)
   - Lines 242-400: WMMA logic

2. **`cudadent42/bench/kernels/fa_wmma_qkt.cu`**
   - WMMA for Q @ K^T only (partial)
   - Cleaner code (no P @ V bugs)
   - Lines 109-200: WMMA Q @ K^T

**How to Use**:
```bash
cd /Users/kiteboard/periodicdent42/cudadent42/bench/kernels

# Read reference implementation
less fa_wmma_qkt.cu

# Look for WMMA patterns
grep -A 10 "wmma::fragment" fa_wmma_qkt.cu
grep -A 10 "mma_sync" fa_wmma_qkt.cu
```

---

## 🧪 Testing Strategy

### Test-Driven Development (TDD)

1. **Start with Tiny Shape**:
   ```bash
   pytest tests/test_correctness.py::test_correctness[tiny-0] -v
   ```
   - Easier to debug (S=32)
   - Fast iteration

2. **Expand Gradually**:
   ```bash
   pytest tests/test_correctness.py::test_correctness[small-0] -v  # S=64
   pytest tests/test_correctness.py::test_correctness[medium-0] -v  # S=128
   ```

3. **Mission Shape (Final)**:
   ```bash
   pytest tests/test_correctness.py::test_correctness[mission-0] -v  # S=512
   ```

4. **All Seeds**:
   ```bash
   pytest tests/test_correctness.py -v  # All 15 tests
   ```

### Debugging Workflow

**If tests fail**:

1. **Check for NaN/Inf**:
   ```python
   # In kernel, add debug prints
   if (threadIdx.x == 0 && blockIdx.x == 0) {
       printf("S_smem[0][0] = %f\n", S_smem[0][0]);
   }
   ```

2. **Verify Fragment Shapes**:
   ```cuda
   // Common bug: wrong leading dimension
   wmma::load_matrix_sync(a_frag, &Q_smem[0][0], HEAD_DIM);  // ✅ Correct
   wmma::load_matrix_sync(a_frag, &Q_smem[0][0], TILE_M);    // ❌ Wrong!
   ```

3. **Check Accumulator Type**:
   ```cuda
   // For Q @ K^T (need high precision):
   wmma::fragment<accumulator, 16, 16, 16, float> acc;  // ✅ FP32

   // For P @ V (can use FP16 if careful):
   wmma::fragment<accumulator, 16, 16, 16, half> out;  // ⚠️ Risky
   ```

4. **Enable Debug Mode**:
   ```bash
   DEBUG=1 python build.py
   pytest tests/test_correctness.py::test_correctness[tiny-0] -v -s
   ```

---

## 📊 Benchmarking

### Latency Measurement

```bash
# Baseline (for comparison)
python benchmarks/benchmark_latency.py --shape mission --iters 100 --out baseline.json

# WMMA (Phase 1)
python benchmarks/benchmark_latency.py --shape mission --iters 100 --out wmma.json

# Compare
python -c "
import json
b = json.load(open('baseline.json'))
w = json.load(open('wmma.json'))
print(f'Baseline: {b["results"]["mission"]["flashcore"]["p50"]:.1f} µs')
print(f'WMMA:     {w["results"]["mission"]["flashcore"]["p50"]:.1f} µs')
print(f'Speedup:  {b["results"]["mission"]["flashcore"]["p50"] / w["results"]["mission"]["flashcore"]["p50"]:.1f}×')
"
```

**Expected**:
```
Baseline: 1500.0 µs
WMMA:      150.0 µs
Speedup:    10.0×
```

---

## 🔬 NCU Profiling

### Tensor Core Utilization

```bash
# Profile WMMA kernel
ncu --set full --launch-skip 10 --launch-count 1 \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    python benchmarks/benchmark_latency.py --shape mission --iters 1 \
    > profiling/ncu_wmma.txt

# Extract TC utilization
grep "sm__pipe_tensor_cycles_active" profiling/ncu_wmma.txt
```

**Target**: >50% Tensor Core utilization

---

## ✅ Success Criteria

### Phase 1 Complete

- ✅ **PTXAS**: ≤120 registers, ≤64 KB SMEM, 0 spills
- ✅ **Correctness**: All 15 tests pass (max_err ≤ 0.06)
- ✅ **Performance**: <200 µs on mission shape (≥7× vs baseline)
- ✅ **NCU**: Tensor Core utilization ≥50%

### Deliverables

1. ✅ `kernels/flashcore_wmma.cu` (working kernel)
2. ✅ `docs/PHASE1_REPORT.md` (results, NCU analysis)
3. ✅ Updated README (v0.2 performance table)
4. ✅ JSON artifacts (benchmark results with git SHA)

---

## 🚨 Common Pitfalls

### 1. Wrong Leading Dimension

```cuda
// ❌ WRONG
wmma::load_matrix_sync(a_frag, &Q_smem[row][col], TILE_M);

// ✅ CORRECT
wmma::load_matrix_sync(a_frag, &Q_smem[row][col], HEAD_DIM);
```

**Fix**: Leading dimension = stride of outer dimension in memory

### 2. Fragment Type Mismatch

```cuda
// ❌ WRONG: FP16 accumulator for softmax (precision loss)
wmma::fragment<accumulator, 16, 16, 16, half> acc;

// ✅ CORRECT: FP32 accumulator
wmma::fragment<accumulator, 16, 16, 16, float> acc;
```

**Fix**: Use FP32 for intermediate computations, FP16 for storage

### 3. Unaligned Memory Access

```cuda
// ❌ WRONG: Not aligned to 16-byte boundary
__shared__ half Q_smem[TILE_M][HEAD_DIM + 1];  // Padding breaks alignment

// ✅ CORRECT: No padding (or pad to multiple of 8 halves)
__shared__ half Q_smem[TILE_M][HEAD_DIM];
```

**Fix**: Keep shared memory aligned to WMMA requirements (16 bytes)

### 4. Warp Sync Issues

```cuda
// ❌ WRONG: Missing syncthreads before WMMA
// (Different warps access shared memory)

// ✅ CORRECT: Sync before and after shared memory access
__syncthreads();
wmma::load_matrix_sync(a_frag, &Q_smem[...], ...);
// No sync needed here (WMMA is warp-local)
```

---

## 📦 Starter Template

See `kernels/flashcore_wmma_template.cu` (create from baseline).

**Steps**:
1. Copy `flashcore_baseline.cu` → `flashcore_wmma.cu`
2. Add `#include <mma.h>` and `using namespace nvcuda;`
3. Replace Q @ K^T loop with WMMA (Step 3 above)
4. Replace P @ V loop with WMMA (Step 4 above)
5. Test with tiny shape first (`pytest tests/test_correctness.py::test_correctness[tiny-0]`)
6. Expand to mission shape

---

## 🎯 Estimated Timeline

| Task | Hours | Cumulative |
|------|-------|------------|
| Study references (periodicdent42) | 4 | 4 |
| Implement Q @ K^T WMMA | 8 | 12 |
| Test & debug Q @ K^T | 4 | 16 |
| Implement P @ V WMMA | 6 | 22 |
| Test & debug P @ V | 4 | 26 |
| Full test suite (15 tests) | 2 | 28 |
| Benchmark & NCU profiling | 4 | 32 |
| Documentation (PHASE1_REPORT.md) | 4 | 36 |

**Total**: 36 hours (~1 week full-time, 2-3 weeks part-time)

---

## 🚀 Ready to Begin?

**Checklist**:
- ✅ Phase 0 complete (baseline validated)
- ✅ GPU access (L4 or similar with Tensor Cores)
- ✅ Read this guide
- ✅ Studied periodicdent42 references
- ✅ Understand WMMA basics (NVIDIA guide)

**First Command**:
```bash
cd /Users/kiteboard/periodicdent42/flashcore
cp kernels/flashcore_baseline.cu kernels/flashcore_wmma.cu
# Edit flashcore_wmma.cu (add WMMA)
```

**Good luck! 🚀**

---

**Questions? See:**
- [NVIDIA WMMA Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- periodicdent42 kernels: `cudadent42/bench/kernels/fa_wmma_qkt.cu`

