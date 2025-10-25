# FlashCore Phase 1C: Tensor Core Implementation Plan

**Date**: October 21, 2025  
**Phase**: 1C - Tensor Core Acceleration (WMMA)  
**Status**: ⏳ READY TO START  
**Goal**: 4-6× speedup → ~110 μs target

---

## 🎯 Objective

Transform scalar/vectorized operations into Tensor Core (WMMA) operations for massive acceleration:

**Current (Phase 1A)**: 546 μs
- Vectorized float4 loads (memory optimized)
- Scalar dot products (compute limited)

**Target (Phase 1C)**: ~110 μs (5× speedup)
- WMMA for Q@K^T (16×16×16 tiles)
- WMMA for P@V (16×16×16 tiles)
- Tensor Core hardware acceleration

---

## 📊 Why Tensor Cores?

### Performance Potential
```
L4 GPU Tensor Core Specs:
- 16×16×16 FP16 matmul per instruction
- ~10× faster than scalar FP32 arithmetic
- Theoretical: 242 TFLOPS (vs 31 TFLOPS FP32)

Expected Impact:
- Q@K^T: 4-6× faster (replaces inner loop)
- P@V: 3-5× faster (replaces accumulation)
- Combined: 4-6× overall speedup
```

### FlashAttention Reference
From research (FlashAttention-2):
- Tensor Cores: 2-3× speedup over vectorized code
- Combined with tiling: 5-10× total improvement
- Our target: Conservative 5× (110 μs)

---

## 🏗️ Implementation Strategy

### Phase 1C.1: Q@K^T with WMMA (4-6 hours)

**Current Code (Phase 1A)**:
```cuda
// Vectorized dot product (8 halfs at a time)
for (int d = 0; d < HEAD_DIM; d += 8) {
    const float4 k_vec = *reinterpret_cast<const float4*>(&K[k_offset + d]);
    const half* k_half = reinterpret_cast<const half*>(&k_vec);
    for (int i = 0; i < 8; i++) {
        score += Q_row[d + i] * __half2float(k_half[i]);
    }
}
```

**Target Code (WMMA)**:
```cuda
// Declare fragments (16×16×16 tiles)
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

// Load Q tile into fragment (convert float→half)
half Q_half[HEAD_DIM];
for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
    Q_half[d] = __float2half(Q_row[d]);
}
__syncthreads();

// Compute Q @ K^T in 16×16 tiles
// HEAD_DIM=64 → 4 tiles (64/16=4)
wmma::fill_fragment(s_frag, 0.0f);
for (int k_tile = 0; k_tile < 4; k_tile++) {
    wmma::load_matrix_sync(q_frag, &Q_half[k_tile*16], 64);
    wmma::load_matrix_sync(k_frag, &K_tile[warp_id*16][k_tile*16], 64);
    wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
}

// Store result
wmma::store_matrix_sync(&S_tile[0][warp_id*16], s_frag, 64, wmma::mem_row_major);
```

**Steps**:
1. Convert Q_row from float→half (Tensor Cores need FP16)
2. Tile K into 16×16 chunks
3. Use WMMA to compute Q @ K^T
4. Store attention scores back to S_tile

**Expected**: 2× speedup on Q@K^T → ~450 μs total

---

### Phase 1C.2: P@V with WMMA (4-6 hours)

**Current Code (Phase 1A)**:
```cuda
for (int d = tid; d < HEAD_DIM; d += THREADS_PER_BLOCK) {
    float acc = 0.0f;
    for (int n_idx = 0; n_idx < block_size; n_idx++) {
        float p_val = expf(S_tile[n_idx] - m_new);
        acc += p_val * __half2float(V[v_offset]);
    }
    atomicAdd(&O_accum[d], acc);
}
```

**Target Code (WMMA)**:
```cuda
// Convert P (attention weights) to half, store in shared memory
__shared__ half P_tile[BLOCK_N][16];  // Attention weights
__shared__ half V_tile[BLOCK_N][HEAD_DIM];  // Values

// Compute P @ V using WMMA
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;

// Load and compute in 16×16 tiles
for (int n_tile = 0; n_tile < num_n_tiles; n_tile++) {
    wmma::load_matrix_sync(p_frag, &P_tile[n_tile*16][0], 16);
    wmma::load_matrix_sync(v_frag, &V_tile[n_tile*16][0], HEAD_DIM);
    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
}

// Store output
wmma::store_matrix_sync(&O_accum[0], o_frag, HEAD_DIM, wmma::mem_row_major);
```

**Steps**:
1. Create P_tile (attention weights as half)
2. Load V_tile into shared memory
3. Use WMMA to compute P @ V
4. Accumulate into O_accum

**Expected**: 2-3× speedup on P@V → ~200-250 μs total

---

### Phase 1C.3: Integration & Tuning (2-4 hours)

**Challenges**:
1. **Dimensions**: Ensure all tiles are 16-aligned
   - HEAD_DIM=64 ✅ (64/16=4 tiles)
   - BLOCK_N=64 ✅ (64/16=4 tiles)

2. **Precision**: FP16 for Tensor Cores, FP32 for accumulators
   - Q_row: float → half conversion needed
   - S_tile: Store as half initially, convert for softmax
   - O_accum: Keep as float (accumulate in FP32)

3. **Shared Memory**: WMMA needs more SMEM
   - Current: 768B
   - Target: ~8-12KB (Q_half, K_tile, V_tile, P_tile)
   - Still under 16KB limit (L4 has 64KB per SM)

4. **Occupancy**: Watch register usage
   - Current: 96 registers
   - WMMA adds ~20-30 registers for fragments
   - Target: <150 registers (should be fine)

**Tuning**:
- Adjust tile sizes if needed (8×8×8 vs 16×16×16)
- Profile with NCU to verify Tensor Core utilization
- Ensure >50% tensor_active time

---

## 🔬 Testing & Validation

### Correctness Tests
```python
# Phase 1C must pass all existing tests
pytest tests/test_correctness.py -v

# Expected:
# - 20/20 tests PASS
# - max_err < 0.06 (FP16 may have slightly higher error)
# - No NaN/Inf values
```

### Performance Benchmarking
```python
python benchmarks/benchmark_latency.py --shape mission --iters 100

# Expected:
# - Latency: ~90-140 μs (target: 110 μs)
# - Speedup vs Phase 1A: 4-6×
# - Speedup vs Baseline: 12-15×
```

### NCU Profiling
```bash
ncu --metrics tensor_active,dram_throughput python test_tc.py

# Expected metrics:
# - sm__pipe_tensor_active: >50% (Tensor Cores engaged!)
# - dram__throughput: <60% (memory not bottleneck)
# - smsp__warps_active: >40% (good occupancy)
```

---

## 📈 Risk Assessment

### HIGH RISK Items

**1. Complexity**
- WMMA is intricate (fragment management, tiling)
- Risk: Bugs, incorrect results
- Mitigation: Incremental development (Q@K^T first, test, then P@V)

**2. Shared Memory**
- Need ~8-12KB (vs 768B current)
- Risk: Reduced occupancy
- Mitigation: Profile occupancy, adjust if needed

**3. Numerical Stability**
- FP16 has less precision than FP32
- Risk: Accumulation errors, overflow
- Mitigation: Use FP32 accumulators, test edge cases

### MEDIUM RISK Items

**4. Register Pressure**
- WMMA fragments use registers
- Risk: Spills to local memory
- Mitigation: Monitor PTXAS output, reduce live variables

**5. Alignment**
- WMMA requires 16-byte aligned pointers
- Risk: Misalignment crash or silent errors
- Mitigation: Verify alignment, use `__align__(16)`

### LOW RISK Items

**6. Performance**
- May not hit 5× speedup
- Risk: Only 3× speedup (still good!)
- Mitigation: Fallback acceptable at 3× (still 182 μs, progress!)

---

## 🎯 Success Criteria

### Minimum Viable (Must Have)
- ✅ Correctness: All tests pass (max_err < 0.06)
- ✅ Performance: <200 μs (at least 2.5× from Phase 1A)
- ✅ Stability: No crashes, no NaN/Inf

### Target (Should Have)
- ✅ Performance: <140 μs (4× from Phase 1A)
- ✅ PTXAS: <150 registers, <16KB smem, 0 spills
- ✅ NCU: tensor_active >40%

### Stretch (Nice to Have)
- ✅ Performance: <110 μs (5× from Phase 1A)
- ✅ NCU: tensor_active >60%
- ✅ All optimizations: Q@K^T + P@V both using WMMA

---

## 💰 Budget & Timeline

### Time Estimate
```
Phase 1C.1 (Q@K^T WMMA):     4-6 hours
Phase 1C.2 (P@V WMMA):       4-6 hours
Phase 1C.3 (Integration):    2-4 hours
────────────────────────────────────────
Total:                       10-16 hours

Realistic: 12 hours ($9 at $0.75/hour)
```

### Budget Status
```
Spent So Far:       $2.07
Phase 1C Budget:    $6-$12
Remaining After:    $23-$29
Phase 2 Budget:     $15-$30

Status: ✅ Sufficient budget for full project
```

---

## 🚀 Getting Started

### Step 1: Environment Setup (5 min)
```bash
# Connect to L4
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c
cd ~/flashcore

# Verify Phase 1A works
python3 test_vec.py  # Should show 546 μs

# Create Phase 1C kernel
cp kernels/flashcore_vec.cu kernels/flashcore_tc.cu
```

### Step 2: Add WMMA Headers (5 min)
```cuda
// Add to top of flashcore_tc.cu
#include <mma.h>
using namespace nvcuda;

// Update constants
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
```

### Step 3: Implement Q@K^T WMMA (4-6 hours)
- Convert Q_row to FP16
- Tile K matrix
- Use WMMA fragments
- Test correctness

### Step 4: Implement P@V WMMA (4-6 hours)
- Create P_tile from attention weights
- Use WMMA for P @ V
- Test correctness

### Step 5: Profile & Tune (2-4 hours)
- NCU profiling
- Adjust tile sizes
- Optimize shared memory usage

---

## 📚 References

### Internal Code
- `~/periodicdent42/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu` - Existing WMMA implementation
- `~/periodicdent42/cudadent42/bench/kernels/fa_phase6_scalar.cu` - Scalar reference

### NVIDIA Documentation
- WMMA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
- Tensor Core Best Practices
- L4 GPU Architecture (Ada Lovelace)

### Papers
- FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al.)
- FlashAttention-2: Faster Attention with Better Parallelism

---

## 🎯 Bottom Line

**Phase 1C Goal**: 5× speedup (546 μs → ~110 μs)

**Method**: WMMA Tensor Cores for Q@K^T and P@V

**Time**: 10-16 hours (~12 hours realistic)

**Cost**: $6-$12 (~$9 expected)

**Risk**: HIGH (complex) but PROVEN technique

**Confidence**: MEDIUM-HIGH (challenging but achievable)

---

**Status**: Plan complete, ready to implement! 🚀

**Next**: Start with Step 1 (environment setup) and proceed incrementally through Q@K^T WMMA implementation.

