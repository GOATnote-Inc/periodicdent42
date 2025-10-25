# Child-V2c Development Log

**Goal**: Achieve 100% correctness with WMMA + proper Q@K^T transpose  
**Target**: 800-1200 μs (2-3× from V2b scalar baseline)  
**Approach**: Systematic TDD iteration

---

## Iteration Timeline

### **Iteration 1: WMMA Skeleton** (0.5 hours)
**Date**: October 18, 2025  
**Goal**: Initial WMMA implementation with XOR swizzle, cp.async, transposed K

**Result**: ❌ Launch failure  
**Error**: `CUDA error: unspecified launch failure`  
**Root Cause**: WMMA fragment handling issues (likely SMEM temporaries)

**Fixes Applied**:
- Added SMEM buffers for score materialization
- Simplified fragment handling (correctness-first approach)

---

### **Iteration 2: Fixed SMEM, Launch Success** (1.0 hours)
**Date**: October 18, 2025  
**Goal**: Fix launch failure, validate kernel launches

**Result**: ✅ Builds, ✅ Launches, ❌ Correctness (max_diff=0.013)  
**Error**: Incorrect attention outputs  
**Root Cause**: WMMA computing Q @ K instead of Q @ K^T

**Analysis**:
```cuda
// Current (incorrect):
wmma::load_matrix_sync(a_frag, q_ptr, HEAD_DIM_PADDED);  // Q row-major
wmma::load_matrix_sync(b_frag, k_ptr, HEAD_DIM_PADDED);  // K row-major
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Computes Q @ K ❌

// Need: Q @ K^T (transpose K)
```

**Key Insight**: WMMA with `matrix_b` expects col-major layout for K^T, but we're loading K as row-major.

**Decision**: Before fixing WMMA transpose, validate rest of infrastructure with scalar Q@K^T.

---

### **Iteration 3: Scalar Q@K^T Validation** (0.5 hours) ⬅️ CURRENT
**Date**: October 18, 2025  
**Goal**: Validate infrastructure (softmax, SMEM, cp.async) before re-adding WMMA

**Changes**:
1. **Replaced WMMA Q@K^T with scalar** (lines 200-228)
   - Mathematically equivalent: scalar Q[i,:] @ K[j,:] = (Q @ K^T)[i,j]
   - Removes WMMA transpose complexity temporarily
   - Uses warp reduction for parallelism

2. **Fixed double-scaling bug** (line 235)
   - Score was being multiplied by `scale` twice
   - Impact: incorrect attention weights (off by factor of `scale`)

3. **Updated documentation**
   - Header comment clarifies iteration 3 goal
   - Added next step: WMMA + K^T handling

**Expected Result**: 5/5 tests pass, ~2400-2500 μs (same as V2b)

**Testing**: Ready for GPU validation (`bash evo-sdpa/TEST_V2C_V3.sh`)

---

### **Iteration 4: WMMA + K^T** (Planned, ~1-2 hours)
**Goal**: Restore WMMA with proper K^T handling

**Approach A: Transpose K During Load** (Recommended)
```cuda
// Load K as col-major in SMEM
for (int idx = tid; idx < kv_len * HEAD_DIM; idx += blockDim.x) {
    int n = idx / HEAD_DIM;
    int c = idx % HEAD_DIM;
    // Write transposed: sK[c * N_PAD + n] instead of sK[n * HEAD_DIM_PAD + c]
    sK[c * N_PADDED + n] = __ldg(&K_bh[(kv_start + n) * d + c]);
}

// WMMA with col-major K^T
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::load_matrix_sync(a_frag, &sQ[m0 * HEAD_DIM_PAD + k0], HEAD_DIM_PAD);
wmma::load_matrix_sync(b_frag, &sK[k0 * N_PAD + n0], N_PAD);  // Correct!
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Q @ K^T ✅
```

**Approach B: Swap MMA Arguments** (Faster Test)
```cuda
// Keep K row-major, compute K @ Q^T, handle transpose on result
wmma::mma_sync(c_frag, b_frag, a_frag, c_frag);  // K @ Q^T
// Transpose when storing to S_scores
```

**Expected**: 800-1200 μs, 100% correctness

---

### **Iteration 5: P@V WMMA** (Optional, ~1 hour)
**Goal**: Replace scalar P@V with WMMA

**Expected**: 400-800 μs (full WMMA pipeline)

---

## Key Metrics

| Iteration | Builds | Launches | Correctness | Latency | Status |
|-----------|--------|----------|-------------|---------|--------|
| **V2c-v1** | ✅ | ❌ | - | - | Launch fail |
| **V2c-v2** | ✅ | ✅ | ❌ (0.013) | - | Transpose bug |
| **V2c-v3** | ✅ | 🔄 | 🔄 | 🔄 | **Testing** |
| **V2c-v4** | ⏳ | ⏳ | ⏳ | 800-1200 μs | Planned |
| **V2c-v5** | ⏳ | ⏳ | ⏳ | 400-800 μs | Stretch |

---

## Bugs Found & Fixed

### **Bug 1: WMMA Fragment Handling** (Iteration 1 → 2)
- **Symptom**: `unspecified launch failure`
- **Cause**: Incorrect SMEM temporaries for WMMA fragments
- **Fix**: Added explicit SMEM buffers for score materialization

### **Bug 2: Q @ K vs Q @ K^T** (Iteration 2 → 3)
- **Symptom**: `max_diff=0.013` (incorrect attention)
- **Cause**: WMMA computing Q @ K (no transpose)
- **Fix**: Temporarily replaced with scalar Q@K^T for validation

### **Bug 3: Double Scaling** (Iteration 3)
- **Symptom**: Would cause incorrect attention weights (not yet tested)
- **Cause**: Score multiplied by `scale` twice
- **Fix**: Removed second scaling operation

---

## Time Tracking

| Phase | Time | Result |
|-------|------|--------|
| V2c-v1 Implementation | 0.5h | Launch failure |
| V2c-v2 SMEM Fix | 1.0h | Transpose bug |
| V2c-v3 Scalar Validation | 0.5h | Ready for test |
| **Total** | **2.0h** | Systematic |

**Remaining Budget**: 4-6 hours to reach 800-1200 μs (on track!)

---

## Decision Log

### **Decision 1: Use SMEM Temporaries** (Iteration 1 → 2)
**Context**: WMMA fragments causing launch failures  
**Choice**: Add explicit SMEM buffers for scores  
**Rationale**: Simpler to debug, correctness-first  
**Result**: Launch success ✅

### **Decision 2: Scalar Q@K^T Before WMMA Fix** (Iteration 2 → 3)
**Context**: WMMA Q@K^T transpose bug identified  
**Choice**: Replace with scalar temporarily  
**Rationale**: Isolate infrastructure bugs from WMMA issues (TDD)  
**Result**: Clear path to validation ✅

### **Decision 3: Approach A for WMMA K^T** (Iteration 3 → 4)
**Context**: Two approaches for WMMA transpose  
**Choice**: Transpose K during load (col-major SMEM)  
**Rationale**: Mathematically cleaner, aligns with WMMA expectations  
**Status**: Pending Iteration 4

---

## Next Actions

### **Immediate** (GPU Instance)
1. Run `bash evo-sdpa/TEST_V2C_V3.sh`
2. Verify 5/5 tests pass
3. Confirm latency ~2400-2500 μs

### **If V2c-v3 Passes** ✅
1. Implement V2c-v4 (WMMA + K^T, Approach A)
2. Test for correctness + performance
3. Target: 800-1200 μs, 100% correct

### **If V2c-v3 Fails** ❌
1. Check debug plan in `V2C_ITER3_STATUS.md`
2. Validate intermediate values (scores, m, l, O_accum)
3. Fix infrastructure bugs before WMMA

---

## Success Criteria

**Iteration 3**: 5/5 tests, ~2400 μs (validates infrastructure)  
**Iteration 4**: 5/5 tests, 800-1200 μs (WMMA Q@K^T working)  
**Iteration 5**: 5/5 tests, 400-800 μs (full WMMA pipeline)

**Final Goal**: < 5 μs (requires additional optimizations beyond V2c)

---

## Lessons Learned

1. **TDD Pays Off**: Scalar validation isolates WMMA bugs from infrastructure bugs
2. **Systematic Iteration**: Each iteration fixes one class of issues (launch → transpose → performance)
3. **Proactive Review**: Found double-scaling bug during code review (excellent!)
4. **Clear Documentation**: Each iteration has clear goal, changes, expected results

---

**Status**: V2c-v3 ready for GPU testing 🚀  
**Next**: Validate on GPU → Proceed to V2c-v4 (WMMA + K^T)


