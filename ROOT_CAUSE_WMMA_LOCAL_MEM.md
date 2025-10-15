# Root Cause Analysis: WMMA Local Memory Issue
## October 15, 2025 - Complete Resolution

---

## Executive Summary

**Issue**: `RuntimeError: Kernel runtime failed: unspecified launch failure`  
**Root Cause**: **WMMA `load_matrix_sync` cannot operate on local memory (registers)**  
**Evidence**: Compute-sanitizer revealed 3389 "Invalid __local__ read" errors  
**Status**: ✅ **ROOT CAUSE IDENTIFIED** (fix requires SMEM refactor)

---

## Evidence Chain

### 1. Build Metrics (PTXAS) ✅ CLEAN
```
Config 0 (16x64): 111 regs, 0 spills, 36KB SMEM, 800B stack
Config 1 (32x64, STAGES=1): 127 regs, 0 spills, 24KB SMEM, 1344B stack
Config 2 (48x64): 95 regs, 0 spills, 45KB SMEM, 1072B stack
Config 3 (32x32): 69 regs, 0 spills, 24KB SMEM, 1216B stack
Config 4 (32x64, STAGES=2): 111 regs, 0 spills, 41KB SMEM, 1344B stack
```

**Verdict**: Build is clean. Issue is **not** register pressure, spills, or SMEM overflow.

### 2. Compiler Warnings ⚠️ CRITICAL CLUE
```
warning: /usr/local/cuda/include/crt/mma.hpp(91): 
Warning: cannot perform wmma load or store on local memory
```

**Repeated 10 times** during compilation.

**What we missed**: This warning was the key. WMMA API requires **shared memory or global memory pointers**, not register arrays.

### 3. Compute-Sanitizer Output ❌ SMOKING GUN

**Command**: `compute-sanitizer --tool memcheck tile_oracle_v3.py`

**Result**: **3389 errors**

**Critical Error**:
```
Invalid __local__ read of size 4 bytes
  at nvcuda::wmma::load_matrix_sync(...) in mma.hpp:91
  by thread (114,0,0) in block (59,0,0)
  Address 0xfffdc8 is out of bounds
```

**Repeated for multiple threads**: (114,0,0), (115,0,0), (116,0,0), etc.

**What this means**:
- `load_matrix_sync` is trying to read from address `0xfffdc8` (local memory / stack)
- This address is **out of bounds** for WMMA fragment loading
- WMMA API expects **aligned shared memory or global memory**, not register/stack arrays

---

## Code Analysis

### Current Implementation (BROKEN)

```cpp
// In compute_block() - fa_s512_v3.cu
half Q_reg[MAX_ROWS_PER_THREAD][HEAD_DIM];  // LOCAL MEMORY (registers/stack)

// ...later...

#if defined(USE_WMMA)
if constexpr (Traits::HEAD_DIM % 16 == 0 && Traits::BLOCK_N % 16 == 0) {
    qk_row_wmma<Traits>(
        &Q_reg[local_row][0],  // ❌ PASSING LOCAL MEMORY POINTER
        &smem->K[stage][0][0],  // ✅ Shared memory (OK)
        S_row,
        softmax_scale,
        Traits::BLOCK_N
    );
}
#endif

// Inside qk_row_wmma
template<typename Traits>
__device__ void qk_row_wmma(...) {
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a;
    load_matrix_sync(a, q_row, Traits::HEAD_DIM);  // ❌ FAILS: q_row is local memory
    // ...
}
```

### Why It Fails

1. **`Q_reg`** is declared as a **local array** (allocated in registers/stack per thread)
2. **WMMA `load_matrix_sync`** requires:
   - **Shared memory** (preferred, allows cooperative loading)
   - **Global memory** (works, but slower)
   - **NOT local memory** (registers/stack) ❌

3. When `load_matrix_sync` tries to read from `&Q_reg[local_row][0]`, it:
   - Interprets the address as a memory space pointer
   - Attempts a memory load
   - Fails because the address is in the **local address space** (thread-private)
   - Results in "Invalid __local__ read" and "Address out of bounds"

---

## Fix Strategy

### Option A: Move Q to Shared Memory (Recommended)

```cpp
// In SharedMemory struct
__align__(16) half Q_tile[BLOCK_M][HEAD_DIM];  // NEW: Q in SMEM
__align__(16) half K[STAGES][BLOCK_N][K_STRIDE];
__align__(16) half V[STAGES][BLOCK_N][V_STRIDE];
__align__(16) float O_accum[BLOCK_M][HEAD_DIM];
```

**Pros**:
- WMMA can load directly from SMEM
- Cooperative loading possible (warp-level)
- Clean architecture

**Cons**:
- Increases SMEM usage by `BLOCK_M * HEAD_DIM * 2 bytes`
- For 32x64: +4KB SMEM (within limits for all configs)

### Option B: Use Fragment Fill + Manual Compute (Workaround)

```cpp
// Instead of load_matrix_sync from local memory
fragment<matrix_a, 16, 16, 16, half, row_major> a;
for (int i = 0; i < a.num_elements; i++) {
    a.x[i] = Q_reg[local_row][i];  // Manual element-wise copy
}
// Then use mma_sync as before
```

**Pros**:
- No SMEM increase
- Minimal code change

**Cons**:
- Loses WMMA load coalescing benefits
- Performance may regress vs scalar

### Option C: Disable WMMA, Use Scalar Path

```cpp
// Simply comment out or #if 0 the WMMA path
// Kernel reverts to scalar QK^T computation
```

**Pros**:
- Immediate fix
- Clean scalar path (already validated)

**Cons**:
- No Tensor Core utilization
- Performance regression vs proper WMMA

---

## Recommended Action

**Phase 1 (Immediate)**: Disable WMMA to unblock benchmarks

```diff
diff --git a/cudadent42/bench/build_v3_release.py b/cudadent42/bench/build_v3_release.py
--- a/cudadent42/bench/build_v3_release.py
+++ b/cudadent42/bench/build_v3_release.py
@@ -10,7 +10,7 @@ def build_v3_release(debug: bool=False):
         "-gencode=arch=compute_89,code=sm_89",
-        "-DUSE_WMMA",
+        # "-DUSE_WMMA",  # Disabled: requires Q in SMEM (not local)
     ]
```

**Phase 2 (Follow-up)**: Implement Option A

1. Move `Q_tile` to `SharedMemory` struct
2. Cooperative load Q from gmem → SMEM (all threads)
3. Update `qk_row_wmma` to load from SMEM
4. Re-enable `-DUSE_WMMA`
5. Validate with sanitizer (should be 0 errors)
6. Benchmark vs SDPA

**Estimated Effort**:
- Phase 1: 5 minutes (comment out flag)
- Phase 2: 2-4 hours (SMEM refactor + validation)

---

## Lessons Learned

1. **Compiler warnings are critical**: The mma.hpp warning was the key clue
2. **Sanitizer is essential**: Without compute-sanitizer, we'd still be guessing
3. **WMMA API constraints**: Not all pointer types are valid for WMMA
4. **Test early with sanitizers**: Should have been first diagnostic, not last

---

## Artifacts Collected

| Artifact | Status | Location | Key Finding |
|:---------|:------:|:---------|:------------|
| PTXAS Stats | ✅ | `artifacts/stats/ptxas.txt` | 0 spills, clean build |
| Sanitizer Log | ✅ | `artifacts/sanitizers/compute-sanitizer.log` | 3389 errors, root cause |
| Build Warnings | ✅ | Build log | "cannot perform wmma on local memory" |
| WMMA Proof | ⚠️ | `artifacts/stats/wmma_proof.txt` | .so missing (minor) |
| Benchmarks | ❌ | Blocked by runtime error | Pending Phase 1 fix |

---

## Next Steps (Ordered)

1. **Commit this report + artifacts** (preserve evidence)
2. **Disable WMMA** (unblock benchmarks)
3. **Run baseline bench** (SDPA vs V3 scalar)
4. **Document performance** (establish scalar baseline)
5. **Implement Phase 2** (SMEM Q tile)
6. **Re-enable WMMA** (with proper SMEM pointers)
7. **Validate** (sanitizer should show 0 errors)
8. **Benchmark** (compare WMMA vs scalar vs SDPA)
9. **Nsight Compute** (profile Tensor Core utilization)
10. **Merge PR** (evidence + fix + baseline)

---

## Success Criteria (Updated)

### Minimum (Evidence Complete) ✅
- ✅ PTXAS: 0 spills
- ✅ Sanitizer: Root cause identified (WMMA local memory)
- ✅ Documentation: Complete analysis with fix strategy
- ⏳ Benchmarks: Baseline with WMMA disabled (next)

### Full (WMMA Working)
- Move Q to SMEM
- Sanitizer: 0 errors
- WMMA enabled + validated
- Benchmark: WMMA ≥ scalar ≥ 0.5× SDPA (realistic target)
- Nsight: Tensor Core utilization > 0%

---

**Status**: ✅ ROOT CAUSE COMPLETE. Fix strategy defined. Ready for Phase 1 (disable WMMA) + baseline bench.

**Date**: October 15, 2025 19:35 UTC  
**Evidence**: 452KB sanitizer log, PTXAS stats, build warnings  
**Next**: Disable WMMA → run bench → commit baseline → Phase 2 SMEM refactor

