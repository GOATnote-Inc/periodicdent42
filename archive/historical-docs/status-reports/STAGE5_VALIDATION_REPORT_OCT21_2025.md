# 🔬 Stage-5 L4 GPU Validation Report

**Date**: October 21, 2025  
**GPU**: NVIDIA L4 (Ada, sm_89)  
**Branch**: `feat/stage5-warp-spec-persistent`  
**Status**: ❌ **Critical Issues Identified**

---

## 📊 Performance Results

| Variant | p50 Latency | vs PyTorch Math | vs PyTorch Flash | Correctness | Max Error |
|---------|-------------|-----------------|------------------|-------------|-----------|
| **PyTorch Math** | **575 μs** | 1.0× (baseline A) | 7.5× slower | ✅ Pass | - |
| **PyTorch Flash** | **77 μs** | 7.5× faster | 1.0× (baseline B) | ✅ Pass | - |
| **Stage-2** | **1793 μs** | **0.32× (3.1× SLOWER)** | **0.043× (23× SLOWER)** | ❌ **FAIL** | `nan` |
| **WS-P1** | **1782 μs** | **0.32× (3.1× SLOWER)** | **0.043× (23× SLOWER)** | ❌ **FAIL** | `nan` |
| **WS-P2** | **1765 μs** | **0.33× (3.1× SLOWER)** | **0.044× (23× SLOWER)** | ❌ **FAIL** | 2.28 |

---

## 🚨 Critical Issues

### 1. Severe Performance Regression (3-25× slower)

**Observation**:
- Our FP8 kernels are **23-25× SLOWER** than PyTorch's optimized flash attention (77 μs)
- Even **3× SLOWER** than PyTorch's unoptimized math backend (575 μs)
- This is the **opposite** of our ≥15× speedup target

**Root Causes**:
1. **FP8 Quantization Overhead**: Simulated FP8 quantization (uint8 + scale factors) adds significant CPU/GPU overhead without actual FP8 hardware benefits
2. **No Tensor Core Utilization**: Despite WMMA code being present, actual tensor core activity is likely minimal
3. **Memory Inefficiency**: Excessive shared memory usage (42-44 KB) may be limiting occupancy
4. **Suboptimal Tile Sizes**: TILE_M=32, TILE_N=32 may not be optimal for L4's architecture

### 2. 100% Correctness Failure Rate

**Observation**:
- **All 3 kernel variants FAIL correctness**
- Stage-2 and WS-P1: `max_err = nan` (computation crash/undefined behavior)
- WS-P2: `max_err = 2.28` (38× higher than 0.06 threshold)

**Root Causes** (Hypothesized):
1. **Softmax Numerical Instability**: Despite having per-row `m_smem`/`l_smem` arrays, there may be bugs in how they're updated across KV tiles
2. **FP8 Quantization Errors**: The simulated FP8 path may have precision loss that compounds errors
3. **WMMA Data Layout Mismatch**: Possible misalignment between WMMA expectations and actual data layout in shared memory
4. **Race Conditions**: Missing synchronization between producer/consumer warps (WS path)

### 3. WS Toggles Not Working

**Observation**:
- **All builds show `USE_WARP_SPECIALIZATION: 0`** in PTXAS output
- WS-P1 and WS-P2 are functionally **identical** to Stage-2 (no specialization)
- Environment variables set in Python wrappers are **NOT being read** by `build.py`

**Root Cause**:
- PyTorch's JIT compilation caches extensions by name
- Even with unique names (`sdpa_fp8_ws_p1`, etc.), the **source files are identical**
- PyTorch sees no file changes → skips rebuild → reuses old binary
- Environment variables are set AFTER PyTorch checks the cache

**Impact**:
- WS-P1 and WS-P2 benchmarks are **invalid** (testing same code 3 times)
- True WS performance is **unknown**

---

## 🔍 Code Review Findings

### Already Implemented Fixes ✅

1. **Per-Row Softmax Statistics**: 
   - `m_smem[TILE_M]` and `l_smem[TILE_M]` arrays exist
   - Each row's max/sum is tracked independently
   - Code at lines 765-823 correctly uses per-row values

2. **Output in Shared Memory**:
   - `U_smem[TILE_M][D_PAD]` used for output accumulation (not registers)
   - Reduces register pressure significantly

3. **Per-Row Work Distribution**:
   - Each warp processes multiple rows: `for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS)`
   - No idle warps during softmax/P·V

### Potential Bugs 🐛

1. **Softmax Update Logic** (Lines 765-823):
   ```cuda
   float m_old = m_smem[r];
   float m_new = m_old;
   for (int n = 0; n < kv_len; ++n) {
       m_new = fmaxf(m_new, S_row[n]);  // ← Per-block max, not per-tile
   }
   
   float rescale = __expf(m_old - m_new);
   float l_new = l_old * rescale + l_add;
   
   // Scale U
   for (int d = lane; d < D; d += 32) {
       U_smem[r][d] *= rescale;  // ← This scales ALL of U, not just current tile's contribution
   }
   ```
   
   **Issue**: The `rescale` factor is computed per KV tile, but `U_smem[r][d] *= rescale` scales the **entire accumulated output** from all previous tiles. This is correct for online softmax, BUT if `m_new` stays at `-INFINITY` (e.g., all scores are very negative), `rescale = exp(-INF - m_new)` can produce `NaN`.

2. **FP8 Quantization Precision**:
   - Using `uint8` with per-head scale factors
   - Possible overflow/underflow during dequantization
   - E4M3 format has limited dynamic range (±448)

3. **WMMA Fragment Usage** (Lines 700-750 for Q@K^T):
   ```cuda
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
   wmma::fill_fragment(c_frag, 0.0f);
   
   for (int kStep = 0; kStep < D; kStep += WMMA_K) {
       wmma::load_matrix_sync(a_frag, &sQ[warp_m][kStep], D_PAD);
       wmma::load_matrix_sync(b_frag, &sKT[warp_n][kStep], D_PAD);  // ← col-major
       wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   }
   
   wmma::store_matrix_sync(&sS[warp_m][warp_n], c_frag, TILE_N, wmma::mem_row_major);
   ```
   
   **Potential Issue**: If `sKT` layout doesn't match WMMA's col-major expectations, results will be incorrect.

---

## 🎯 Recommended Next Steps

### Option A: Document as Valid Negative Result ⭐ **RECOMMENDED**

**Rationale**:
- 100% correctness failure + 3-25× perf regression = fundamental issues
- Debugging requires deep dive into softmax math, FP8 precision, and WMMA layouts
- Infrastructure works (pipeline, benchmarking, NCU) → valuable learning

**Actions**:
1. Create `STAGE5_VALID_NEGATIVE_RESULT.md` documenting:
   - What we built (EvoEngineer-Full pipeline, WS kernel structure)
   - What we found (correctness failures, perf regression, root causes)
   - Lessons learned (FP8 overhead, tensor core challenges, build system caching)
2. Commit artifacts (`artifacts/bench/*.json`, fixed scripts)
3. Tag as `v5.0-stage5-learning-milestone`
4. Move forward with alternative approaches (e.g., native FP16, different tile sizes)

**Timeline**: 2-3 hours

---

### Option B: Quick Debug Session (Build System + Softmax)

**Rationale**:
- WS toggles not working is a **solvable** build system issue
- Softmax NaN may be fixable with numerical guards

**Actions**:
1. **Fix Build Caching**:
   - Modify `build.py` to include compile flags in cache key
   - Or force rebuild by touching source files after env var changes
   - Re-run benchmarks with TRUE WS-P1/WS-P2 kernels

2. **Add Softmax Guards**:
   ```cuda
   // Prevent NaN from m_new = -INF
   if (m_new == -INFINITY || !isfinite(m_new)) {
       m_new = -1e10f;  // Large negative, but finite
   }
   
   // Clamp rescale to prevent overflow
   float rescale = __expf(fmaxf(m_old - m_new, -20.0f));  // exp(-20) ≈ 2e-9
   ```

3. **Validate on Small Case**:
   - Test (1, 2, 64, 64) shape
   - Check if max_err drops below 0.06

**Timeline**: 4-6 hours

---

### Option C: Revert to Stage-2 Baseline (Pre-WS)

**Rationale**:
- Stage-2 kernel (from previous sessions) was validated as correct
- WS additions may have introduced bugs

**Actions**:
1. Checkout `main` branch
2. Run Stage-2 kernel from `tasks/fp8_sdpa_stage_c_wmma/`
3. Verify correctness and performance
4. Compare with current broken version to isolate WS-related bugs

**Timeline**: 2-3 hours

---

## 📈 Performance Analysis

### Why is FP8 Slow?

**Hypothesis**: Simulated FP8 is adding overhead without benefits

**Evidence**:
1. **Quantization Cost**: Each input tensor requires:
   - Per-head max computation (reduction over B×S×D elements)
   - Scale factor calculation (127 / max)
   - Element-wise multiply + clamp + cast to uint8
   - **Estimated overhead**: ~50-100 μs per tensor × 3 tensors = 150-300 μs

2. **Dequantization Cost**: In kernel, each element loaded requires:
   - `uint8` → `float` cast
   - Multiply by scale factor
   - `float` → `half` cast (for WMMA)
   - **No hardware FP8 instructions used!**

3. **Lack of Tensor Core Benefits**: L4 has FP8 Tensor Cores, but:
   - We're not using native FP8 WMMA (requires CUDA 12.4+ and explicit FP8 fragments)
   - Current code uses FP16 WMMA on dequantized data
   - Missing 2× throughput boost from native FP8

**Conclusion**: Simulated FP8 adds ~200-300 μs overhead + misses 2× TC speedup = net loss

---

### Why is PyTorch So Fast?

**PyTorch Flash Attention** (77 μs) uses:
1. **Native FP16** (no quantization overhead)
2. **FlashAttention-2 algorithm** (optimized tiling, minimizes HBM traffic)
3. **Highly tuned kernels** (years of optimization, profile-guided)
4. **No Python/JIT overhead** (pre-compiled kernels)

**PyTorch Math** (575 μs) uses:
1. **cuBLAS GEMM** for Q@K^T and P@V (highly optimized)
2. **Standard attention** (materializes full S matrix)
3. **Still faster than our kernel!** (cuBLAS is very good)

---

## 🧪 Infrastructure Validation

### What Worked ✅

1. **EvoEngineer-Full Pipeline**:
   - `kbench.py`: Deterministic benchmarking with p50/p90/CI
   - `evo_tune.py`: Elite preservation search (untested due to broken kernels)
   - `profile.sh`: NCU automation (skipped due to broken kernels)
   - `summarize.py`: Auto-report generation

2. **Build System**:
   - Multiple extension names (`sdpa_fp8_stage2_baseline`, `sdpa_fp8_ws_p1`, etc.)
   - PTXAS stats captured (96 regs, 37 KB SMEM, 0 spills)
   - Ninja integration working

3. **Correctness Harness**:
   - Comparison vs PyTorch SDPA (flash backend)
   - Max abs error threshold (0.06 for FP8)
   - Detected all failures correctly

### What Didn't Work ❌

1. **Environment Variable Propagation**:
   - Python `os.environ` changes don't affect PyTorch JIT cache
   - Need to modify source files OR use compile flags in cache key

2. **WS Toggle Testing**:
   - All 3 variants compiled identically (USE_WARP_SPECIALIZATION=0)
   - Invalid benchmark data for WS-P1 and WS-P2

3. **NCU Profiling Skipped**:
   - No point profiling broken kernels
   - Would have provided valuable bottleneck insights if kernels worked

---

## 💡 Key Lessons Learned

### Technical

1. **FP8 Quantization is Non-Trivial**:
   - Simulated FP8 adds overhead without benefits
   - Native FP8 requires CUDA 12.4+ and careful programming
   - FP16 is simpler and often faster for prototyping

2. **PyTorch JIT Caching is Aggressive**:
   - Changing environment variables doesn't trigger rebuild
   - Need to either touch source files OR include flags in cache key
   - Unique extension names alone are insufficient

3. **Correctness is Hard**:
   - Online softmax requires careful numerical handling
   - FP8 quantization compounds precision errors
   - WMMA layout mismatches produce wrong results silently

### Methodological

1. **"GREEN before FAST" is Critical**:
   - We tried to optimize (WS, FP8) before achieving correctness
   - Should have validated Stage-2 baseline first, then added WS
   - TDD approach would have caught issues earlier

2. **Infrastructure is Reusable**:
   - Even with kernel failures, the pipeline/benchmarking code is valuable
   - Can be used for future kernel development
   - EvoEngineer-Full framework is solid

3. **Valid Negative Results are Valuable**:
   - Learning what doesn't work is progress
   - Documenting failure modes helps future work
   - Builds institutional knowledge

---

## 📚 Artifacts Generated

```
sdpa_ws_pipeline/
├── artifacts/
│   ├── bench/
│   │   ├── baseline_a.json         # PyTorch math: 575 μs
│   │   ├── baseline_b.json         # PyTorch flash: 77 μs
│   │   ├── candidate_stage2.json   # Our kernel: 1793 μs, FAIL
│   │   ├── candidate_ws_p1.json    # Same as stage2 (toggle bug)
│   │   └── candidate_ws_p2.json    # Same as stage2 (toggle bug)
│   └── manifest.yaml (not created - env capture failed)
└── reports/ (not created - skipped due to failures)

scripts/
├── kbench.py                       # ✅ Working, validated
├── evo_tune.py                     # ⚠️ Untested (skipped)
├── profile.sh                      # ⚠️ Untested (skipped)
├── parse_ncu.py                    # ⚠️ Untested (skipped)
├── capture_env.py                  # ⚠️ Needs pyyaml install
├── bench.sh                        # ✅ Working
├── repro.sh                        # ⚠️ Partial (stopped at autotune)
└── summarize.py                    # ⚠️ Untested (skipped)
```

---

## 🎓 Conclusion

**Stage-5 validation revealed fundamental correctness and performance issues**:
- ❌ 100% correctness failure (NaN/2.28 errors vs 0.06 threshold)
- ❌ 3-25× performance regression (opposite of ≥15× target)
- ❌ WS toggles not working (build system bug)

**However, we successfully built valuable infrastructure**:
- ✅ EvoEngineer-Full pipeline framework
- ✅ Robust benchmarking harness
- ✅ Multi-variant kernel wrappers
- ✅ Comprehensive documentation

**Recommendation**: **Document as valid negative result** and pivot to:
1. Native FP16 kernels (remove FP8 overhead)
2. Fix build system caching (for future WS testing)
3. Validate Stage-2 baseline from scratch
4. Apply incremental optimizations with correctness gates

---

**Next Action**: Choose Option A, B, or C above.

**Estimated Time to Recovery**:
- Option A (Document): 2-3 hours
- Option B (Debug): 4-6 hours
- Option C (Revert): 2-3 hours

---

**Report Generated**: October 21, 2025  
**Status**: Awaiting decision on next steps

