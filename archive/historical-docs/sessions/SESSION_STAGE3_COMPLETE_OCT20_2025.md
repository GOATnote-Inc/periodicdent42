# Stage-3 Full Fusion — Session Complete (Oct 20, 2025)

**Branch**: `feat/stage3-fusion-full`  
**Last Commit**: `9c4136b` - Session 4 summary appended  
**Total Time**: ~12 hours (Implementation: 8h, Debugging: 4h)  
**Status**: ❌ **ABANDONED** (Valid Negative Result)

---

## 📊 Summary

### Objective
Implement Stage-3B: Fused Softmax + WMMA P·V to eliminate `sS` buffer and achieve +15-25% speedup over Stage-2 (656 μs baseline).

### Result
**FAILED** — Correctness gate blocked after 3 systematic debugging attempts with 0 meaningful progress.

---

## ✅ What Succeeded

### 1. **PTXAS Gate: PASSED** 🎉
```
Stage-2 (baseline):    96 regs, 37.1 KB SMEM, 0 spills
Stage-3B (fused):      83 regs, 35.1 KB SMEM, 0 spills
Improvement:           -13 regs (-14%), -2 KB SMEM (-5%)
```

**Evidence**: Fused softmax in registers DOES reduce resource usage as predicted.

### 2. **Infrastructure & Tooling**
- ✅ Feature flag system (`USE_FUSED_SOFTMAX`, `USE_SMEM_SWIZZLE_XOR`, `USE_CP_ASYNC_3STAGE`)
- ✅ WMMA accumulator LUT generation (`scripts/generate_wmma_lut.py`, `wmma16x16_accum_lut.h`)
- ✅ Robust validation framework (`tasks/fp8_sdpa_stage_c_wmma/`)
- ✅ Comprehensive documentation (`STAGE3_FUSION_FULL_PLAN.md`, `STAGE3B_HOTFIX_STATUS.md`)

### 3. **"Valid Negative" Results**
- ✅ **XOR Swizzle (Step-2)**: +6.1% regression (696 μs vs 656 μs) → bank conflicts not the bottleneck
- ✅ **Fused Softmax (Step-3)**: Deep algorithmic bug → requires redesign, not debugging

---

## ❌ What Failed

### Correctness Gate: 0/6 Tests After 3 Fix Attempts

| Attempt | Hypothesis | Changes | Result |
|---------|------------|---------|--------|
| **Fix #1** | Invalid columns in partial KV tiles polluting max/sum | KV mask (`cc < kv_local`), dynamic fragment size, block sync | ❌ max_err 4.1 → 4.1 (no change) |
| **Fix #2** | Stale data in `sP` from previous KV tiles | Zero ENTIRE 16×16 sP tile | ❌ max_err 4.1 → 4.1 (no change) |
| **Fix #3** | Cross-warp race: lane 0 writes `m_smem`, all read | `__syncthreads()` after online softmax | ❌ max_err 4.1 → 2.4 (marginal) |

**Error Magnitudes**:
- Stage-2 (baseline): max_err 0.06-0.16 (FP8 quantization noise)
- Stage-3B (fused): max_err 2.4-4.1 (**40-60× worse** than Stage-2)

**Conclusion**: The fused softmax implementation has a fundamental algorithmic bug (not a minor race or masking issue).

---

## 🔍 Root Cause Hypotheses (Unresolved)

### Top Suspects
1. **WMMA Fragment Element Distribution**  
   - Not all lanes may see all 16×16 elements → incomplete reductions
   - LUT might be incorrect or misused in the masked reduction loops
   - **Test needed**: Verify `WMMA_ACCUM_LUT` covers all (row, col) pairs exactly once

2. **Warp Reduction Over Sparse Data**  
   - Masking `if (rr == r && cc < kv_local)` leaves many lanes with `-INFINITY`
   - `warp_reduce_max(-INFINITY)` behavior may be incorrect
   - **Test needed**: Compare per-row max against Stage-2 scalar loop

3. **Online Softmax Update Math Error**  
   - The register-based rescale/accumulate logic may differ from Stage-2's SMEM-based path
   - `U_smem[r_glob][d] *= rescale` may have visibility issues across warps
   - **Test needed**: Step-by-step comparison of `m_new`, `l_add`, `rescale` vs Stage-2

### Ruled Out
- ❌ Partial tile masking (fixed in attempt #1, no change)
- ❌ Stale `sP` data (fixed in attempt #2, no change)
- ❌ Simple cross-warp sync (fixed in attempt #3, minimal improvement)

---

## 📈 Performance Context

### Stage Evolution (Mission Shape: B=2, H=8, S=256, D=64)

| Stage | Description | p50 (μs) | vs Stage-1 | vs Stage-2 | Status |
|-------|-------------|----------|------------|------------|--------|
| **Stage-1** | `cp.async` double-buffer K/V | 761 | Baseline | — | ✅ Merged |
| **Stage-2** | Stage-1 + WMMA P·V | **656** | +13.8% | Baseline | ✅ Merged (`v2.0-stage2-wmma-pv`) |
| **Stage-3A** | Stage-2 + reuse `sS` for `sP` | 655 | +13.9% | +0.2% | ⚠️ Marginal (valid negative) |
| **Stage-3B** | Stage-2 + fused softmax in registers | ❌ N/A | — | — | ❌ Correctness failed |

**Key Insight**: Stage-2 (656 μs) is a **solid, validated baseline**. Further optimization requires a different approach than register-level fusion.

---

## 🎓 Lessons Learned

### 1. **"GREEN before FAST" is Non-Negotiable**
- Spent 4 hours debugging a kernel with **0% correctness** → no path to optimization
- Should have caught this earlier with smaller unit tests (e.g., single 16×16 tile)

### 2. **Complex Transforms Need Phased Validation**
- Moving from SMEM-based softmax (Stage-2) to register-based (Stage-3B) is a **major refactor**, not a simple optimization
- Should have validated intermediate steps: QK^T in regs (✓), max in regs (✓), softmax in regs (❌)

### 3. **WMMA Fragment Layout is Non-Trivial**
- The `WMMA_ACCUM_LUT` mapping is architecture-specific and lane-dependent
- Reduction patterns (`if (rr == r && cc < kv_local)`) must account for fragment distribution across 32 lanes
- **Recommendation**: Write standalone unit test for WMMA reductions before integrating

### 4. **"Valid Negative" is a Valuable Result**
- XOR swizzle (Step-2): -6.1% → bank conflicts not the bottleneck ✅
- Fused softmax (Step-3B): 0/6 tests → register fusion too complex for current design ✅
- Both results inform future work

---

## 📁 Artifacts

### Code
- **Kernel**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`
  - Lines 451-550: Fused softmax (non-cp.async path)
  - Lines 826-925: Fused softmax (cp.async path)
- **LUT**: `cudadent42/bench/kernels/wmma16x16_accum_lut.h`
- **Build**: `tasks/fp8_sdpa_stage_c_wmma/build.py`

### Documentation
- **Implementation Plan**: `STAGE3_FUSION_FULL_PLAN.md`
- **Progress Checkpoint**: `STAGE3_PROGRESS_CHECKPOINT.md`
- **Hotfix Status**: `STAGE3B_HOTFIX_STATUS.md` (3 fix attempts detailed)
- **This Summary**: `SESSION_STAGE3_COMPLETE_OCT20_2025.md`

### Commits
```
62ca01e  docs(stage3): Comprehensive implementation plan
742da83  feat(stage3): Add feature flags infrastructure
... (15 commits total)
9c4136b  docs(stage3): Append Session 4 summary (3 hotfix attempts, abandoned)
```

### Results
```
results/2025-Stage3B-Fused-Validation/
  ├── fix-kvmask-spzero/
  │   ├── .build_s2.log, .build_s3b.log
  │   ├── .corr_s2.log, .corr_s3b.log
  │   └── (6 correctness tests, all FAILED)
  └── README.md
```

---

## 🚀 Recommended Next Steps

### Option A: Stage-4 (3-Stage cp.async Pipeline)
**Description**: Extend 2-stage K/V prefetch to 3-stage for long sequences  
**Target**: +5-10% speedup over Stage-2 (≤590 μs)  
**Effort**: 4-6 hours  
**Risk**: Low (no algorithmic changes, just pipelining)  
**Rationale**: Proven technique; FlashAttention-3 uses similar approach

**Changes**:
```cuda
#if USE_CP_ASYNC_3STAGE
__shared__ sK_u8[3][TILE_N][D_PAD];  // 3-stage ring buffer
__shared__ sV_u8[3][TILE_N][D_PAD];
// Prefetch tile t+2 while computing t
#endif
```

**Gates**:
- PTXAS: ≤120 regs, ≤52 KB SMEM, 0 spills
- Correctness: 9/9 tests (bit-exact with Stage-2)
- Perf: p50 ≤ 590 μs on mission shape (≥+10%)

---

### Option B: Stage-5 (Redesigned Fused Softmax)
**Description**: Clean-slate fused softmax with unit-tested building blocks  
**Approach**:
1. **Phase 1**: Unit test WMMA reductions (max, sum) in isolation (2h)
2. **Phase 2**: Implement partial fusion: QK^T in regs, softmax in SMEM, P·V with WMMA (2h)
3. **Phase 3**: Full fusion only if Phase 2 validates (4h)

**Target**: +15-25% speedup (≤557-525 μs)  
**Effort**: 8-12 hours  
**Risk**: Medium-High (same complexity as Stage-3B)  
**Rationale**: If successful, highest payoff; but requires careful phased validation

---

### Option C: Pivot to Different Optimization
**Alternatives**:
- **Persistent CTAs**: Keep blocks resident across batches (for inference servers)
- **Kernel Fusion**: Fuse layernorm or QKV projection with SDPA
- **Multi-Query/Grouped-Query Attention**: Optimize for MQA/GQA (fewer KV heads)
- **FP8 Tensor Core Native**: Replace simulated FP8 with real E4M3/E5M2 CUDA types

**Rationale**: Explore orthogonal optimizations; fused softmax may not be the critical path

---

## ⚖️ Decision Recommendation

**Immediate (Next Session)**:
1. ✅ Close `feat/stage3-fusion-full` branch (keep for historical reference)
2. ✅ Checkout `main` (Stage-2 already merged as `v2.0-stage2-wmma-pv`)
3. ✅ Document Stage-3 as "Valid Negative Result" in `README.md`

**Short-Term (Next 1-2 Sessions)**:
- **Option A (Recommended)**: Implement Stage-4 (3-stage cp.async) for quick, low-risk win
- **Option C (Alternative)**: Pivot to persistent CTAs or kernel fusion

**Long-Term (Future Work)**:
- Revisit Stage-5 (redesigned fused softmax) with unit-tested WMMA reductions
- Or accept Stage-2 (656 μs) as production-ready and shift focus to higher-level optimizations

---

## 📊 Final Metrics

| Metric | Stage-2 (Baseline) | Stage-3B (Fused) | Verdict |
|--------|-------------------|------------------|---------|
| **PTXAS** | 96 regs, 37 KB SMEM | 83 regs, 35 KB SMEM | ✅ IMPROVED |
| **Correctness** | 6/6 tests (max_err 0.06-0.16) | 0/6 tests (max_err 2.4-4.1) | ❌ FAILED |
| **Performance** | 656 μs (validated) | N/A (blocked) | — |
| **Development Time** | 8 hours | 12 hours (+50%) | — |
| **Risk-Adjusted ROI** | High (merged, tagged) | Negative (abandoned) | — |

---

## 🎯 Key Takeaway

**Stage-3B fused softmax was a high-risk, high-reward attempt that didn't pan out.** After 3 systematic debugging attempts with 0 progress, the correct engineering decision is to:
1. **Document** the attempt (✅ done)
2. **Preserve** the code for future reference (✅ branch retained)
3. **Move forward** with lower-risk optimizations (⏭ Stage-4 or pivot)

**This is a successful application of the EvoEngineer "fail fast" principle.** Not every optimization attempt succeeds, and recognizing when to cut losses is as valuable as finding the next speedup.

---

**Session End**: 2025-10-20 20:30 UTC  
**Next Action**: User decision on Stage-4 (3-stage cp.async) vs pivot  
**Status**: ✅ **DOCUMENTED & CLOSED**

---

## 🔗 Related Documents
- `STAGE2_GPU_VALIDATION_SUMMARY.md` - Stage-2 baseline (656 μs, merged)
- `STAGE3_FUSION_FULL_PLAN.md` - Original Stage-3 plan
- `STAGE3B_HOTFIX_STATUS.md` - Detailed debugging log
- `SESSION_STAGE1_STAGE2_COMPLETE.md` - Previous milestone (4.4× speedup achieved)

---

**Signed off by**: AI Assistant (Claude Sonnet 4.5)  
**Reviewed by**: (Pending user review)

