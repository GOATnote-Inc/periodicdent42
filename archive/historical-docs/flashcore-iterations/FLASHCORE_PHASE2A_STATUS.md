# FlashCore Phase 2A Status Report

**Date**: October 22, 2025  
**Session**: Extended Phase 2A Implementation Attempt  
**Status**: **BLOCKED** - Fundamental SMEM Constraints

---

## 🎯 **Original Goal**

**Phase 2A**: 64×64 tiles + FP32 P  
**Expected**: 279 → 120 μs (2× speedup), error 0.34 → <0.10 (6× improvement)  
**Confidence**: 80-85% (user-provided estimate)

---

## 📊 **Current Status**

### **What Works** ✅
- **32×32 tiles (FP16 P)**: 279 μs, 0.34 error, 91 regs, 0 spills, 39KB SMEM
- Build quality excellent
- Correctness reasonably good (93% reduction from initial 7.87 error)
- Performance: 5.0× speedup from baseline (1398 → 279 μs)

### **What Doesn't Work** ❌
1. **32×32 tiles + FP32 P**: 52.5 KB SMEM > 48 KB limit
2. **64×64 tiles (any variant)**: Multiple issues:
   - Illegal memory access (CUDA invalid argument)
   - NaN outputs (buffer overlap)
   - WMMA fragment mapping complexity
   - SMEM requirements (75-107 KB depending on approach)

---

## 🔬 **Technical Deep Dive**

### **32×32 Tile SMEM Breakdown**

| Component | FP16 P | FP32 P | Difference |
|-----------|--------|--------|------------|
| sQ | 5 KB | 5 KB | - |
| sKT | 5 KB | 5 KB | - |
| sV | 5 KB | 5 KB | - |
| sS_f32 | 4 KB | 4 KB | - |
| **sP** | **2 KB** | **4 KB** | **+2 KB** |
| **sP_fp16** | **-** | **2 KB** | **+2 KB** |
| m_smem | 0.125 KB | 0.125 KB | - |
| l_smem | 0.125 KB | 0.125 KB | - |
| U_smem | 10 KB | 10 KB | - |
| sU_part | 8 KB | 8 KB | - |
| **TOTAL** | **39.25 KB** | **43.25 KB** | **+4 KB** |
| **Status** | ✅ Fits | ❌ **52.5 KB actual!** | **+13 KB unexpected** |

**Mystery**: PTXAS reports 52.5 KB, but manual calculation shows 43.25 KB. Likely due to:
- Alignment padding (16-byte alignment × many arrays)
- Bank conflict avoidance padding in `HEAD_DIM_SMEM` (64 → 80)
- Compiler-inserted temporary buffers

### **64×64 Tile Attempts**

**Attempt 1**: Per-warp partials (atomic-free)
- SMEM: 107 KB (exceeded 96 KB default limit)
- Result: "CUDA error: invalid argument"

**Attempt 2**: Buffer reuse (`sScores` → `sU_part`)
- SMEM: 82.5 KB (should fit with opt-in)
- Result: NaN output (buffer overlap with `sP_fp16` still in use)

**Attempt 3**: Simplified with atomics
- SMEM: 74.5 KB (should fit!)
- Result: "CUDA error: invalid argument"
- Likely issue: Incorrect WMMA fragment mapping in atomic accumulation loop

---

## 🧩 **Root Causes**

### **1. SMEM Pressure on 32×32 Tiles**
- FP32 P requires +4 KB (sP: 2→4 KB, sP_fp16: +2 KB)
- Compiler alignment/padding adds ~9 KB overhead
- **No room to grow** without exceeding 48 KB default limit

### **2. 64×64 Tile Complexity**
- **8 warps** (4×2 grid) vs 4 warps (2×2) for 32×32
- **Atomic-free reduction** requires 32 KB `sU_part` (8 warps × 16 rows × 64 cols × 4B)
- **WMMA fragment layout** for 8 threads × 8 elements is implementation-defined and non-trivial
- **Buffer reuse** requires careful temporal separation analysis

### **3. L4 SMEM Limits**
- Default: 48 KB per block
- Opt-in max: 164 KB per block (requires `cudaFuncSetAttribute`)
- Our attempts to opt-in failed silently or crashed

---

## 🛣️ **Path Forward Options**

### **Option A: Accept 32×32 FP16 P as Phase 2A Baseline** (Conservative)
**Current**: 279 μs, 0.34 error  
**Action**: Document current state, proceed to Phase 2B (cp.async)  
**Pros**:
- ✅ Works reliably
- ✅ 5× speedup from baseline
- ✅ Error acceptable for many applications (< 1% relative)
- ✅ Can iterate on performance first, precision later

**Cons**:
- ❌ Error not <0.10 (user's Phase 2A goal)
- ❌ Misses "fix error + performance" in one shot
- ❌ 279 μs still 7× away from <40 μs target

**Next Steps**:
1. Commit current 32×32 FP16 P kernel
2. Implement Phase 2B: cp.async (target: 279 → 140 μs)
3. Implement Phase 2C: Micro-optimizations (target: 140 → 70 μs)
4. **Then** revisit precision with 64×64 tiles or double-precision softmax

**Timeline**: 4-6 hours, 80% confidence  
**Outcome**: ~70 μs, 0.34 error (still 1.75× away from <40 μs target)

---

### **Option B: Fix 64×64 Atomics Systematically** (Moderate Risk)
**Goal**: Get 64×64 tiles working with atomics (simple, proven pattern)  
**Action**: Fix WMMA fragment→global index mapping, verify correctness  

**Pros**:
- ✅ 4× more work per block → better amortization
- ✅ Room for FP32 P (can allocate 90-100 KB SMEM)
- ✅ Performance gain from larger tiles (279 → ~160 μs estimated)
- ✅ Atomic overhead small on Ada L4 (L2-cached)

**Cons**:
- ❌ WMMA accumulator layout is non-trivial
- ❌ Already spent 3+ hours debugging
- ❌ Risk of more "invalid argument" errors

**Next Steps**:
1. Study WMMA accumulator layout spec for sm_89
2. Implement correct fragment→global mapping
3. Test with `DEBUG_QK_ONLY` and `DEBUG_PV_ONLY` gates
4. Add comprehensive error checking

**Timeline**: 6-10 hours, 60% confidence  
**Outcome**: ~160 μs, <0.10 error (still 4× away from <40 μs target)

---

### **Option C: Hybrid 32×32 + Dynamic SMEM** (High Risk, Experimental)
**Goal**: Request >48 KB SMEM for 32×32 tiles to fit FP32 P  
**Action**: Use `cudaFuncSetAttribute` to opt-in to 64 KB SMEM  

**Pros**:
- ✅ Minimal code changes to working kernel
- ✅ L4 supports up to 164 KB per block
- ✅ FP32 P fits with 52.5 KB

**Cons**:
- ❌ Reduces occupancy (fewer blocks per SM)
- ❌ May hurt performance despite fixing error
- ❌ Our previous opt-in attempts failed mysteriously
- ❌ Doesn't address performance gap to <40 μs

**Next Steps**:
1. Add `cudaFuncSetAttribute` to 32×32 kernel
2. Verify SMEM allocation succeeds
3. Test performance and occupancy impact

**Timeline**: 2-4 hours, 40% confidence  
**Outcome**: ~300 μs (slower!), <0.10 error (7.5× away from <40 μs target)

---

### **Option D: Skip Phase 2A, Jump to cp.async** (Aggressive)
**Goal**: Focus on performance, accept error=0.34 for now  
**Action**: Implement 2-stage pipelined K/V loads with `__pipeline_memcpy_async`  

**Pros**:
- ✅ Directly addresses performance bottleneck
- ✅ cp.async proven to give 2-2.5× speedup
- ✅ Error can be fixed later with double-precision softmax
- ✅ Aligns with "performance first" philosophy

**Cons**:
- ❌ Deviates from user's Phase 2A plan
- ❌ Error remains 0.34
- ❌ More complex debugging if cp.async has issues
- ❌ May hit new SMEM pressure with double-buffered K/V

**Next Steps**:
1. Implement 2-stage cp.async for K/V tiles
2. Profile memory vs compute overlap
3. Measure speedup (target: 279 → 120 μs)

**Timeline**: 8-12 hours, 70% confidence  
**Outcome**: ~120 μs, 0.34 error (3× away from <40 μs target, but momentum!)

---

## 💡 **Recommendation**

### **Option A + D Hybrid** (Pragmatic)

**Phase 2A.5** (Compromise):
1. Accept 32×32 FP16 P as "Phase 2A baseline" (279 μs, 0.34 error)
2. Implement cp.async immediately (Phase 2B)
3. Target: 279 → 120 μs (2.3× speedup)
4. **Then** revisit precision with either:
   - 64×64 tiles (if time permits and cp.async succeeded)
   - Or double-precision softmax (simpler, 5-10% perf cost)

**Rationale**:
- **Performance >> Precision** for current gap (279 μs vs <40 μs target)
- We're 7× away from target; error is secondary
- cp.async is proven, well-documented, lower risk than 64×64 debugging
- User's <40 μs target requires aggressive performance optimization
- Precision can be fixed later with minimal perf cost

**Timeline**: 8-10 hours total
**Confidence**: 75-80% for cp.async success, 60% for final <40 μs with all optimizations

---

## 📈 **Projected Path to <40 μs**

### **Realistic Timeline** (Option A + D)

```
Phase 2A.5 (Accept current): 279 μs, 0.34 error ✅
Phase 2B (cp.async):         279 → 110 μs (2.5×), 0.34 error
Phase 2C (Micro-opt):        110 → 60 μs (1.8×), 0.34 error
Phase 2D (Launch bounds):    60 → 45 μs (1.3×), 0.34 error
Phase 2E (64×64 + FP32):     45 → 35 μs (1.3×), <0.10 error ✅✅

Total: 12-18 hours, 70% overall confidence
```

### **Aggressive Timeline** (Option B or D alone)

```
Option B (64×64 atomics):
Phase 2A: 279 → 160 μs, <0.10 error (if successful)
Phase 2B: 160 → 70 μs (cp.async)
Phase 2C: 70 → 40 μs (micro-opt) ✅
Total: 14-22 hours, 60% confidence

Option D (cp.async first):
Phase 2B: 279 → 120 μs, 0.34 error
Phase 2C: 120 → 60 μs (micro-opt)
Phase 2D: 60 → 38 μs (64×64 + FP32) ✅
Total: 16-24 hours, 65% confidence
```

---

## 🏁 **Decision Point**

**User must choose**:
1. **Conservative**: Accept current 32×32, iterate on performance (Option A)
2. **Moderate**: Debug 64×64 atomics thoroughly (Option B)
3. **Experimental**: Try dynamic SMEM for 32×32 (Option C)
4. **Aggressive**: Skip to cp.async, fix precision later (Option D)
5. **Hybrid**: A + D (Recommended)

**My recommendation**: **Hybrid (A + D)** for best risk/reward ratio.

---

## 📁 **Artifacts**

**Working Code**:
- ✅ `flashcore/kernels/flashcore_fused_wmma.cu` (32×32, FP16 P, 279 μs, 0.34 error)
- ✅ `flashcore/kernels/flashcore_fused_wmma_64x64.cu` (64×64 attempts, all broken)

**Test Scripts**:
- ✅ `flashcore/test_fused.py` (32×32)
- ✅ `flashcore/test_64x64.py` (64×64)

**Documentation**:
- ✅ This status report
- ✅ `FLASHCORE_EPIC_SESSION_COMPLETE.md` (atomic-free 32×32 success)
- ✅ `FLASHCORE_ULTIMATE_SESSION_STATUS.md` (register pressure fix)
- ✅ `FLASHCORE_FP32_P_ATTEMPT_REPORT.md` (Phase 1 FP32 P learnings)

---

**Status**: Awaiting user direction  
**Next Session**: Implement chosen option  
**Time Invested**: ~6 hours (Phase 2A attempts)  
**Remaining Budget**: ~42 hours (Phase 2B-E + polish)

**LET'S CHOOSE THE PATH AND ACHIEVE <40 μs! 🚀**

