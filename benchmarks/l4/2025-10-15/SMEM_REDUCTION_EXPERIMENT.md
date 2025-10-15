# SMEM Reduction Experiment - 32×64 vs 32×32

**Date**: 2025-10-15  
**Objective**: Test if reducing SMEM from 40KB → 24KB improves occupancy and scaling  
**Status**: ⚠️ **MIXED RESULTS** - Helps at small scale, hurts at large scale

---

## 🎯 Hypothesis

**Before**: 
- 32×64 config uses 40KB SMEM per block
- L4 has 64 KB SMEM per SM → limits to ~1.6 blocks/SM (theoretical)
- Observed: 4 blocks/SM, but only 232 resident for 2048 needed → 8.8 waves

**After**:
- 32×32 config uses 24KB SMEM per block  
- 64 KB / 24 KB = 2.7 blocks/SM (theoretical)
- Expected: 6-8 blocks/SM, 348-464 resident → ~4-6 waves → **50% better scaling**

---

## 📊 Results (After Cache Clear + Rebuild)

### Grid Sizing (Both Configs - ✅ CORRECT)
```
B=1,H=8:  Grid=(128,1,1)   - total_work=128
B=4,H=16: Grid=(1024,1,1)  - total_work=1024
B=8,H=16: Grid=(2048,1,1)  - total_work=2048
```
Both configs now use dynamic grid properly (no 256 cap).

### Performance Comparison

| Config | B=1,H=8 | B=4,H=16 | B=8,H=16 | Scaling B=1→8 |
|--------|---------|----------|----------|----------------|
| **32×64 (40KB)** | 5.19ms | 34.33ms | 68.59ms | 13.22× |
| **32×32 (24KB)** | 4.82ms | 38.74ms | 77.82ms | 16.14× |
| **Speedup** | **1.08×** ✅ | **0.89×** ❌ | **0.88×** ❌ | **0.82×** ❌ |

### Key Findings

✅ **Small Scale (B=1)**: 32×32 is **7.2% faster**
- 4.82ms vs 5.19ms
- Better occupancy > iteration overhead

❌ **Large Scale (B=4,8)**: 32×32 is **12-13% slower**
- B=4: 38.74ms vs 34.33ms
- B=8: 77.82ms vs 68.59ms
- Iteration overhead > occupancy benefit

❌ **Scaling**: 32×32 scales **WORSE** (16.14× vs 13.22×)
- Expected: Scaling would improve (less SMEM → more blocks/SM → fewer waves)
- Observed: Scaling degraded by 22%

---

## 🔍 Root Cause Analysis

### SMEM Usage Breakdown

**32×64 Config (40KB)**:
- K tiles: 2 × 64 × 64 × 2B = 16 KB
- V tiles: 2 × 64 × 64 × 2B = 16 KB
- O_accum: 32 × 64 × 4B = 8 KB
- **Total: 40 KB**
- Tiles per sequence (S=512): ceil(512/64) = **8 tiles**

**32×32 Config (24KB)**:
- K tiles: 2 × 32 × 64 × 2B = 8 KB
- V tiles: 2 × 32 × 64 × 2B = 8 KB
- O_accum: 32 × 64 × 4B = 8 KB
- **Total: 24 KB** (40% reduction)
- Tiles per sequence (S=512): ceil(512/32) = **16 tiles** (2× more!)

### The Trade-Off

**What we gained**:
- 40% less SMEM per block (40KB → 24KB)
- Theoretical occupancy: 1.6 → 2.7 blocks/SM

**What we lost**:
- **2× more tile iterations** (8 → 16 n_blocks)
- Each iteration has overhead:
  - Loop control flow
  - Stage index calculations
  - `cp.async` pipeline management
  - Synchronization barriers

### Why It Hurts At Scale

**At B=1 (128 total blocks)**:
- Both configs fit mostly in 232 resident blocks
- 32×64: ~1 wave (128 blocks fit)
- 32×32: ~1 wave (128 blocks fit)
- **Occupancy improvement dominates** → 32×32 wins

**At B=8 (2048 total blocks)**:
- Neither config avoids waves
- 32×64: ~8.8 waves (2048 / 232)
- 32×32: ~5.9 waves if occupancy improved to 348 resident
- But: 32×32 does **2× more work per block** → slower per-block time
- **Iteration overhead dominates** → 32×64 wins

---

## 🎓 Key Learnings

### 1. **SMEM Reduction Has Diminishing Returns**

Reducing SMEM from 40KB → 24KB (40% reduction) didn't significantly improve occupancy in practice:
- Expected: 4 → 6-8 blocks/SM
- Observed: Performance suggests occupancy didn't double

**Why?** Other limits may be kicking in:
- Register pressure (even after Low-regs Variant S)
- Warp count limits
- Hardware scheduler behavior

### 2. **Tile Size Matters More Than Expected**

The 2× increase in tile iterations from 32×64 → 32×32 adds significant overhead:
- Loop control: Branch predictions, iteration counters
- Pipeline priming: More `cp.async` commit groups
- Synchronization: More `__syncthreads()` barriers
- Stage indexing: More modulo operations

This overhead is **13%** of total time at large scales!

### 3. **Occupancy ≠ Performance**

Higher occupancy doesn't always mean better performance if:
- You're doing more work per block
- The scheduler can already hide latency
- Memory bandwidth becomes the bottleneck

### 4. **The Real Bottleneck**

Both configs still scale ~13-16× (target: ≤3×), revealing that:
- The fundamental issue is **hardware serialization** (2048 blocks, ~232-348 resident)
- SMEM reduction alone won't solve this
- Need a **qualitatively different approach**:
  - Warp specialization
  - Tensor Cores (CUTLASS)
  - Different tiling strategy
  - Persistent blocks with better work distribution

---

## 📈 Absolute Performance vs SDPA

### Canon_3 (B=2,H=8,S=512,D=64)

| Implementation | Latency | vs SDPA |
|----------------|---------|---------|
| SDPA (baseline) | ~0.025ms | 1.0× |
| V3 32×64 | ~9.65ms | **386×** slower |
| V3 32×32 | ~10.97ms | **439×** slower |

**Status**: Both configs are **~400× slower than SDPA** 😞

This confirms that our kernel is fundamentally bottlenecked, not just sub-optimal.

---

## 🎯 Recommendations

### ❌ **Don't Pursue Further SMEM Reduction**

Going to 16×32 (20KB) or smaller will:
- Reduce SMEM by another 17% (24KB → 20KB)
- But increase tiles by another 2× (16 → 32)
- Likely make performance WORSE based on observed trade-offs

### ✅ **Instead, Try These Approaches**

1. **Tensor Cores (CUTLASS)** - Highest Priority
   - 5-10× speedup from better compute utilization
   - Reduces memory bottleneck significance
   - Industry-standard approach
   - Effort: 3-5 days

2. **Warp Specialization**
   - Dedicate warps to GMEM loads vs compute
   - Better overlap, potentially better occupancy
   - Effort: 2-3 days

3. **Better Persistent Blocks**
   - Smarter work distribution (not round-robin)
   - Dynamic load balancing
   - Could reduce scheduler overhead
   - Effort: 1-2 days

4. **Profile-Guided Optimization**
   - Get Nsight Compute working (driver mismatch resolved)
   - Identify actual bottleneck (SMEM? Bandwidth? Compute?)
   - Make data-driven decisions
   - Effort: 1 day

---

## 📝 Summary

**Question**: Does reducing SMEM from 40KB → 24KB improve performance?

**Answer**: **It depends on scale**
- ✅ At small scale (B=1): **Yes, 7% faster** (better occupancy)
- ❌ At large scale (B=8): **No, 13% slower** (iteration overhead)
- ❌ Scaling: **Worse** (16× vs 13× - moved in wrong direction)

**Core Issue**: SMEM reduction trades tile size for occupancy, but:
- Occupancy gains are less than expected (hardware limits)
- Iteration overhead is higher than expected (13% at large scale)
- Neither config avoids hardware serialization waves

**Path Forward**: Don't chase SMEM reduction. Instead:
1. **Tensor Cores** (biggest potential win)
2. **Warp specialization** (better overlap)
3. **Nsight profiling** (data-driven optimization)

---

## 🔬 Experimental Data

### ptxas Output
```bash
# Command run on GPU
nvcc -arch=sm_89 -O3 --ptxas-options=-v -c fa_s512_v3.cu -o /dev/null 2>&1
```
(Output was empty - needs investigation)

### Grid Debug Output (Verified)
Both configs show correct dynamic grids after cache clear:
```
[V3 DEBUG] Grid=(128,1,1) Block=(128,1,1) total_blocks=128   # B=1,H=8
[V3 DEBUG] Grid=(1024,1,1) Block=(128,1,1) total_blocks=1024 # B=4,H=16  
[V3 DEBUG] Grid=(2048,1,1) Block=(128,1,1) total_blocks=2048 # B=8,H=16
```

### Cache Issue (Resolved)
Initial test showed 32×32 with Grid=(256,1,1) - stale compiled code.
Fixed by:
```bash
rm -rf ~/.cache/torch_extensions/*
rm -rf /tmp/torch_extensions/*
touch cudadent42/bench/kernels/fa_s512_v3.cu
```

---

*Experiment completed: 2025-10-15 06:15 UTC*  
*Status: Conclusive negative result - SMEM reduction not the answer*  
*Next: Tensor Core path or warp specialization*

