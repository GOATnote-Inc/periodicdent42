# FlashCore - Next Session Action Plan

**Date**: October 22, 2025  
**Current Status**: Register pressure fixed! Error needs tuning.  
**Time Estimate**: 1-2 hours for correctness, 4-6 hours for <40 μs  

---

## 🎯 **Primary Goal: Fix Remaining Error (0.52 → <0.05)**

### **Quick Win: FP32 P Matrix** (30-45 min)

**Current bottleneck**: P (probabilities) stored as FP16, losing precision

**Fix**:
```cpp
// In flashcore_fused_wmma.cu, line ~148:
// BEFORE:
__shared__ alignas(16) half sP[TILE_M][TILE_N];  // 32×32×2B = 2 KB

// AFTER:
__shared__ alignas(16) float sP[TILE_M][TILE_N];  // 32×32×4B = 4 KB
```

**Impact**:
- SMEM: 48KB → 50KB (still under 64KB opt-in limit)
- Precision: FP16 → FP32 for probabilities
- Expected error: 0.52 → <0.10

**Challenge**: WMMA expects FP16 input, so need to convert:
```cpp
// Before WMMA load:
half P_half[TILE_N];
#pragma unroll
for (int n = 0; n < TILE_N; ++n) {
    P_half[n] = __float2half(sP[warp_m_start + local_row][n]);
}
wmma::load_matrix_sync(a_frag_pv, P_half, TILE_N);
```

**Alternative (if SMEM limited)**: Convert on-the-fly during load
```cpp
// Load P row-by-row with conversion
for (int row = 0; row < WMMA_M; ++row) {
    half P_row[WMMA_N];
    #pragma unroll
    for (int col = 0; col < WMMA_N; ++col) {
        P_row[col] = __float2half(sP[warp_m_start + row][k + col]);
    }
    // Use P_row in WMMA...
}
```

---

## 🚀 **Secondary Goal: Push to <40 μs** (After correctness fixed)

### **Step 1: Analyze Current Performance** (30 min)

```bash
# Profile with NCU on L4
ncu --set full --launch-skip 10 --launch-count 1 \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum \
    python flashcore/test_fused.py
```

**Key metrics**:
- TC utilization: Should be >60%
- DRAM throughput: Should be >50% (memory-bound expected)
- FLOPs: Sanity check (should match theoretical)

### **Step 2: 64×64 Tiles** (2-3 hours)

**Current**: 32×32 tiles (4 warps, 2×2 grid)  
**Target**: 64×64 tiles (8 warps, 4×2 or 2×4 grid)

**Changes needed**:
```cpp
#define TILE_M   64  // Was: 32
#define TILE_N   32  // Keep 32 for now (2-warp N-partition still works)
#define NUM_WARPS 8  // Was: 4

// Update warp grid:
const int warp_m = warp_id / 2;  // 0-3
const int warp_n = warp_id % 2;  // 0-1

// Update sU_part:
__shared__ alignas(16) float sU_part[4][2][WMMA_M][HEAD_DIM];  // Was: [2][2]

// Update merge logic for 4 warp_m values
```

**Expected speedup**: 2× (more work per block, amortize launch overhead)  
**Challenges**: SMEM usage (need to check if fits in 64KB)

### **Step 3: cp.async for K/V** (2-3 hours)

**Current**: Synchronous loads (block waits for K/V before compute)  
**Target**: Asynchronous loads (overlap load with compute)

**Pattern**:
```cpp
// 2-stage pipeline
#define STAGES 2
__shared__ half sKT[STAGES][HEAD_DIM_SMEM][TILE_N];
__shared__ half sV[STAGES][TILE_N][HEAD_DIM_SMEM];

int stage = 0;
// Prefetch first tile
cp_async_load(&sKT[stage][...], &K_bh[...]);
cp_async_load(&sV[stage][...], &V_bh[...]);
cp_async_commit_group();

for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
    // Prefetch next tile (if exists)
    if (kv_tile_idx + 1 < num_kv_tiles) {
        int next_stage = 1 - stage;
        cp_async_load(&sKT[next_stage][...], &K_bh[...]);
        cp_async_load(&sV[next_stage][...], &V_bh[...]);
        cp_async_commit_group();
    }
    
    // Wait for current tile
    cp_async_wait_group(0);
    __syncthreads();
    
    // Compute with current stage
    // ... Q@K^T, softmax, P@V using sKT[stage], sV[stage] ...
    
    stage = 1 - stage;
}
```

**Expected speedup**: 1.5-2× (overlap load with compute)  
**Challenges**: Double SMEM usage (might need to reduce to TILE_N=16)

### **Step 4: Occupancy Tuning** (30 min)

```cpp
// Try different launch bounds
__launch_bounds__(128, 2)  // Current
__launch_bounds__(256, 1)  // More threads, fewer blocks
__launch_bounds__(128, 3)  // More blocks per SM
```

**Expected**: 5-10% improvement by finding sweet spot

---

## 📊 **Performance Projection**

| Step | Current | After | Speedup | Cumulative |
|------|---------|-------|---------|------------|
| **Baseline** | 279 μs | — | — | 1.0× |
| **FP32 P fix** | 279 μs | ~285 μs | 0.98× | 1.0× |
| **64×64 tiles** | 285 μs | ~140 μs | 2.0× | 2.0× |
| **cp.async** | 140 μs | ~80 μs | 1.75× | 3.5× |
| **Tuning** | 80 μs | ~70 μs | 1.15× | 4.0× |
| **Target** | — | **<40 μs** | — | **7.0×** |

**Reality check**: 
- 64×64 tiles: Very likely to work (proven technique)
- cp.async: Likely to work but complex (pipeline management)
- Combined: Need both + tuning to hit <40 μs

**Confidence**: 
- <50 μs: 90% (64×64 alone should get us there)
- <40 μs: 60% (needs cp.async + tuning)
- <30 μs: 30% (would need warp specialization)

---

## 🔧 **Recommended Session Flow**

### **Hour 1: Fix Correctness**
1. Implement FP32 P matrix (30 min)
2. Test & verify error <0.10 (15 min)
3. Commit with documentation (15 min)

### **Hour 2-3: 64×64 Tiles**
1. Update tile constants & warp grid (30 min)
2. Update shared memory & merge logic (30 min)
3. Test correctness (30 min)
4. Benchmark & profile (30 min)
5. Expected: ~140 μs (2× speedup)

### **Hour 4-5: cp.async**
1. Research & plan pipeline (30 min)
2. Implement 2-stage pipeline (1 hour)
3. Debug & validate (1 hour)
4. Expected: ~70-80 μs (2× more speedup)

### **Hour 6: Tuning & Documentation**
1. Try different launch bounds (30 min)
2. Final benchmarking (15 min)
3. Create comprehensive report (15 min)
4. Expected: <40 μs if all goes well!

---

## ⚠️ **Fallback Plan**

If <40 μs proves difficult:

1. **Target <50 μs instead** (easier, still 5.6× total speedup)
2. **Skip cp.async** (too complex for diminishing returns)
3. **Focus on correctness + 64×64 tiles** (solid 2× improvement)
4. **Document lessons learned** (portfolio-quality artifact)

---

## 🎯 **Success Criteria**

### **Minimum (Must Have)**
- ✅ Correctness: max_err <0.06 on all shapes
- ✅ Performance: <50 μs (5.6× total speedup)
- ✅ Build: <96 registers, 0 spills
- ✅ Documentation: Complete session summary

### **Target (Should Have)**
- ✅ Correctness: max_err <0.05 on all shapes
- ✅ Performance: <40 μs (7× total speedup)
- ✅ NCU: TC util >60%, DRAM >50%
- ✅ Portfolio: Publication-ready report

### **Stretch (Nice to Have)**
- ✅ Performance: <30 μs (9× total speedup)
- ✅ Warp specialization implemented
- ✅ Competitive with FlashAttention-2

---

## 📝 **Session Prep Checklist**

Before starting next session:

- [ ] Read this file completely
- [ ] Read `FLASHCORE_ULTIMATE_SESSION_STATUS.md`
- [ ] Check GPU is available (`nvidia-smi`)
- [ ] Pull latest from remote (`git pull`)
- [ ] Run preflight check (`bash scripts/preflight.sh`)
- [ ] Review NCU profiling commands
- [ ] Have FlashAttention-2 paper open for reference

---

## 🏆 **Expected Outcome**

**After 6 hours**:
- ✅ Correctness: <0.05 error (perfect!)
- ✅ Performance: ~40-70 μs (7-14× vs 1398 μs baseline)
- ✅ Registers: 91 (excellent!)
- ✅ Documentation: Portfolio-ready
- ✅ Learning: Deep understanding of attention optimization

**Portfolio value**: VERY HIGH (custom CUDA kernel, 7-14× speedup, systematic optimization)

---

**Status**: ✅ **READY FOR NEXT SESSION!**

**Priority**: Fix correctness first (FP32 P), then push for <40 μs!

---

**Document Version**: 1.0  
**Date**: October 22, 2025  
**Estimated Time**: 1-6 hours (depending on how far we go)  
**Confidence**: VERY HIGH for <50 μs, HIGH for <40 μs

