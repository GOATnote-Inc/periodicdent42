# Complete Fix Analysis: Dynamic Grid + cp.async

**Date**: 2025-10-15  
**Fixes Applied**: Dynamic occupancy-based grid sizing + 2-stage cp.async pipeline  
**Status**: ✅ APPLIED, ❌ NO PERFORMANCE IMPROVEMENT

---

## Executive Summary

**Applied Two Major Fixes**:
1. **Dynamic Grid Sizing**: Replaced hard cap with occupancy-based calculation
2. **cp.async 2-Stage Pipeline**: Proper pipeline priming with `wait_group<1>`

**Result**: Grid sizing now occupancy-aware (348 blocks for all shapes), but **performance unchanged** (14.62× scaling, same as before).

**Root Cause Identified**: **Severe occupancy limitation** (only 3 active blocks/SM on L4's 58 SMs)

---

## Fixes Applied

### Fix 1: Dynamic Occupancy-Based Grid Sizing

**Before**:
```cpp
const int num_blocks = min(total_work, 256);  // Hard cap
```

**After**:
```cpp
int device = 0, sm_count = 0, active_per_sm = 0;
cudaGetDevice(&device);
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_per_sm,
    flash_attention_s512_v3_kernel<Traits>,
    Traits::NUM_THREADS,
    0 /* dynamic smem */
);

const int oversub = 2; // mild oversubscription
const int target = (sm_count > 0 ? sm_count : 64) * (active_per_sm > 0 ? active_per_sm : 1) * oversub;
num_blocks = (total_work < target) ? total_work : target;
if (num_blocks <= 0) num_blocks = (total_work > 0 ? min(total_work, 1024) : 0);
```

**Verification**: Grid now launches **348 blocks** for all shapes (vs 256 before, or 128-2048 in naive fix)

### Fix 2: cp.async 2-Stage Pipeline

**Before** (FIX A - wait_group<0>):
```cpp
// Prefetch first stage
load_K_async(..., 0, ...);
load_V_async(..., 0, ...);
cp_async_commit_group();
cp_async_wait_group<0>();  // Wait for ALL groups (no pipelining)

for (int n_block = 0; n_block < num_blocks_n; n_block++) {
    if (n_block + 1 < num_blocks_n) {
        load_K_async(..., stage_load, ..., n_block + 1, ...);
        load_V_async(..., stage_load, ..., n_block + 1, ...);
        cp_async_commit_group();
    }
    cp_async_wait_group<0>();  // Wait for ALL groups
    compute_block(...);
}
```

**After** (FIX B - 2-stage with wait_group<1>):
```cpp
// Prefetch stage 0
load_K_async(..., 0, ...);
load_V_async(..., 0, ...);
cp_async_commit_group();

// Prefetch stage 1 (enables wait_group<1> for proper pipelining)
if (num_blocks_n > 1) {
    load_K_async(..., 1, ..., 1, ...);
    load_V_async(..., 1, ..., 1, ...);
    cp_async_commit_group();
}

for (int n_block = 0; n_block < num_blocks_n; n_block++) {
    // With 2 groups in flight, wait for the older one
    cp_async_wait_group<1>();
    __syncthreads();
    
    // Keep 2 stages in flight
    if (n_block + 1 < num_blocks_n) {
        load_K_async(..., stage_load, ..., n_block + 1, ...);
        load_V_async(..., stage_load, ..., n_block + 1, ...);
        cp_async_commit_group();
    }
    
    compute_block(...);
}
```

**Benefits**:
- Proper memory/compute overlap
- Avoids "read-before-ready" hazards
- Keeps 2 async loads in flight

---

## Performance Results

### After Complete Fix

| Shape | V3 Time | SDPA Time | Slowdown | Scaling vs B=1 | Grid Blocks |
|-------|--------:|----------:|---------:|---------------:|------------:|
| B=1,H=8  | 7.782ms | 0.026ms | 294× | 1.00× | 348 |
| B=4,H=16 | 56.9ms | 0.095ms | 599× | 7.31× | 348 |
| B=8,H=16 | 113.7ms | 0.159ms | 713× | 14.62× | 348 |

### Comparison with Previous Attempts

| Fix | B=1→B=8 Scaling | Grid for B=8,H=16 | Status |
|-----|----------------:|------------------:|:------:|
| **Original (256 cap)** | 17.2× | 256 (capped) | ❌ |
| **Naive (total_work)** | 14.7× | 2048 (uncapped) | 🟡 |
| **Dynamic + cp.async** | **14.62×** | **348 (occupancy)** | 🟡 |

**Key Finding**: Performance essentially unchanged (14.62× vs 14.70×) despite correct grid sizing.

---

## Root Cause: Severe Occupancy Limitation

### Occupancy Calculation

From debug output: `Grid=348` for all shapes

**Reverse calculation**:
```
Grid = sm_count × active_per_sm × oversub
348 = 58 SMs × active_per_sm × 2 oversub
active_per_sm = 348 / (58 × 2) = 3
```

**Result**: Only **3 active blocks per SM** (should be 8-16 for good occupancy)

### Why Low Occupancy?

L4 (Ada, sm_89) specifications:
- 58 SMs
- 128 registers per thread max
- 64 KB shared memory per SM
- 2048 threads per SM max

**Likely Causes** (in order of probability):

#### 1. **Excessive Register Usage** (Most Likely)
- **Kernel uses 128 threads/block**
- If each thread uses **>42 registers**:
  - 128 threads × 42 regs/thread = 5,376 regs/block
  - SM has 65,536 regs total
  - Max blocks/SM = 65,536 / 5,376 = **12 blocks/SM**
- If each thread uses **>64 registers**:
  - 128 threads × 64 regs/thread = 8,192 regs/block
  - Max blocks/SM = 65,536 / 8,192 = **8 blocks/SM**
- If each thread uses **>85 registers**:
  - 128 threads × 85 regs/thread = 10,880 regs/block
  - Max blocks/SM = 65,536 / 10,880 = **6 blocks/SM**
- **If >100 registers/thread**:
  - Max blocks/SM = **≤4**, matching our observed **3**

**Culprits in V3 Kernel**:
```cpp
// Per-warp register arrays (likely spillover)
half Q_reg[Traits::BLOCK_M / Traits::NUM_WARPS][Traits::HEAD_DIM];
float O_acc[Traits::BLOCK_M / Traits::NUM_WARPS][Traits::HEAD_DIM];
float m_i[Traits::BLOCK_M / Traits::NUM_WARPS];
float l_i[Traits::BLOCK_M / Traits::NUM_WARPS];

// For BLOCK_M=32, NUM_WARPS=4, HEAD_DIM=64:
// Q_reg: 8 × 64 = 512 halfs = 512 bytes = 128 regs (!)
// O_acc: 8 × 64 = 512 floats = 2048 bytes = 512 regs (!!)
// Total per thread: ~640 regs (WAY over limit)
```

This is almost certainly spilling to local memory, **destroying performance**.

#### 2. **Excessive Shared Memory** (Possible)
- V3 uses ~32-40 KB SMEM per block
- L4 has 64 KB SMEM per SM
- This limits to **1-2 blocks/SM** from SMEM alone
- Combined with register pressure: **occupancy = min(3 from regs, 1-2 from SMEM) = 1-2**
- But CUDA reports **3**, so registers are the primary limit

#### 3. **cp.async Barrier Overhead** (Minor)
- 2-stage pipeline requires proper sequencing
- If not overlapping well, performance stays flat
- But unlikely to cause 300-700× slowdown

---

## Why Previous Fixes Didn't Help

### 1. Grid Sizing Fix
- **Enabled**: Proper parallelism across batches/heads
- **Benefit**: Small improvement (17.2× → 14.7×)
- **Limitation**: Occupancy still bottlenecked

### 2. Dynamic Grid + cp.async
- **Grid now occupancy-aware**: 348 blocks (58 SMs × 3 active/SM × 2 oversub)
- **Pipeline now correct**: 2-stage with proper wait_group
- **No improvement**: Occupancy is the bottleneck, not grid size or pipeline

**The Vicious Cycle**:
1. Excessive register usage → low occupancy (3 blocks/SM)
2. Low occupancy → poor latency hiding
3. Poor latency hiding → memory-bound performance
4. More blocks won't help if they can't fit!

---

## Next Steps (Priority Order)

### 1. **Confirm Register Usage** (15 minutes)
```bash
# Check register count per thread
nvcc -arch=sm_89 -O3 --ptxas-options=-v fa_s512_v3.cu 2>&1 | grep "registers"

# Expected output: "... uses X registers" where X > 100 confirms our hypothesis
```

**If X > 85**: Register spilling is the culprit

### 2. **Reduce Register Pressure** (2-3 hours)
**Option A: Move Large Arrays to Shared Memory**
```cpp
// Instead of per-thread register arrays:
// half Q_reg[8][64];  // 128 regs/thread ❌

// Use per-warp SMEM:
__shared__ half Q_smem[NUM_WARPS][8][64];  // 32 KB
half* Q_reg = &Q_smem[warpIdx][laneIdx];  // Pointer instead
```

**Option B: Reduce Tile Sizes**
```cpp
// Current: BLOCK_M=32, BLOCK_N=64, NUM_WARPS=4
// Smaller: BLOCK_M=16, BLOCK_N=32, NUM_WARPS=2
// Cuts register usage by 4×
```

**Option C: Use Half Precision Accumulation**
```cpp
// Current: float O_acc[...];  // 512 floats = 512 regs
// Change to: half O_acc[...];  // 512 halfs = 128 regs
// Then upcast to float only for softmax
```

### 3. **Occupancy Tuning** (1 hour)
After reducing registers, retune block size:
```cpp
// Try different THREADS_PER_BLOCK values
constexpr int THREADS_PER_BLOCK = 256;  // vs current 128
// This may allow better occupancy if reg/thread is low enough
```

### 4. **Validate Improvement** (30 minutes)
Re-run benchmarks expecting:
- Occupancy: 3 → 8-12 blocks/SM
- Scaling: 14.6× → ≤3×
- Absolute: 7.78ms → <2ms

---

## Alternative Approach: Use Tensor Cores

**Observation**: SDPA is 300-700× faster partly because it uses **Tensor Cores** (HMMA instructions)

**Tensor Core Benefits on L4**:
- 128 TFLOP/s FP16 (vs 30 TFLOP/s FP16 CUDA cores)
- Built-in for matmul (QK^T, PV)
- Better occupancy (less register pressure)

**Implementation**:
- Use CUTLASS 3.x WGMMA API for Ada (sm_89)
- Or use PTX `mma.sync` instructions directly
- FlashAttention-3 style (split-K, warp specialization)

**Effort**: 2-3 days, but likely to achieve competitive performance

---

## Recommendations

### **For This Session** (Remaining ~4 hours)
1. ✅ **Document findings** (this report)
2. 🔄 **Check register usage** (15 min) - confirm hypothesis
3. ⏳ **Apply Option A** (move Q_reg, O_acc to SMEM) if time permits
4. ⏳ **Quick validation** (30 min) if fix applied

### **For Next Session**
1. Full register pressure reduction (Option A + B)
2. Occupancy tuning and validation
3. Consider tensor core path if still far from target

### **Long Term**
- Implement tensor core path (CUTLASS WGMMA)
- Full FlashAttention-3 style optimizations
- Cross-bench validation with CUTLASS profiler

---

## Conclusion

✅ **Both fixes successfully applied**:
- Dynamic occupancy-based grid sizing
- 2-stage cp.async pipeline

❌ **No performance improvement** because:
- **Root cause is occupancy** (3 blocks/SM vs target 8-16)
- **Likely cause is excessive register usage** (>100 regs/thread)
- Grid sizing and pipeline are no longer the bottleneck

📊 **Status**: Infrastructure complete, root cause identified, clear path forward.

🎯 **Next critical step**: Reduce register pressure to improve occupancy from 3 → 8-12 blocks/SM.

---

*Report generated: 2025-10-15 03:15 UTC*  
*GPU Time Remaining: ~4 hours*  
*Recommendation: Validate register usage hypothesis (15 min), then decide on register reduction approach*

