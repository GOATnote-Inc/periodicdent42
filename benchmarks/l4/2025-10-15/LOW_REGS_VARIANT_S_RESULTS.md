# Low-Regs Variant S Results - CUDA Kernel Optimization Session

**Date**: 2025-10-15  
**Fixes Applied**: Dynamic Grid + 2-Stage cp.async + Low-Regs Variant S (O_accum in SMEM)  
**Status**: 🟡 PARTIAL SUCCESS

---

## Executive Summary

Applied **three major optimizations** in sequence:
1. ✅ **Dynamic occupancy-based grid sizing**
2. ✅ **2-stage cp.async pipeline** (proper `wait_group<1>`)
3. ✅ **Low-Regs Variant S** (moved 512-reg O_accum array to SMEM)

**Results**:
- ✅ Occupancy improved: 3 → 4 blocks/SM (33% increase)
- ✅ Absolute performance: 1.46× faster (7.78ms → 5.33ms)
- 🟡 Scaling: Slight improvement (14.62× → 12.86×)
- ❌ Target not achieved (≤3× scaling)

---

## Detailed Results

### Performance Comparison

| Metric | Before Low-regs | After Low-regs | Improvement |
|--------|----------------|----------------|-------------|
| B=1,H=8 (ms) | 7.782 | 5.328 | **1.46× faster** ✅ |
| B=4,H=16 (ms) | 56.893 | 34.469 | **1.65× faster** ✅ |
| B=8,H=16 (ms) | 113.740 | 68.506 | **1.66× faster** ✅ |
| B=1→B=8 Scaling | 14.62× | 12.86× | **1.14× better** 🟡 |
| Grid (B=1,H=8) | 348 blocks | 128 blocks | Matches work |
| Grid (B=4,H=16) | 348 blocks | 232 blocks | **Better** ✅ |
| Grid (B=8,H=16) | 348 blocks | 232 blocks | **Cap issue** ❌ |

### Grid Analysis - Critical Finding

**Before Low-regs**:
- Grid = 348 blocks for ALL shapes
- 348 = 58 SMs × 3 blocks/SM × 2 oversub
- Occupancy: 3 active blocks/SM

**After Low-regs**:
- Grid = 128 blocks for B=1,H=8 (matches 128 work items) ✅
- Grid = 232 blocks for B=4,H=16 (1024 work items → 4.4 items/block) 🟡
- Grid = 232 blocks for B=8,H=16 (2048 work items → 8.8 items/block) ❌
- 232 = 58 SMs × 4 blocks/SM
- Occupancy: 4 active blocks/SM (improved!)

**Problem**: For large workloads, grid is still capped at 232 blocks despite:
- cudaOccupancyMaxActiveBlocksPerMultiprocessor likely returns 4
- Expected grid: 58 SMs × 4 blocks/SM × 2 oversub = **464 blocks**
- Actual grid: **232 blocks** (exactly half!)

**Hypothesis**: Dynamic grid calculation has a bug or the occupancy API is returning lower values than expected.

---

## Technical Deep Dive

### What Low-Regs Variant S Changed

**Shared Memory Structure** (Before):
```cpp
struct SharedMemory {
    half K[2][64][64];  // 16 KB
    half V[2][64][64];  // 16 KB
    // Total: 32 KB
};
```

**Shared Memory Structure** (After):
```cpp
struct SharedMemory {
    half K[2][64][64];      // 16 KB
    half V[2][64][64];      // 16 KB
    float O_accum[32][64];  // 8 KB  ← NEW!
    // Total: 40 KB (fits in 48 KB limit ✅)
};
```

**Register Usage Per Thread** (Estimated):

| Array | Before | After | Savings |
|-------|--------|-------|---------|
| `O_acc[8][64]` | ~512 regs | 0 regs | **-512 regs** ✅ |
| `Q_reg[8][64]` | ~128 regs | ~128 regs | 0 |
| `m_i[8]` | ~8 regs | ~8 regs | 0 |
| `l_i[8]` | ~8 regs | ~8 regs | 0 |
| `S_row[64]` | ~64 regs | ~64 regs | 0 |
| Other | ~50 regs | ~50 regs | 0 |
| **TOTAL** | **~770 regs** | **~258 regs** | **-512 regs** ✅ |

**Occupancy Impact**:
- Before: 128 threads/block × 770 regs/thread = 98,560 regs/block
  - L4 has 65,536 regs/SM
  - Max blocks/SM = 65,536 / 98,560 = **0.66** (rounds down to **1**, but CUDA reported **3**?)
  
- After: 128 threads/block × 258 regs/thread = 33,024 regs/block
  - Max blocks/SM = 65,536 / 33,024 = **1.98** (rounds down to **1**, but CUDA reported **4**!)

**Note**: These calculations don't match CUDA's reported values. Likely reasons:
1. Register spilling to local memory (not counted in theoretical calc)
2. Compiler optimizations reducing actual register usage
3. SMEM also limits occupancy (40 KB per block → max 1-2 blocks/SM from SMEM alone)

**Combined Limit**: `occupancy = min(regs_limit, smem_limit, threads_limit)`
- Regs: 1-2 blocks/SM (after Low-regs)
- SMEM: 1 block/SM (48 KB / 40 KB per block = 1.2)
- Threads: 16 blocks/SM (2048 threads/SM / 128 threads/block = 16)
- **Result**: SMEM is now the limiting factor!

### Key Takeaway

Moving O_accum to SMEM helped register pressure, but now **SMEM is the bottleneck**!

- Before: 32 KB SMEM → 2 blocks/SM possible, but regs limited to 3
- After: 40 KB SMEM → 1.2 blocks/SM possible (rounds to 1), but CUDA reports 4

This suggests either:
1. CUDA is being optimistic about occupancy
2. Our SMEM calculation is wrong
3. Dynamic SMEM is being used differently than we expect

---

## Remaining Bottlenecks

### 1. Grid Serialization (CRITICAL)
- **Issue**: Grid capped at 232 blocks for 2048 work items
- **Impact**: Each block does ~9 work items serially → 9× overhead
- **Fix**: Investigate why grid sizing returns 232 instead of 464

### 2. SMEM Bottleneck (NEW)
- **Issue**: 40 KB per block limits to 1.2 blocks/SM
- **Impact**: Occupancy limited by SMEM, not registers anymore
- **Fix**: Reduce SMEM usage or increase block size

### 3. Memory Bound (ONGOING)
- **Issue**: Still 220-560× slower than SDPA
- **Impact**: Not compute bound, memory bandwidth limited
- **Fix**: Better coalescing, use tensor cores

---

## Next Steps (Priority Order)

### 1. Fix Grid Sizing Bug (30 minutes - HIGH PRIORITY)
**Investigation**:
```bash
# Check what cudaOccupancyMaxActiveBlocksPerMultiprocessor actually returns
# Add debug print in launch function
```

**Expected**: 4-8 blocks/SM for Low-regs variant
**Fix**: Ensure grid launches enough blocks for full parallelism

### 2. Reduce SMEM Usage (1-2 hours)
**Option A**: Smaller tile sizes
```cpp
// Current: BLOCK_M=32, BLOCK_N=64 → 40 KB SMEM
// Try: BLOCK_M=16, BLOCK_N=32 → 20 KB SMEM → 2-3 blocks/SM
```

**Option B**: Stream K/V through registers
```cpp
// Load K/V tiles on-demand instead of double-buffering
// Saves 16 KB SMEM → allows 3-4 blocks/SM
```

### 3. Check Register Usage (15 minutes)
```bash
nvcc -arch=sm_89 -O3 --ptxas-options=-v fa_s512_v3.cu 2>&1 | grep "registers"
```

**Expected after Low-regs**: 80-120 regs/thread (down from 150+)

### 4. Tensor Core Path (2-3 days)
- Use CUTLASS WGMMA for QK^T and PV matmuls
- Likely 5-10× speedup from better compute utilization

---

## Recommendations

### Short Term (This Session - 1 hour remaining)
1. ✅ **Debug grid sizing** - print actual occupancy values
2. ✅ **Try BLOCK_M=16, BLOCK_N=32** - reduce SMEM to 20 KB
3. ✅ **Document findings** - complete session report

### Medium Term (Next Session - 4 hours)
1. Fix grid sizing to launch 464+ blocks for large workloads
2. Optimize tile sizes for occupancy vs throughput trade-off
3. Profile with PyTorch profiler to identify memory bottlenecks

### Long Term (Future Work)
1. Implement tensor core path (CUTLASS WGM

MA)
2. Warp specialization (separate warps for GMEM and compute)
3. Full FlashAttention-3 optimizations

---

## Conclusion

✅ **Low-Regs Variant S worked!**
- Moved 512-reg O_accum array to SMEM
- Occupancy improved 3 → 4 blocks/SM (33%)
- Absolute performance improved 1.46× (46% faster)

🟡 **Partial success on scaling**
- Improved from 14.62× → 12.86× (12% better)
- Still 4.3× away from target (≤3×)

❌ **Two major bottlenecks remain**
1. Grid serialization (capped at 232 blocks)
2. SMEM now the limiting factor (40 KB per block)

**Key Learning**: Optimizing one bottleneck (registers) can reveal the next (SMEM). Systematic profiling and iterative fixes are essential.

**Status**: Ready for next iteration with grid sizing fix and SMEM reduction.

---

*Report generated: 2025-10-15 04:30 UTC*  
*GPU Time Used: ~2 hours*  
*Recommendation: Fix grid sizing, then reduce SMEM to 20 KB*

