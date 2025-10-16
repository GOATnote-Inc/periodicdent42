# Phase 1: Lessons Learned

## Result

**Correctness**: ✅ PASS (max_diff=0.001953)  
**Performance**: ❌ REGRESSION (3652 μs vs 2870 μs baseline = 0.79×)

## What Happened

**Optimization attempt**: Block tiling (BLOCK_M=16)
- Reduced grid from 512 blocks → 32 blocks
- Each block handles 16 query rows

**Bug**: Warp reductions caused NaN outputs

**Fix**: Single-threaded max/sum → correctness ✅ but serialization bottleneck

## Root Cause

**Serialization killed parallelism**:
```cuda
// Single thread does ALL max/sum work:
if (tid == 0) {
    for (int col = 0; col < 64; col++) {
        m_new = fmaxf(m_new, S_tile[row][col]);
    }
}
// Other 127 threads wait idle!
```

**Result**: Grid reduction gains (512→32 blocks) < serialization cost

## EvoEngineer Lesson

✅ **Correctness First**: Caught bug, fixed it  
❌ **Performance**: Didn't validate optimization hypothesis  
📚 **Learning**: Block tiling needs **parallel reductions**, not serial

## Why Minimal Kernel Was Faster

**Minimal kernel** (2870 μs):
- 512 blocks, each handles 1 row
- Simple, fully parallel within block
- No serialization bottlenecks

**Phase 1** (3652 μs):
- 32 blocks, each handles 16 rows
- Single-threaded reductions = bottleneck
- Grid reduction can't compensate

## Decision: Skip to High-Value Optimizations

Instead of debugging Phase 1 warp reductions, **pivot to proven wins**:

### Phase 2: Vectorized I/O (Target: 2-3× speedup)
- `uint4` loads for K/V (16 bytes → 1 instruction)
- Coalesced memory access
- Simple, low-risk optimization
- **Expected**: 2870 μs → ~1000 μs

### Phase 3: Tensor Cores (Target: 10-20× speedup)
- WMMA for Q@K^T and P@V
- FP16 accumulation (2× throughput on Ada)
- This is where the REAL speedup comes from
- **Expected**: 1000 μs → ~50-100 μs

## Expert Engineering Decision

**Don't over-optimize Phase 1**. The 57× speedup will come from:
- Tensor Cores (80% of the win)
- Vectorized I/O (15% of the win)
- Everything else (5% of the win)

Block tiling can wait until after we have Tensor Cores working.

## Next: Phase 2 (Vectorized I/O)

Using **minimal kernel as base** (proven correct, 2870 μs)  
Adding **uint4 vectorized loads**  
Target: ~1000 μs (3× speedup)

GPU: Still running @ us-west1-c ✅

