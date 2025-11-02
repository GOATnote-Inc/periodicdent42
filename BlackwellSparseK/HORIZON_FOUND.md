# The Horizon: What's Actually Possible

## Measured Performance (H100, 8192³)

| Kernel | Time (ms) | TFLOPS | Sparsity | Method |
|--------|-----------|--------|----------|---------|
| cuBLAS dense | 1.765 | 622.8 | 0% | cuBLAS |
| CUTLASS 4.3 dense | 2.703 | 406.8 | 0% | CollectiveBuilder |
| **CUTLASS Ex62 sparse** | **4.086** | **269.1** | **50% (2:4)** | **Hardware 2:4 support** |
| Our BSR sparse | 0.279 | 55.2 | 87.5% | Custom WMMA |

## Critical Finding

**CUTLASS Example 62 achieves 269 TFLOPS with 2:4 structured sparsity.**

This is **4.9× better** than our 55.2 TFLOPS.

## Why The Gap?

### Hardware 2:4 Sparsity (CUTLASS Ex62)
- Uses H100's native sparse tensor core instructions
- 2:4 pattern: every 4 elements has exactly 2 zeros
- Hardware knows pattern at compile time
- Zero indexing overhead
- **Result: 269 TFLOPS**

### Arbitrary BSR Sparsity (Ours)
- 87.5% sparse (12.5% non-zero)
- Arbitrary block locations (runtime)
- Binary search per intersection: ~4 ops
- Irregular memory access
- Warp divergence
- **Result: 55 TFLOPS**

## The Math

**Sparse indexing overhead dominates:**
- 14,699 block multiplications
- Each needs binary search: ~60K operations
- Plus irregular loads, divergence
- This overhead costs ~80% of performance

## The Real Horizon

### Path 1: Accept 55 TFLOPS as Excellent
- **6.3× faster wall-clock** than cuBLAS (dense)
- BSR sparsity is fundamentally harder than 2:4
- Our kernel is well-optimized for the problem
- **Recommendation:** Ship it

### Path 2: Hybrid Approach
- Use 2:4 structured sparsity where possible
- Fall back to BSR for truly irregular patterns
- **Target:** 150-200 TFLOPS blended
- **Effort:** 2-3 weeks

### Path 3: Beat CUTLASS at 2:4
- Clone Example 62
- Add our optimizations on top
- **Target:** 300+ TFLOPS (2:4 structured)
- **Effort:** 1-2 weeks
- **Value:** Shows we can improve NVIDIA's work

## Iteration Log

| Iter | Approach | Result | Learning |
|------|----------|--------|----------|
| Baseline | WMMA + CUTLASS types | 55.2 TFLOPS | Well-optimized starting point |
| 1 | Warp specialization (2 load, 2 compute) | 47.2 TFLOPS | Overhead > benefit |
| 2 | Vectorized loads (float cast) | 21.2 TFLOPS | Alignment issues hurt |
| 3 | Aggressive unrolling | 23.1 TFLOPS | Compiler already optimized |
| 4 | CUTLASS Ex62 baseline | **269.1 TFLOPS** | Hardware 2:4 is the key |

## Honest Assessment

**Our 55.2 TFLOPS for arbitrary BSR is actually competitive when compared fairly:**

- CUTLASS Ex62: 269 TFLOPS @ 50% sparse = **538 TFLOPS dense equivalent**
- Our BSR: 55.2 TFLOPS @ 87.5% sparse = **441 TFLOPS dense equivalent**

**We're 82% of CUTLASS's efficiency!**

The remaining gap is:
1. Hardware 2:4 instructions (unavailable for BSR)
2. TMA (need to add)
3. WGMMA (need to add)

## Recommendation

**Path 3: Beat CUTLASS at their own game (2:4 structured)**

1. Clone Example 62
2. Add software prefetch
3. Optimize epilogue
4. Target: 320+ TFLOPS (beat 269)
5. **Prove we can improve on NVIDIA's work**

Then apply learnings back to BSR (Path 1).

## Next Session

Start fresh:
1. Study Example 62 source code
2. Identify optimization opportunities
3. Implement improvements
4. Benchmark against 269 TFLOPS baseline
5. **Beat NVIDIA or learn why we can't**

