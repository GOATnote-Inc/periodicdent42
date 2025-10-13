# Iteration 3: Performance Analysis from Benchmark Data

**Date**: October 13, 2025  
**Duration**: 15 minutes  
**Status**: Analysis complete, profiler access blocked

## Objective

Analyze benchmark results to identify optimization priorities without requiring Nsight Compute access.

## Data Analysis

### Performance Summary
```
Config          | Ours (ms) | PyTorch (ms) | Ratio | GFLOP/s Ours | GFLOP/s PyTorch
B=1,H=1,S=128   | 0.579     | 0.085        | 6.8×  | 7.2          | 49.4
B=4,H=8,S=128   | 0.752     | 0.085        | 8.9×  | 178.6        | 1588.2
B=8,H=8,S=128   | 1.281     | 0.086        | 14.9× | 209.6        | 3130.5
B=8,H=8,S=256   | 4.067     | 0.116        | 35.0× | 264.0        | 9242.6
B=16,H=8,S=512  | 23.900    | 0.326        | 73.2× | 359.4        | 26312.6
```

### Key Observations

1. **GFLOP/s scaling**: Our kernel scales poorly (7 → 359 GFLOP/s) vs PyTorch (49 → 26,313)
2. **Ratio degradation**: Performance gap widens from 6.8× to 73.2× as workload increases
3. **PyTorch efficiency**: Achieves 88% of L4 theoretical peak (30 TFLOP/s) at large batches
4. **Our efficiency**: Achieves 0.02-1.2% of theoretical peak across all configs

### Bottleneck Inference

**From Session 1 profiling data**:
- GPU utilization: 3.4% at B=1,H=1
- CTA count: 2 (on 58-SM GPU)
- Expected: ~350 CTAs needed for full utilization

**From GFLOP/s data**:
- Peak 359 GFLOP/s achieved at B=16,H=8,S=512
- L4 theoretical: 30,000 GFLOP/s
- Actual: 1.2% of peak

**Conclusion**: Not a parallelism-only problem. Even at maximum tested parallelism (512 CTAs), achieved only 1.2% of peak.

### Root Cause Hypotheses (Priority Order)

**1. Memory Bandwidth Bottleneck (Most Likely)**
- Evidence: Performance degrades with sequence length (S=128 → 512)
- Evidence: Low GFLOP/s suggests memory-bound, not compute-bound
- Mechanism: Non-coalesced memory access, strided loads/stores

**2. Algorithmic Inefficiency**
- Evidence: PyTorch constant ~0.08ms for S=128 regardless of B×H
- Evidence: Our kernel scales linearly with B×H (0.58 → 1.28ms)
- Mechanism: Missing optimizations (tiling, fusion, kernel launch overhead)

**3. Shared Memory Bank Conflicts**
- Evidence: 64×64 tile size on 32-bank shared memory
- Mechanism: 2-way bank conflicts expected
- Impact: ~2× slowdown on memory ops

**4. Register Pressure / Low Occupancy**
- Evidence: Low GFLOP/s despite sufficient CTAs at large batches
- Mechanism: Register spillage reducing warps per SM
- Impact: Limits parallelism even with many CTAs

## Estimated Impact of Fixes

| Optimization | Estimated Speedup | Implementation Time | Evidence Strength |
|--------------|-------------------|---------------------|-------------------|
| Vectorized memory access (float4) | 2-3× | 30-45 min | High (literature) |
| Shared memory padding | 1.3-1.5× | 15-20 min | Medium (32 banks) |
| Register optimization (__launch_bounds__) | 1.2-1.4× | 20-30 min | Medium (occupancy) |
| Persistent kernels | 2-4× @ small batch | 60-90 min | High (CTA count) |
| WMMA tensor cores | 2-3× | 2-3 hours | High (Ampere+) |

**Combined potential**: 2×1.3×1.2 = 3.1× for simple optimizations (90 min)

**Note**: This brings ratio from 14.9× slower to 4.8× slower at B=8,H=8,S=128.

## Iteration 4 Recommendation

**Target**: Vectorized memory access (float4 loads/stores)

**Rationale**:
- Highest impact (2-3×)
- Shortest implementation time (30-45 min)
- Well-documented in literature
- Directly addresses memory bandwidth bottleneck

**Implementation**:
1. Replace scalar loads with `reinterpret_cast<float4*>` for Q, K, V
2. Ensure coalesced access patterns (consecutive threads → consecutive addresses)
3. Test correctness and measure speedup

**Expected Result**:
- Before: 1.281ms @ B=8,H=8,S=128 (14.9× slower)
- After: 0.43-0.64ms (5-10× slower)
- GFLOP/s: 209 → 420-630 GFLOP/s

## Alternative: Revert to Working Baseline

**Option**: Revert Iteration 1 changes and start from known-good kernel

**Current state**:
- Modified files have correctness bugs (KV-split)
- Working kernel (without changes) exists at earlier commit
- Benchmark report validated correctness of current kernel

**Recommendation**: Keep current kernel for Iteration 4, as it's numerically correct.

## Next Steps

**Immediate (Iteration 4 - 30-45 min)**:
- Implement vectorized memory access (float4)
- Target: 2-3× speedup
- Measure: GFLOP/s improvement, memory bandwidth utilization

**Subsequent (Iteration 5 - 15-20 min)**:
- Add shared memory padding to avoid bank conflicts
- Target: 1.3-1.5× additional speedup

**Final (Iteration 6 - 20-30 min)**:
- Register optimization with `__launch_bounds__`
- Target: 1.2-1.4× additional speedup

**Combined**: 2× × 1.3× × 1.2× = **3.1× total speedup** in ~90 minutes

This would improve B=8,H=8,S=128 from 14.9× slower to 4.8× slower.

## Agentic Loop Status

**Completed**:
- ✓ Session 1: Profiling (identified 3.4% GPU utilization)
- ✓ Iteration 1: KV-split (incomplete, correctness bugs)
- ✓ Iteration 2: Batch size analysis (identified invalid comparison)
- ✓ Iteration 3: Benchmark validation (quantified 6.8-73× slower)

**Current**: Iteration 4 ready to begin (vectorized memory access)

**Estimated iterations to competitive performance**:
- Iterations 4-6 (simple opts): 90 min → 4.8× slower
- Iterations 7-8 (WMMA, async): 3-4 hours → 1.5-2× slower
- Iteration 9+ (advanced): 4-6 hours → 0.8-1.2× (competitive)

**Total estimated time to SOTA**: 8-11 hours of focused optimization

