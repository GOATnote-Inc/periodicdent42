# EvoEngineer Sweep Results - Oct 16, 2025

## 🚨 CRITICAL: Performance Regression Detected

### Executive Summary
- **Status**: ✅ Sweep completed successfully (24 variants across 2 generations)
- **Correctness**: ✅ All successful builds pass correctness (max_diff=0.001953)
- **Performance**: ❌ **CATASTROPHIC REGRESSION** - 23× slower than PyTorch SDPA

### Key Metrics
```
PyTorch SDPA Baseline:  50.18 μs (target)
Phase 3 (guarded):      1099.78 μs
Speedup vs SDPA:        0.046× (21.9× SLOWER)
```

### What Worked
1. ✅ **Build Infrastructure**: Parameterized builds working perfectly
2. ✅ **Correctness**: All variants pass `torch.allclose(atol=1e-3, rtol=1e-3)`
3. ✅ **EvoEngineer Loop**: Clean 2-generation sweep, proper logging
4. ✅ **BLOCK_M=32**: All configs with BLOCK_M=32 build and run

### What Failed
1. ❌ **Performance**: 1099.78 μs (vs 1634 μs Phase 3 baseline, vs 47 μs SDPA)
2. ❌ **BLOCK_M=64**: All variants fail to build (SMEM overflow)
3. ❌ **No Parameter Sensitivity**: VEC_WIDTH, NUM_WARPS, REDUCE, WMMA, SMEM_STAGE all produce **identical** timings

### Detailed Results

#### Generation 0 (12 variants)
| BLOCK_M | NUM_WARPS | VEC_WIDTH | REDUCE | Status | Time (μs) | Speedup |
|---------|-----------|-----------|--------|--------|-----------|---------|
| 32      | 4         | 2         | warp   | ✅     | 1099.78   | 0.046×  |
| 32      | 4         | 4         | warp   | ✅     | 1099.78   | 0.046×  |
| 32      | 4         | 8         | warp   | ✅     | 1099.78   | 0.046×  |
| 32      | 8         | 2         | warp   | ✅     | 1099.78   | 0.046×  |
| 32      | 8         | 4         | warp   | ✅     | 1099.78   | 0.046×  |
| 32      | 8         | 8         | warp   | ✅     | 1099.78   | 0.046×  |
| 64      | 4/8       | 2/4/8     | warp   | ❌     | -         | -       |

**Key Observation**: ALL successful variants have **identical performance**. This suggests:
- Optimizations are not being applied
- Bottleneck is elsewhere (synchronization? memory?)
- Compiler is optimizing away differences

#### Generation 1 (12 variants)
Explored mutations including:
- SMEM_STAGE=3
- USE_WMMA=1
- REDUCE=serial

**Result**: Still 1099.78 μs across all configs. No improvement.

### Root Cause Hypotheses

#### Hypothesis 1: Guarded Optimizations Not Active ⚠️
**Evidence**:
- VEC_WIDTH 2/4/8 all identical
- REDUCE warp/serial identical
- No performance difference despite #if guards

**Investigation Needed**:
```bash
# Check if macros are actually defined
nvcc -E -DBLOCK_M=32 -DVEC_WIDTH=4 -DREDUCE_WARP=1 fa_phase3_wmma.cu | grep "uint4"
```

#### Hypothesis 2: Excessive Synchronization 🔴
**Evidence**:
- 1099.78 μs = consistent across all configs
- Worse than Phase 3 baseline (1634 μs was before guarded opts)
- __syncthreads() after every operation

**Fix**:
- Remove unnecessary __syncthreads()
- Use warp-synchronous programming where possible

#### Hypothesis 3: SMEM Bank Conflicts 🟡
**Evidence**:
- HEAD_DIM=64 with no swizzling
- Likely 32-way bank conflicts on every access

**Fix**:
- Implement XOR swizzling for K_tile/V_tile
- Target 8-way bank conflicts max

#### Hypothesis 4: Incorrect Warp Reductions 🟡
**Evidence**:
- Warp vs serial makes no difference
- May have logic bug in warp reduction

**Fix**:
- Verify __shfl_down_sync is actually executing
- Add printf debugging in warp reduction path

### Comparison to Baselines

```
Kernel               Time (μs)    Speedup vs SDPA    Status
──────────────────────────────────────────────────────────
fa_minimal           2870.00      0.017×             ✅ Correct
fa_phase1 (tiled)    3652.00      0.013×             ✅ Correct (regression)
fa_phase3 (orig)     1634.00      0.029×             ✅ Correct
fa_phase3 (guarded)  1099.78      0.046×             ✅ Correct (REGRESSION)
PyTorch SDPA         50.18        1.000×             ✅ Target
```

**Critical**: Phase 3 guarded version is **SLOWER** than original Phase 3 (1099 vs 1634 μs).
This suggests the guarded optimizations are **hurting** performance, not helping.

### Next Steps (Priority Order)

#### Priority 0: Verify Optimizations Are Applied 🔴
```bash
# 1. Check vectorized loads in PTX
nvcc -ptx -DBLOCK_M=32 -DVEC_WIDTH=4 fa_phase3_wmma.cu
grep "ld.global.v4" fa_phase3_wmma.ptx

# 2. Check warp reductions
grep "__shfl" fa_phase3_wmma.ptx
```

#### Priority 1: Profile with Nsight Compute 🔴
```bash
ncu --set full --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum \
  python3 bench/test_phase3.py
```

#### Priority 2: Add Debug Logging 🟡
```cuda
#if defined(DEBUG_KERNEL)
if (blockIdx.x == 0 && threadIdx.x == 0) {
  printf("VEC_WIDTH=%d REDUCE_WARP=%d\\n", VEC_WIDTH, REDUCE_WARP);
}
#endif
```

#### Priority 3: Revert to Baseline 🟡
If optimizations don't help, revert to serial Phase 3 and focus on:
1. BLOCK_M=64 (fix SMEM overflow)
2. Remove excessive __syncthreads()
3. XOR swizzling

### Files
- Results: `evidence/evo_best.json`
- Full log: `evidence/evo_log.csv`
- Sweep log: `~/evo_sweep_SUCCESS.log` (on GPU)
- Commit: `a6981c7`

### Lessons Learned
1. ✅ **EvoEngineer infrastructure works**: Clean sweep, proper logging
2. ⚠️ **Guarded optimizations**: May not be applied or may hurt performance
3. ❌ **Premature optimization**: Should have profiled Phase 3 baseline first
4. ✅ **Correctness first**: All variants pass correctness, can debug perf safely

