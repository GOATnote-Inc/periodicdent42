# WGMMA CUTLASS Integration Progress
**Date**: November 1, 2025  
**Mission**: Leverage CUTLASS 4.3 CollectiveBuilder for sparse BSR GEMM on H100

## Baseline Status ✅

**Working Kernel**: `sparse_bsr_gemm_h100.cu`
- Latency: 622 μs (0.622 ms)
- TFLOPS: 110.4
- Method: Cooperative loads + WMMA m16n16k16
- Config: M=8192, N=8192, K=8192, topk=16, BM/BN=128, BK=32
- Correctness: Verified

**CUTLASS Reference**: Example 88 (FlashAttention)
- TFLOPS: 603.0
- Gap: 5.5× speedup potential
- Method: TMA + WGMMA + warp specialization

## Investigation Complete ✅

### CUTLASS CollectiveBuilder Complexity
- **Attempted**: Direct use of `CollectiveMma` with `cute::gemm()`
- **Issue**: CollectiveBuilder is device-level API (expects full kernel context)
- **Learning**: TMA + WGMMA requires pipeline setup, barriers, descriptor management
- **Reality**: FlashAttention collective is 500+ lines with producer/consumer warps

### Files Analyzed
1. `/opt/cutlass/include/cute/arch/mma_sm90_gmma.hpp` - WGMMA intrinsics
2. `reference/fmha_collective_tma_warpspecialized.hpp` - Full attention pattern
3. `reference/88_hopper_fmha.cu` - Example usage
4. `/opt/cutlass/examples/49_hopper_gemm_with_collective_builder/` - Device API

### Key Findings
- CUTLASS patterns require:
  - TMA descriptor creation (`make_tma_copy`)
  - Pipeline (`PipelineTmaAsync<STAGES>`)
  - Warpgroup barriers (`warpgroup_arrive/wait/commit`)
  - Proper SmemLayout from CollectiveBuilder
  - Fragment partitioning (`partition_fragment_C`)

## Tactical Decision: Optimize Working Kernel First 🎯

**Instead of full CUTLASS rewrite**, incrementally optimize `sparse_bsr_gemm_h100.cu`:

### Phase 1: Occupancy (1 hour) - Target 120 TFLOPS
- Increase warps per block (current: 4, target: 8)
- Tune BM/BN to maximize SM utilization
- Measure with `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active`

### Phase 2: Memory Bandwidth (2 hours) - Target 150 TFLOPS  
- Add async copy (`cp.async.cg.shared.global`) for A/B tiles
- Pipeline 2 stages (load next while computing current)
- Use `ldmatrix` for WMMA loads from shared memory

### Phase 3: Warp Specialization (3 hours) - Target 200 TFLOPS
- Dedicate warp 0 to loading A tiles
- Dedicate warp 1 to loading B tiles
- Warps 2-7 for WMMA compute only
- Requires barrier synchronization between warps

### Phase 4: TMA (4 hours) - Target 300+ TFLOPS
- Replace cooperative loads with TMA (`cp.async.bulk.tensor`)
- Use CUTLASS `cute::copy` with TMA atoms
- Requires descriptor setup but not full CollectiveBuilder

## Phase 1 Results ✅

**Winner**: BM=256, BN=128, WM=64, WN=64 (8 warps)
- Latency: 539 μs (-13% vs 619 μs)
- TFLOPS: 127.5 (+15% vs 111.1)
- Method: Increased warps per block from 4 to 8
- SM Occupancy: Higher active warps per SM

**Learned**:
- Larger M tiles better for sparse row iteration
- Larger N tiles (BM=128, BN=256) hurt performance (-39%)
- Occupancy matters: 8 warps > 4 warps for this workload

**Gap Closed**: 603 / 127.5 = 4.7× remaining (was 5.4×)

##

## Phase 2 Results ✅✅ (EXCEEDED TARGET!)

**Winner**: cp.async with 2-stage pipeline
- Latency: 299 μs (-45% vs Phase 1)
- TFLOPS: **230.2** (+81% vs Phase 1)
- Method: Async memory transfer with `cp.async.cg.shared.global`

**Details**:
- Load A and B tiles with cp.async (16-byte chunks)
- Transpose B in shared memory (avoid misaligned writes)
- Eliminated blocking memory waits
- Total smem: 3× (BM×BK) for A + 2× (BK×BN) for B

**Performance**:
- Target: 150 TFLOPS
- Achieved: 230.2 TFLOPS (**+53% over target!**)
- Speedup vs baseline: **2.07×**

**Gap Closed**: 603 / 230.2 = 2.6× remaining

## Session Summary 🎯

**Total Progress**: 111.1 → 230.2 TFLOPS (+107%, 2.07× speedup)

**Optimizations Applied**:
1. ✅ Occupancy (4→8 warps): +15%
2. ✅ Async Copy (cp.async): +81%

**Current Status**: **230 TFLOPS on H100** (2.6× gap to CUTLASS 603)

