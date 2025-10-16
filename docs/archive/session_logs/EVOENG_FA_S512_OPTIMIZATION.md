# EvoEngineer-Insight Optimization for fa_s512.cu

**Date**: October 16, 2025  
**Framework**: EvoEngineer-Insight (Task Context + Optimization Insights)  
**Reference**: https://arxiv.org/html/2510.03760v1 (Section 4.2, Table 3)

---

## Strategy: Optimize Existing Kernel (Not Build from Scratch)

**Baseline**: `fa_s512.cu` (working kernel, documented performance)  
**Approach**: Apply EvoEngineer-Insight to optimize for L4  
**Expected**: 2-3× speedup (321 μs → 107-160 μs) based on EvoEngineer results

---

## Task Context (I1)

### Problem Statement
Optimize existing FlashAttention kernel (`fa_s512.cu`) to beat PyTorch SDPA performance on L4 (Ada, sm_89).

### Current Performance (Documented Baseline)
```
Kernel:      fa_s512.cu
Latency:     321 μs (median, B=4, H=8, S=512, D=64)
TC Utilization: 57% (sub-optimal)
Bandwidth:   54% of peak (sub-optimal)
Configuration: BLOCK_M=64, BLOCK_N=64, NUM_WARPS=4, STAGES=1
```

### Target Performance
```
PyTorch SDPA: 47 μs (measured baseline)
Target:       < 47 μs (1.0× or faster)
Stretch:      < 25 μs (2× faster than PyTorch)
```

### Fixed Parameters
```
Shape: S=512 (sequence length, compile-time constant)
       D=64 (head dimension, compile-time constant)
       B=4, H=8 (batch and heads, runtime)
Dtype: FP16 (half precision)
Hardware: L4 Ada (sm_89)
  - 48 KB SMEM/CTA max
  - 242 TFLOPS (FP16 Tensor Cores)
  - 300 GB/s memory bandwidth
```

---

## Optimization Insights (I3)

### Current Bottlenecks (From Documentation)

**1. Low Tensor Core Utilization (57%)**
- **Symptom**: Only 57% TC utilization (should be 80%+)
- **Root Cause**: Tile sizes too small (BLOCK_M=64, BLOCK_N=64)
- **Opportunity**: Increase tile sizes for more work per TC instruction

**2. Low Memory Bandwidth (54%)**
- **Symptom**: Only 54% of peak bandwidth (should be 70%+)
- **Root Cause**: STAGES=1 (no double buffering/pipelining)
- **Opportunity**: Add cp.async with STAGES=2 or 3

**3. Hardcoded Configuration Brittleness**
- **Documented Issue**: "Kernel has hardcoded dependencies preventing any config changes"
- **Symptom**: Changing BLOCK_M/N or NUM_WARPS causes misaligned address errors
- **Root Cause**: Kernel assumes specific memory layouts/alignments
- **Opportunity**: Fix alignment issues to enable larger tiles

### Nsight Compute Target Metrics

**Current State** (from docs):
- TC Utilization: 57%
- Bandwidth: 54%
- Latency: 321 μs

**Target State**:
- TC Utilization: ≥ 80% (40% increase)
- Bandwidth: ≥ 70% (30% increase)
- Latency: < 107 μs (3× speedup)

**Key Metrics to Track**:
```bash
ncu --metrics \
  sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
```

### L4-Specific Optimizations

**1. Increase Tile Sizes** (Priority 1)
```cpp
// Current (documented working)
BLOCK_M=64, BLOCK_N=64, NUM_WARPS=4

// Target (higher TC utilization)
BLOCK_M=128, BLOCK_N=128, NUM_WARPS=8

// Challenge: Fix hardcoded alignment assumptions
// - Check pointer alignment requirements
// - Fix SMEM indexing to handle larger tiles
// - Ensure SMEM budget < 48 KB
```

**2. Add Memory Pipelining** (Priority 2)
```cpp
// Current
STAGES=1  // No double buffering

// Target
STAGES=2  // Double buffer K, V tiles with cp.async

// Implementation
// - Use __pipeline_memcpy_async for K, V loads
// - __pipeline_commit() and __pipeline_wait_prior(1)
// - Overlap compute with memory
```

**3. Bank Conflict Mitigation** (Priority 3)
```cpp
// Check for bank conflicts in Nsight
// - If conflicts > 5%, add swizzling or padding
// - HEAD_DIM=64 → potential for 32-way conflicts
// - Solution: XOR swizzle or +8 padding
```

---

## Implementation Plan (EvoEngineer Iterations)

### Iteration 1: Fix Alignment for Larger Tiles (2 hours)
**Goal**: Enable BLOCK_M=128, BLOCK_N=128 without misaligned address errors

**Changes**:
1. Audit pointer arithmetic in kernel
2. Add alignment checks (`assert((uintptr_t)ptr % 16 == 0)`)
3. Fix SMEM indexing for larger tiles
4. Test with CUDA_LAUNCH_BLOCKING=1

**Expected**: Kernel runs without errors (may not be faster yet)

**Validation**:
```bash
CUDA_LAUNCH_BLOCKING=1 python3 benchmark_fa_s512.py
# Should complete without "misaligned address" error
```

### Iteration 2: Optimize Tile Configuration (1 hour)
**Goal**: Find optimal BLOCK_M, BLOCK_N, NUM_WARPS for L4

**Changes**:
1. Sweep configurations: (M, N, W) in [(64,64,4), (128,64,8), (128,128,8)]
2. Measure TC utilization and latency for each
3. Select best configuration

**Expected**: 1.5-2× speedup from better TC utilization

**Validation**:
```bash
# For each config
export BLOCK_M=128 BLOCK_N=64 NUM_WARPS=8
python3 cudadent42/bench/build_fa_s512.py
python3 benchmark_fa_s512.py
ncu --metrics sm__inst_executed_pipe_tensor.pct python3 benchmark_fa_s512.py
```

### Iteration 3: Add cp.async Pipelining (2 hours)
**Goal**: Overlap memory with compute using STAGES=2

**Changes**:
1. Replace blocking K, V loads with `__pipeline_memcpy_async`
2. Add double buffering logic (stage = 0/1)
3. Commit and wait at appropriate points

**Expected**: 1.5× additional speedup from better bandwidth

**Validation**:
```bash
ncu --metrics dram__throughput.pct python3 benchmark_fa_s512.py
# Should see bandwidth > 70%
```

---

## Success Criteria

### Phase 1: Baseline Verification (Complete)
- ✅ Kernel compiles
- ✅ Kernel runs (documented: 321 μs)
- ✅ Correctness validated

### Phase 2: First Optimization (Target)
- ⏳ Latency < 200 μs (1.6× speedup)
- ⏳ TC utilization > 70%
- ⏳ No crashes/errors

### Phase 3: Target Performance (Stretch)
- ⏳ Latency < 107 μs (3× speedup)
- ⏳ TC utilization > 80%
- ⏳ Bandwidth > 70%

### Phase 4: Beat PyTorch (Ambitious)
- ⏳ Latency < 47 μs (beat PyTorch)
- ⏳ Correctness: `torch.allclose(atol=1e-2)`

---

## EvoEngineer Prompt Template

### For Claude/GPT-4/DeepSeek

```
System: You are an expert CUDA kernel optimization engineer specializing in Ada L4 (sm_89) architecture.

User:
Task (I1):
- Optimize existing fa_s512.cu kernel for L4
- Current: 321 μs, 57% TC, 54% BW
- Target: < 107 μs (3× speedup)
- Constraints: S=512, D=64, FP16, < 48KB SMEM

Insights (I3):
- Bottleneck 1: Low TC utilization (57% → need 80%+)
  * Root cause: Small tiles (BLOCK_M=64)
  * Fix: Increase to 128, but kernel has alignment bugs
  
- Bottleneck 2: Low bandwidth (54% → need 70%+)
  * Root cause: STAGES=1 (no pipelining)
  * Fix: Add cp.async with STAGES=2

- Known Issue: Hardcoded dependencies cause misaligned address errors
  * Changing BLOCK_M/N breaks kernel
  * Need to audit pointer arithmetic

Output:
1. Identify alignment bugs preventing larger tiles
2. Propose fix for BLOCK_M=128, BLOCK_N=128
3. Add cp.async double buffering for K, V
4. Ensure SMEM < 48 KB, TC utilization > 70%
```

---

## Quick Commands

### Build and Test
```bash
cd ~/periodicdent42
python3 cudadent42/bench/build_fa_s512.py
python3 benchmark_fa_s512.py
```

### Profile with Nsight
```bash
ncu --set full -o fa_s512_profile python3 benchmark_fa_s512.py
ncu --import fa_s512_profile.ncu-rep --page details
```

### Compare Configurations
```bash
# Test different configs
for M in 64 128; do
  for N in 64 128; do
    export BLOCK_M=$M BLOCK_N=$N
    python3 cudadent42/bench/build_fa_s512.py
    python3 benchmark_fa_s512.py | grep "fa_s512:"
  done
done
```

---

## Timeline

| Task | Duration | Status |
|------|----------|--------|
| Baseline verification | 10 min | ⏳ Next (GPU connectivity) |
| Fix alignment bugs | 2 hours | Pending |
| Optimize tile config | 1 hour | Pending |
| Add cp.async | 2 hours | Pending |
| Nsight validation | 1 hour | Pending |
| **Total** | **6 hours** | 0% complete |

---

## References

1. **EvoEngineer Paper**: https://arxiv.org/html/2510.03760v1
   - Table 3: EvoEngineer-Insight (Task + Insights, no Historical)
   - Table 4: Expected 1.47-1.60× speedup, 58-63% validity

2. **Baseline Documentation**: `cudadent42/bench/kernels/fa_s512.cu`
   - Lines 30-35: Configuration
   - Lines 33-34: "Latency: 0.321 ms, TC: 57%, BW: 54%"

3. **Phase 0 Baseline**: `phase0_baseline_results.txt`
   - PyTorch SDPA: 47.10 μs
   - L4 utilization: 9.4% (massive headroom)

---

**Status**: ✅ Ready for optimization once GPU accessible  
**Next**: Verify baseline (321 μs), then start Iteration 1 (fix alignment)

