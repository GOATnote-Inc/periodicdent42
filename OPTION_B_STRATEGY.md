# Option B Strategy: Official FlashAttention-2 → EvoEngineer Optimization

**Date**: October 16, 2025  
**Status**: 🚀 IN PROGRESS - Installing FlashAttention-2  
**Method**: Use proven kernel, measure baseline, apply EvoEngineer-Insight

---

## Why Option B?

After discovering the baseline `fa_s512.cu` was fundamentally broken (crashes for ALL batch sizes), we pivoted to **Option B**: Use a proven working kernel and optimize it with EvoEngineer.

### Options Considered
- ❌ **Option A**: Fix broken `fa_s512.cu` first (2-4 hours debugging unknown bugs)
- ✅ **Option B**: Use official FlashAttention-2 (1-2 hours setup + proven to work)
- ❌ **Option C**: Return to `fa_s512_v3.cu` (4-6 hours, also had issues)
- ❌ **Option D**: Document as research (interesting, but doesn't achieve speed goal)

---

## Implementation Plan

### Phase 1: Install & Validate (30-45 minutes) ⏳ IN PROGRESS

**Step 1.1**: Install official flash-attn package ⏳ RUNNING
```bash
cd ~/periodicdent42/ext/flash-attention-2
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="8.9" pip install -e .
```

**Step 1.2**: Validate correctness & measure baseline ⏳ PENDING
```python
from flash_attn import flash_attn_func
# Test: B=4, H=8, S=512, D=64
# Compare with PyTorch SDPA
# Expected: 5-10 μs (vs 47 μs PyTorch)
```

**Success Criteria**:
- ✅ Compiles for sm_89 (L4 Ada)
- ✅ `torch.allclose(atol=1e-2)` passes
- ✅ Latency < 47 μs (faster than PyTorch)

### Phase 2: Extract Kernel for Optimization (1-2 hours)

**Goal**: Get the specific forward kernel source that flash-attn uses for our config (hdim64, fp16, sm_80+)

**Files to extract**:
1. **Kernel**: `csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu` (13 lines, template inst)
2. **Header**: `csrc/flash_attn/src/flash_fwd_launch_template.h` (actual kernel code)
3. **Kernel impl**: `csrc/flash_attn/src/flash_fwd_kernel.h` (core algorithm)

**Create standalone version**:
- `cudadent42/bench/kernels/flash_attn_fwd_standalone.cu` (simplified, single file)
- Strip unnecessary templates/abstractions
- Keep core algorithm intact
- Target: ~500-1000 lines (vs FlashAttention-2's ~5000+ lines)

### Phase 3: EvoEngineer-Insight Analysis (1 hour)

**Apply EvoEngineer-Insight methodology**:

**I1 (Task Context)**:
- Goal: Optimize FlashAttention-2 forward kernel for L4
- Current: X μs (measured baseline from Phase 1)
- Target: < X/2 μs (2× speedup minimum)
- Constraints: S=512, D=64, FP16, sm_89, < 48KB SMEM

**I3 (Optimization Insights)**:
- Bottleneck 1: TBD after profiling (likely TC utilization or bandwidth)
- Bottleneck 2: TBD after Nsight Compute analysis
- L4-specific: Leverage 48MB L2, 242 TFLOPS TC, avoid bank conflicts

**Expected bottlenecks** (to be confirmed):
1. Tile sizes may not be optimal for L4's 48KB SMEM
2. cp.async pipeline depth may be suboptimal
3. Warp-level parallelism may have room for improvement

### Phase 4: Apply Optimizations (2-3 hours)

**Iteration 1**: Fix identified bottleneck #1  
**Iteration 2**: Optimize tile configuration  
**Iteration 3**: Add L4-specific optimizations (L2 persistence, etc.)

**Validation after each iteration**:
- ✅ Correctness: `torch.allclose(atol=1e-2)`
- ✅ Performance: Measure speedup
- ✅ Nsight: Validate metrics improve

---

## Expected Results

### Baseline (Phase 1)
```
FlashAttention-2: ~5-10 μs (official implementation)
PyTorch SDPA:     ~47 μs (measured earlier)
Speedup:          ~5-9x faster than PyTorch
```

### After EvoEngineer (Phase 4)
```
Optimized FA2: ~3-5 μs (2× speedup over FA2)
PyTorch SDPA:  ~47 μs
Speedup:       ~10-15x faster than PyTorch
```

**Stretch goal**: < 3 μs (approaching theoretical limit)

---

## Why FlashAttention-2?

### ✅ Proven Track Record
- Used in production by Meta, OpenAI, Google, etc.
- Extensively tested across architectures
- Optimized by experts (Tri Dao et al.)

### ✅ Known Performance
- Benchmarked extensively
- Well-documented performance characteristics
- Established baselines to beat

### ✅ Scientific Integrity
- Can measure true baseline (not "documented" fake baseline)
- Validates EvoEngineer on WORKING code
- Shows optimization methodology works

### ✅ Publication-Ready
- Comparison against industry standard
- Can cite FlashAttention-2 paper
- Demonstrates systematic improvement over SOTA

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| **1.1** | Install flash-attn | 30-45 min | ⏳ RUNNING |
| **1.2** | Validate & baseline | 10 min | ⏳ PENDING |
| **2** | Extract standalone kernel | 1-2 hours | ⏳ PENDING |
| **3** | EvoEngineer analysis | 1 hour | ⏳ PENDING |
| **4** | Apply optimizations | 2-3 hours | ⏳ PENDING |
| **Total** | | **4-6 hours** | 10% complete |

---

## Current Status

### Installation Progress ⏳
```
Screen session: 4684.flash_install
Command: ./install_flash_attn2.sh
Log: ~/periodicdent42/flash_install_output.txt
Expected completion: 10-15 minutes (started 16:10 UTC)
```

**Waiting for**:
1. ✅ Ninja build system compiles kernels
2. ✅ pip installs package
3. ✅ Validation script runs
4. ✅ Baseline measurements

**Next action**: Check progress in 2 minutes

---

## Key Differences from Iteration 1

### Iteration 1 (fa_s512.cu)
- ❌ Started with broken kernel
- ✅ Correct SMEM analysis
- ❌ Couldn't validate because baseline crashed
- **Lesson**: Always validate baseline first!

### Option B (FlashAttention-2)
- ✅ Start with proven working kernel
- ✅ Can measure true baseline
- ✅ Can validate optimizations incrementally
- ✅ Scientific integrity (real measurements)

---

## EvoEngineer Methodology (Planned)

### What Worked in Iteration 1 ✅
1. **SMEM budget calculation** - 100% accurate (49,152 bytes predicted = measured)
2. **Root cause identification** - Correctly identified SMEM overflow
3. **Solution design** - FP16 S_smem was correct fix

### What to Apply to FlashAttention-2
1. **Pre-validate baseline** - Measure performance BEFORE optimization
2. **Systematic analysis** - Use I1 + I3 (Task + Insights)
3. **Incremental validation** - Test after each change
4. **L4-specific optimizations** - Leverage 48MB L2, avoid bank conflicts

---

## Success Criteria

### Minimum (Must Achieve)
- ✅ FlashAttention-2 baseline < 47 μs (faster than PyTorch)
- ✅ Correctness validated (`torch.allclose`)
- ✅ Optimization achieves ≥ 1.5× speedup over FA2 baseline

### Target (Goal)
- ✅ Optimized kernel ≥ 2× faster than FA2 baseline
- ✅ ≥ 10× faster than PyTorch SDPA (47 μs → < 5 μs)
- ✅ All optimization insights documented for publication

### Stretch (Ambitious)
- ✅ ≥ 3× faster than FA2 baseline
- ✅ < 3 μs latency (approaching theoretical limit)
- ✅ Technique generalizes to other shapes/dtypes

---

## Documentation Plan

### Files Created (so far)
1. `ITER1_CRITICAL_FINDINGS.md` - Why baseline was broken
2. `OPTION_B_STRATEGY.md` (this file) - New approach
3. `install_flash_attn2.sh` - Installation script

### Files to Create (upcoming)
4. `FA2_BASELINE_RESULTS.md` - Phase 1 measurements
5. `FA2_STANDALONE_KERNEL.cu` - Extracted kernel
6. `FA2_EVOENG_ANALYSIS.md` - EvoEngineer-Insight analysis
7. `FA2_OPTIMIZATION_LOG.md` - Iteration-by-iteration improvements
8. `FA2_FINAL_RESULTS.md` - Complete comparison

---

## Git History

```
be473f6 feat: Add FlashAttention-2 installation script for Option B
d94ad4d docs: Critical findings - baseline fa_s512.cu kernel is broken
58aefaa fix: Update build script to use Iteration 1 config
2e2f28c feat(iter1): Fix SMEM overflow, enable BLOCK_M=128
```

**Branch**: `feature/v3_clean_slate`  
**Status**: All changes committed and pushed

---

**Next**: Monitor installation progress, then proceed to Phase 1.2 (validation)

**ETA**: 4-6 hours to complete full optimization pipeline

**Confidence**: HIGH - Using proven kernel eliminates "baseline is broken" risk

