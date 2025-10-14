# Loop 1 Optimization Plan - FlashAttention Inverted Kernel

**Date**: October 14, 2025  
**Duration**: 14 hours (~$7.14 GPU cost)  
**Engineer**: Expert CUDA kernel engineer (best-in-class approach)  
**GPU**: L4 (Ada Lovelace, SM 8.9)

---

## 🎯 Mission: Close 8.3× Performance Gap

**Current State**:
- Correctness: ✅ PERFECT (7/7 tests passed, 0 bugs)
- Performance: ⚠️ 0.12× vs PyTorch SDPA (8.3× slower)
  - PyTorch SDPA: 0.0637 ms
  - Our kernel: 0.5257 ms

**Target State**:
- Correctness: ✅ Maintain (0 bugs)
- Performance: 🎯 **1.0-1.2× vs PyTorch SDPA** (competitive)
  - Target: 0.053-0.064 ms
  - Required speedup: **8.3-10×**

---

## 📋 Systematic Optimization Loop (CUDA Expert Approach)

### Methodology: Profile → Identify → Fix → Measure → Repeat

Following **CUDA Engineering Cookbook** best practices:
1. **Profile first** (understand bottleneck, don't guess)
2. **Fix one thing** (isolate impact)
3. **Validate correctness** (all 7 tests must pass)
4. **Measure performance** (bootstrap CIs, statistical significance)
5. **Document & commit** (reproducible, honest reporting)

---

## 🔥 Priority 1: Add Tensor Core Support (Target: 6-8× speedup)

### Root Cause (Hypothesis → Verify with ncu)
- **Current**: Manual FP16 arithmetic (`half_to_float`, scalar multiply/add)
- **SDPA**: Uses `wmma::mma_sync` (Tensor Core instructions)
- **Expected impact**: 6-8× speedup (Tensor Cores are ~8× faster than FP16 CUDA cores)

### Implementation Plan (2-3 hours)

**Step 1: Profile Current Kernel** (30 min)
- Fix ncu PATH issue (`/usr/local/cuda/bin/ncu`)
- Capture baseline metrics:
  - SM throughput (%)
  - DRAM throughput (%)
  - Tensor Core utilization (should be 0% currently)
  - Warp occupancy
  - Register usage
- Save to `artifacts/ncu/inverted_v1_baseline.ncu-rep`

**Step 2: Add Tensor Core Support to QK Matmul** (60 min)
- Replace `compute_QK()` with `wmma` fragments
- Use `m16n8k16` (Ada Lovelace FP16 Tensor Cores)
- Layout: Q rows, K^T columns
- Accumulate in FP32 (for numerical stability)

**Step 3: Add Tensor Core Support to SV Matmul** (60 min)
- Replace `compute_SV()` with `wmma` fragments
- Layout: S (attention weights) × V
- Keep online softmax in FP32 shared memory

**Step 4: Validate** (30 min)
- Run all 7 correctness tests (must pass with same tolerance)
- Benchmark performance (N=100, bootstrap CIs)
- Profile with ncu (expect Tensor Core utilization ~40-60%)

**Expected Outcome**:
- Latency: 0.5257 ms → **~0.08 ms** (6.5× speedup)
- Tensor Core utilization: 0% → 40-60%
- Correctness: All 7 tests still pass

---

## 🎯 Priority 2: Optimize Tile Sizes (Target: 1.5-2× speedup)

### Root Cause (Hypothesis → Verify with ncu)
- **Current**: TILE_M=32, TILE_N=32 (conservative, fits in 14KB SMEM)
- **Optimal**: TILE_M=64, TILE_N=64 with double-buffering (use 40KB of 48KB)
- **Expected impact**: 1.5-2× speedup (better data reuse, higher occupancy)

### Implementation Plan (2-3 hours)

**Step 1: Profile Post-Priority-1 Kernel** (30 min)
- Identify new bottleneck (likely SMEM bandwidth or occupancy)
- Check: Is SMEM a bottleneck? Is occupancy < 50%?

**Step 2: Increase Tile Sizes** (60 min)
- Change TILE_M=64, TILE_N=64
- Verify SMEM usage < 48KB:
  - Q: 64 × 64 × 2 = 8KB
  - K: 64 × 64 × 2 = 8KB
  - V: 64 × 64 × 2 = 8KB
  - S: 64 × 64 × 2 = 8KB
  - Total: 32KB << 48KB ✓

**Step 3: Add Double-Buffering (STAGES=2)** (60 min)
- Pipeline: Load tile N+1 while computing tile N
- Uses 2× SMEM (32KB × 2 = 64KB) → too much!
- **Adjust**: Keep TILE_M=64, TILE_N=64, STAGES=1 (fits in 48KB)
- Or: TILE_M=48, TILE_N=48, STAGES=2 (fits in 48KB)

**Step 4: Validate** (30 min)
- Run all 7 correctness tests
- Benchmark performance (N=100, bootstrap CIs)
- Profile with ncu (expect higher occupancy, better SMEM reuse)

**Expected Outcome**:
- Latency: ~0.08 ms → **~0.05 ms** (1.6× speedup)
- Occupancy: Increase by 10-20%
- Correctness: All 7 tests still pass

---

## 🚀 Priority 3: Async Memory Copies (Target: 1.2-1.5× speedup)

### Root Cause (Hypothesis → Verify with ncu)
- **Current**: Synchronous loads (memory stalls compute)
- **Optimal**: `cp.async` for Q/K/V (overlap memory with compute)
- **Expected impact**: 1.2-1.5× speedup (hide memory latency)

### Implementation Plan (2-3 hours)

**Step 1: Profile Post-Priority-2 Kernel** (30 min)
- Check: Are there memory stalls? Is DRAM bandwidth high?
- Identify: Which loads benefit most from async?

**Step 2: Add cp.async for Q/K/V Loads** (90 min)
- Use `__pipeline_memcpy_async` for tile loads
- Add `__pipeline_wait_group<0>()` before compute
- Requires: 16-byte aligned addresses (validate in bindings)

**Step 3: Optimize Async Copy Schedule** (60 min)
- Overlap: Load tile N+1 while computing tile N (software pipelining)
- May need: Additional shared memory for double-buffering

**Step 4: Validate** (30 min)
- Run all 7 correctness tests
- Benchmark performance (N=100, bootstrap CIs)
- Profile with ncu (expect lower memory stalls)

**Expected Outcome**:
- Latency: ~0.05 ms → **~0.04 ms** (1.25× speedup)
- Memory stalls: Reduced by 20-30%
- Correctness: All 7 tests still pass

---

## 📊 Success Criteria (Exit Conditions)

### Minimum Bar (Must Achieve)
- ✅ All 7 correctness tests pass (max error < 0.02)
- ✅ 0 known bugs maintained
- ✅ Performance: **≥ 0.8× vs PyTorch SDPA** (within 20%)
  - Target: ≤ 0.08 ms latency
  - Current SDPA: 0.0637 ms

### Stretch Goal (Best Case)
- ✅ Performance: **1.0-1.2× vs PyTorch SDPA** (competitive or better)
  - Target: 0.053-0.064 ms latency
  - Statistical significance: Non-overlapping 95% CIs

### Excellence (Publication-Ready)
- ✅ Complete profiling artifacts (baseline, Priority 1, 2, 3)
- ✅ Ablation study table (each optimization's impact)
- ✅ Honest documentation (what worked, what didn't, why)

---

## 🛠️ Tools & Resources

### Profiling
- **ncu** (Nsight Compute): `/usr/local/cuda/bin/ncu`
- **Metrics**: SM throughput, DRAM throughput, Tensor Core util, occupancy, register usage
- **Command**:
  ```bash
  ncu --set full --target-processes all \
      -o profile_inverted_v2 \
      python3 cudadent42/bench/test_fa_inverted_prod.py
  ```

### Benchmarking
- **Script**: `cudadent42/bench/test_fa_inverted_prod.py`
- **Config**: N=100, warmup=20, bootstrap CIs, significance testing
- **Baseline**: `.ci/baseline_inverted_v1.json` (0.5257 ms)

### Correctness
- **Test Suite**: 7 shapes (128-2048 tokens, causal/non-causal)
- **Tolerance**: 2e-2 (FP16 conservative)
- **Validation**: Must pass all 7 tests after each change

### Documentation
- **Cookbook**: `docs/CUDA_COOKBOOK.md`
- **Methodology**: `cudadent42/docs/OPTIMIZATION_THROUGH_INVERSION.md`
- **Best Practices**: `docs/perf_guardrails.md`

---

## 📈 Expected Timeline (14 hours)

### Phase 1: Baseline & Profiling (1 hour)
- Fix ncu PATH
- Profile current kernel
- Capture baseline metrics
- Document bottlenecks

### Phase 2: Priority 1 - Tensor Cores (3 hours)
- Implement wmma for QK matmul (1 hour)
- Implement wmma for SV matmul (1 hour)
- Validate correctness + performance (1 hour)
- Expected: 0.5257 ms → 0.08 ms (6.5× speedup)

### Phase 3: Priority 2 - Tile Sizes (3 hours)
- Profile post-Priority-1 (0.5 hour)
- Increase tile sizes (1.5 hours)
- Validate correctness + performance (1 hour)
- Expected: 0.08 ms → 0.05 ms (1.6× speedup)

### Phase 4: Priority 3 - Async Copies (3 hours)
- Profile post-Priority-2 (0.5 hour)
- Implement cp.async (1.5 hours)
- Validate correctness + performance (1 hour)
- Expected: 0.05 ms → 0.04 ms (1.25× speedup)

### Phase 5: Final Validation & Documentation (2 hours)
- Full profiling suite (all versions)
- Ablation study table
- Performance comparison vs SDPA
- Update documentation
- Commit with honest results

### Buffer (2 hours)
- Debugging if needed
- Additional optimization iterations
- Profiling deep-dives

**Total: 14 hours**

---

## 💰 Budget

**GPU Costs**:
- L4: $0.51/hour × 14 hours = **$7.14**

**Expected ROI**:
- $7.14 → 10-13× speedup (0.5257 ms → ~0.04 ms)
- Publication-ready artifact (competitive with SDPA)
- Complete optimization methodology demonstration

---

## 🎓 Success Metrics

### Technical
- ✅ 10-13× combined speedup (0.5257 → 0.04 ms)
- ✅ Match or beat PyTorch SDPA (0.0637 ms)
- ✅ All 7 correctness tests pass
- ✅ 0 known bugs maintained

### Scientific
- ✅ Complete ablation study (each optimization's impact)
- ✅ Profiling artifacts (baseline, v2, v3, v4)
- ✅ Statistical validation (bootstrap CIs, significance)
- ✅ Honest documentation (what worked, limitations)

### Engineering
- ✅ Systematic approach (profile → fix → validate → repeat)
- ✅ One change at a time (isolate impact)
- ✅ Reproducible (all configs version-controlled)
- ✅ Professional (clean commits, clear documentation)

---

## 🚀 Let's Begin!

**Status**: GPU starting (34.10.174.115)  
**Next**: Phase 1 - Baseline & Profiling (fix ncu PATH, capture metrics)  
**Duration**: ~1 hour  
**Cost**: $0.51

**This is what a best-in-class CUDA kernel engineer would do on Oct 14, 2025.** 🔥

