# FlashCore: Beat PyTorch SDPA Plan

**Date**: October 22, 2025  
**Target**: **<26 μs** (beat PyTorch SDPA on L4)  
**Current**: 279 μs (32×32 WMMA, working)  
**Gap**: 10.7× speedup needed  
**Philosophy**: Stand on shoulders of giants (periodicdent42 + FlashAttention)

---

## 🎯 **Mission Recalibration**

### **The Truth**
- PyTorch SDPA: **26 μs** on L4 (our competition)
- Our target: **<26 μs** (beat SOTA)
- Current 32×32: 279 μs (working, but 10.7× too slow)

### **The Strategy**
**LEVERAGE EXISTING WINNERS** from `periodicdent42`:
1. ✅ `sdpa_fp8_stage_c_wmma.cu` (1323 lines, production-quality)
2. ✅ `detail/cp_async.hpp` (proven async copy)
3. ✅ EvoEngineer infrastructure (autotune working)
4. ✅ NCU profiling scripts (know what to measure)

**DON'T**: Debug mysterious async bugs for 8+ hours  
**DO**: Copy working patterns, adapt, profile, iterate

---

## 📊 **Gap Analysis (Evidence-Based)**

From `periodicdent42` NCU profiling and benchmarks:

| Optimization | Current | After | Gain | Confidence |
|--------------|---------|-------|------|------------|
| **Baseline** | 279 μs | — | — | ✅ 100% |
| **64×64 tiles** | 279 μs | ~140 μs | 2.0× | 70% (SMEM issue?) |
| **Fused softmax** | 140 μs | ~70 μs | 2.0× | 80% (proven) |
| **cp.async K/V** | 70 μs | ~35 μs | 2.0× | 75% (have code) |
| **Warp specialization** | 35 μs | ~**22 μs** | 1.6× | 60% (complex) |

**Compound**: 279 → 140 → 70 → 35 → **22 μs** ✅ **BEATS PYTORCH!**

---

## 🚀 **5-Phase Action Plan**

### **Phase 1: Port Proven WMMA from periodicdent42** (TODAY, 3-4h)

**DON'T**: Try to fix our 64×32 async kernel  
**DO**: Copy working `sdpa_fp8_stage_c_wmma.cu` WMMA loops

**Action**:
```bash
cd /Users/kiteboard/periodicdent42

# Copy WMMA reference
cp cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu flashcore/kernels/REFERENCE_wmma.cu

# Create new kernel based on working 32×32 + proven WMMA
cp flashcore/kernels/flashcore_fused_wmma.cu flashcore/kernels/flashcore_phase1_proven_wmma.cu

# Extract WMMA Q@K^T loop (lines 225-280 from reference)
# Extract WMMA P@V loop (lines 420-490 from reference)
# Adapt to our 32×32 tile size
```

**Success Criteria**:
- ✅ Builds (no errors)
- ✅ Correctness: error < 0.40
- ✅ Performance: 180-220 μs (1.3-1.5× from 279 μs)
- ✅ PTXAS: ≤120 regs, 0 spills

**Expected**: 279 → **~200 μs** (1.4× gain)

---

### **Phase 2: Fused Online Softmax** (Day 2, 4-6h)

**Reference**: FlashAttention-2 Algorithm 1 + `periodicdent42` Stage-C softmax

**Key Changes**:
1. Keep QK scores in registers (don't materialize to SMEM)
2. Online max/sum: `m_new = max(m_old, m_tile)`, `l_new = l_old * exp(m_old - m_new) + l_tile`
3. Rescale O accumulator on-the-fly

**Action**:
```bash
cp flashcore/kernels/flashcore_phase1_proven_wmma.cu flashcore/kernels/flashcore_phase2_fused_softmax.cu

# Add online softmax (reference: RESEARCH_SUMMARY.md lines 229-233)
# Remove sS SMEM buffer
# Keep m/l stats in registers per warp
```

**Success Criteria**:
- ✅ Error maintained (< 0.40)
- ✅ Performance: 90-120 μs (1.7-2.2× from Phase 1)

**Expected**: 200 → **~100 μs** (2.0× gain)

---

### **Phase 3: Proven cp.async (NOT our broken version)** (Day 3, 3-4h)

**Reference**: `periodicdent42/cudadent42/bench/kernels/detail/cp_async.hpp`

**DON'T**: Use our "fixed" kernel that still crashes  
**DO**: Copy working cp_async.hpp and use its API

**Action**:
```bash
cp cudadent42/bench/kernels/detail/cp_async.hpp flashcore/kernels/detail/

# In Phase 3 kernel:
#include "detail/cp_async.hpp"

// Use proven API (not __pipeline_memcpy_async directly)
cp_async_cg<128>(&sK[stage][...], &K_global[...]);
cp_async_commit_group();
cp_async_wait_group<STAGES-1>();
```

**Success Criteria**:
- ✅ Builds and RUNS (no crashes!)
- ✅ Performance: 50-70 μs (1.4-2.0× from Phase 2)

**Expected**: 100 → **~60 μs** (1.7× gain)

---

### **Phase 4: Warp Specialization** (Day 4-5, 6-8h)

**Reference**: `periodicdent42` Stage-5 warp specialization

**Pattern**:
```cuda
if (warp_id < NUM_COMPUTE_WARPS) {
    // Compute warps: WMMA Q@K^T, softmax, P@V
} else {
    // Load warps: cp.async prefetch K/V
}
```

**Action**:
```bash
cp flashcore/kernels/flashcore_phase3_cpasync.cu flashcore/kernels/flashcore_phase4_warp_spec.cu

# Add warp role assignment
# Producer warps: async copy only
# Consumer warps: WMMA compute only
```

**Success Criteria**:
- ✅ Error maintained
- ✅ Performance: 30-45 μs (1.3-2.0× from Phase 3)
- ✅ NCU: Reduced barriers (< 20 vs baseline 48)

**Expected**: 60 → **~35 μs** (1.7× gain)

---

### **Phase 5: Micro-Optimizations** (Day 6-7, 4-6h)

**Techniques** (from LAUNCH_PLAN.md):
1. XOR swizzling (bank conflict avoidance)
2. Fast math (approx exp if accuracy permits)
3. Register blocking (reduce SMEM roundtrips)
4. Launch bounds tuning

**Action**:
```bash
# Try multiple configs with EvoEngineer
cd flashcore
python search/autotune_flashcore.py --base phase4_warp_spec

# Test elite configs
# Keep best 3
```

**Success Criteria**:
- ✅ Performance: **<26 μs** ✅ **BEATS PYTORCH!**
- ✅ Correctness maintained

**Expected**: 35 → **~22 μs** (1.6× gain)

---

## 📋 **Daily Execution Checklist**

### **Every Morning** (5 min)
```bash
cd /Users/kiteboard/periodicdent42/flashcore
source scripts/env_cuda_l4.sh
bash scripts/preflight.sh
git status -sb
```

### **Every Phase** (Standard TDD Loop)
```bash
# 1. Implement (edit kernel)
# 2. Build
python build_phaseN.py

# 3. Test correctness
pytest tests/test_correctness.py -v

# 4. Benchmark
python benchmarks/benchmark_latency.py --shape mission --iters 100

# 5. Profile (if needed)
ncu --set full --launch-skip 10 --launch-count 1 \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    python benchmarks/benchmark_latency.py --shape mission --iters 1

# 6. Document
# Update progress, commit with perf numbers

# 7. Proceed or debug
# If blocked > 3h, skip and try next phase
```

### **Every Evening** (10 min)
```bash
# Commit progress
git add flashcore/
git commit -m "feat(flashcore): Phase N - XXX μs achieved (Nx speedup)"

# Update dashboard
echo "Phase N: XXX μs" >> FLASHCORE_PROGRESS.md

# Plan tomorrow
cat FLASHCORE_BEAT_PYTORCH_PLAN.md | grep "Day $(date +%d)"
```

---

## 🎯 **Success Criteria (Final)**

| Metric | Target | Why |
|--------|--------|-----|
| **Latency (p50)** | **<26 μs** | Beat PyTorch SDPA ✅ |
| **Correctness** | error < 0.40 | Acceptable FP16 tolerance |
| **Resource Usage** | ≤120 regs, ≤64KB SMEM | Fit on L4 without issues |
| **Reproducible** | JSON artifacts + git SHA | Science, not magic |

**Grade System**:
- <26 μs: **A** (beats PyTorch)
- 26-35 μs: **B+** (competitive)
- 35-50 μs: **B** (good, but slower than PyTorch)
- >50 μs: **C** (learned, but not production-ready)

---

## 🔬 **Why This Will Work**

### **Evidence From periodicdent42**
1. ✅ Stage-C WMMA: 656 μs (with FP8 overhead)
2. ✅ FP16 version should be faster (no quant/dequant)
3. ✅ Warp specialization: 14% time savings proven
4. ✅ cp.async code EXISTS and WORKS

### **Evidence From Literature**
1. ✅ FlashAttention-2: 15-30 μs on A100 (similar architecture)
2. ✅ L4 is Ada (newer than A100), should be competitive
3. ✅ Fused attention: 2-4× faster than unfused (proven)
4. ✅ WMMA: 5-10× faster than scalar (proven)

### **Our Advantages**
1. ✅ Working 32×32 baseline (279 μs, correct)
2. ✅ Access to L4 GPU (can iterate fast)
3. ✅ Proven reference implementations
4. ✅ EvoEngineer autotune (can explore config space)

**Confidence**: 75% for <26 μs with all 5 phases

---

## ⚠️ **Risk Mitigation**

### **If Phase 1 Fails** (WMMA port)
- **Fallback**: Keep current 32×32, try Phase 2 (fused softmax) directly
- **Time Loss**: 3-4h
- **Impact**: Medium (WMMA is big win, but softmax fusion might compensate)

### **If Phase 3 Fails** (cp.async)
- **Fallback**: Skip async, proceed to Phase 4 (warp spec)
- **Expected**: 100 → ~45 μs (still competitive!)
- **Impact**: Low (warp spec alone may be sufficient)

### **If Phase 4 Fails** (warp spec)
- **Fallback**: Accept Phase 3 result (~60 μs)
- **Expected**: Slower than PyTorch, but 4.6× from baseline
- **Impact**: Medium (grade B instead of A)

### **Overall Strategy**
- Each phase is independent
- Can skip problematic phases
- Even 3/5 phases successful = ~60 μs (good result)

---

## 📚 **Key References (Copy-Paste Ready)**

### **WMMA Q@K^T** (Phase 1)
```bash
cat cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu | sed -n '225,280p'
```

### **Fused Softmax** (Phase 2)
```bash
cat cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu | sed -n '29,32p'
# Also: FlashAttention-2 paper Algorithm 1
```

### **cp.async Wrapper** (Phase 3)
```bash
cat cudadent42/bench/kernels/detail/cp_async.hpp
```

### **Warp Specialization** (Phase 4)
```bash
cat cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu | grep -A 50 "USE_WARP_SPECIALIZATION"
```

### **EvoEngineer Autotune** (Phase 5)
```bash
cat scripts/evo_full_iteration.py
```

---

## 🚀 **START NOW: Phase 1 Commands**

```bash
cd /Users/kiteboard/periodicdent42

# Setup
source scripts/env_cuda_l4.sh
bash scripts/preflight.sh

# Copy reference
cp cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu flashcore/kernels/REFERENCE_wmma.cu

# Create Phase 1 kernel (from working 32×32)
cp flashcore/kernels/flashcore_fused_wmma.cu flashcore/kernels/flashcore_phase1_wmma.cu

# NOW: Edit flashcore_phase1_wmma.cu
# - Extract WMMA Q@K^T from REFERENCE_wmma.cu lines 225-280
# - Extract WMMA P@V from REFERENCE_wmma.cu lines 420-490
# - Adapt to our tile sizes

# Build
cd flashcore
cat > build_phase1.py << 'EOF'
import os
from torch.utils.cpp_extension import load

def build_phase1():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [
        os.path.join(script_dir, 'kernels', 'flashcore_phase1_wmma.cu'),
        os.path.join(script_dir, 'kernels', 'flashcore_fused_wmma_bindings.cu'),
    ]
    
    extra_cuda_cflags = [
        '-O3', '-arch=sm_89', '--use_fast_math', '-lineinfo',
        '-Xptxas', '-v', '-std=c++17', '--expt-relaxed-constexpr',
    ]
    
    return load(name='flashcore_phase1', sources=sources, 
                extra_cuda_cflags=extra_cuda_cflags, verbose=True)

if __name__ == '__main__':
    build_phase1()
EOF

python build_phase1.py

# Test
pytest tests/test_correctness.py -v

# Benchmark
python benchmarks/benchmark_latency.py --shape mission --iters 100
```

**Expected Output**:
```
Mission shape (B=1, H=8, S=512, D=64):
  p50: 180-220 μs  (1.3-1.5× from 279 μs)
  error: <0.40
  ✅ PASS
```

**If successful**: Proceed to Phase 2 (fused softmax)  
**If blocked >3h**: Skip to Phase 2, try WMMA later

---

## 💡 **Key Insights**

### **What We Learned (Costly)**
1. ❌ Debugging async bugs without compute-sanitizer = 7+ hours wasted
2. ❌ "Fixing" alignment bugs blindly = more wasted time
3. ❌ Perfect is the enemy of good (279 μs works, build on it!)

### **What We're Doing Now (Smart)**
1. ✅ Copy PROVEN code from periodicdent42
2. ✅ Adapt incrementally (one phase at a time)
3. ✅ Test after each change (catch regressions early)
4. ✅ Profile to confirm wins (NCU metrics, not guesses)
5. ✅ Skip blockers (3h rule: move on if stuck)

### **Philosophy**
> "Standing on shoulders of giants means USING THEIR CODE, not reimplementing from scratch."

---

## 🏁 **Timeline & Milestones**

| Day | Phase | Target | Confidence | Cumulative |
|-----|-------|--------|------------|------------|
| **Today** | Phase 1 | 200 μs | 80% | 1.4× |
| **+1** | Phase 2 | 100 μs | 75% | 2.8× |
| **+2** | Phase 3 | 60 μs | 70% | 4.7× |
| **+3-4** | Phase 4 | 35 μs | 60% | 8.0× |
| **+5-6** | Phase 5 | **<26 μs** | 50% | **>10.7×** ✅ |

**Total Time**: 5-7 days (full-time) or 10-14 days (part-time)

**Checkpoints**:
- **Day 1**: If Phase 1 works → HIGH confidence for <50 μs
- **Day 2**: If Phase 2 works → MEDIUM confidence for <35 μs
- **Day 3**: If Phase 3 works → GOOD chance for <26 μs

---

## 🎯 **BOTTOM LINE**

**Old Plan**: Debug mysterious 64×32 async bugs forever  
**New Plan**: Copy proven code, adapt, test, profile, iterate

**Old Target**: <40 μs (slower than PyTorch)  
**New Target**: **<26 μs** (BEAT PyTorch)

**Old Timeline**: Unknown (debugging hell)  
**New Timeline**: 5-7 days (systematic execution)

**Confidence**: 75% for <26 μs, 90% for <35 μs, 95% for <50 μs

---

## ✅ **READY TO EXECUTE**

All infrastructure exists:
- ✅ Working baseline (279 μs)
- ✅ Reference implementations (periodicdent42)
- ✅ Build system (can copy/adapt)
- ✅ Test framework (15 test cases)
- ✅ Benchmark harness (100-run medians)
- ✅ GPU access (L4 on GCP)

**Next command**: Start Phase 1 (copy WMMA from reference)

**Let's build this! 🚀**

**Deeds, not words. Excellence, not excuses.**

