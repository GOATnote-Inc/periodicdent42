# FlashCore L4 GPU Findings & Action Plan

**Date**: October 21, 2025  
**Instance**: cudadent42-l4-dev (us-west1-c)  
**Status**: ✅ Executing on L4, Correctness Issues Found

---

## 🎯 Findings Summary

### ✅ GPU Environment (PASSED)
```
Device:        NVIDIA L4
Driver:        535.274.02  
Memory:        23 GB
CUDA:          12.2
PyTorch:       2.5.1+cu121
Compute Cap:   8.9 (sm_89) ✅
```

### ✅ PyTorch SDPA Baseline (MEASURED)
```
Shape: B=1, H=8, S=512, D=64

PyTorch SDPA Performance:
  p50: 45.09 μs
  p90: 55.19 μs
  min: 44.03 μs
  max: 100.35 μs

This is our target to beat! 🎯
```

### ❌ FP8 Stage C Kernel (CORRECTNESS FAILED)
```
Build Status: ✅ Successful
  - Registers: 96 (Stage-2), 100 (Stage-5 WS)
  - SMEM: 37 KB
  - No spills

Correctness Results:
  small (64):   ✅ PASS (max_err=0.048)
  mission (512): ❌ FAIL (max_err=0.116)
  long (2048):  ❌ FAIL (max_err=0.167)

Performance (where it passed):
  small: 107.52 μs (0.6× vs PyTorch, 2× slower)
  
Root Cause: FP8 E4M3 quantization (±448 range) causes:
  1. Overflow/underflow on long sequences
  2. Error accumulation in 512-step online softmax
  3. NaN propagation despite numerical guards
```

---

## 🔧 Recommended Fix: Switch to FP16 Path

### Why FP16?
1. ✅ **Wider dynamic range**: ±65,504 vs FP8's ±448
2. ✅ **Better precision**: 10 mantissa bits vs 3
3. ✅ **Phase D reported GREEN**: FP16 minimal passed all tests
4. ✅ **Native L4 support**: Tensor Cores work with FP16

### Available FP16 Kernels
```bash
cudadent42/bench/kernels/sdpa_minimal_fp16.cu          # Phase D baseline
cudadent42/bench/kernels/sdpa_minimal_fp16_bindings.cpp
cudadent42/bench/kernels/fa_minimal.cu                  # FlashAttention minimal
cudadent42/bench/kernels/fa_minimal_bindings.cpp
```

---

## 🚀 Action Plan (Next 30 Minutes)

### Step 1: Build FP16 Minimal (5 min)
```bash
cd ~/periodicdent42
# Find the build script for FP16 minimal
python3 bench/build_phaseD.py  # or similar

# Expected: Clean build, no errors
```

### Step 2: Run Correctness Tests (10 min)
```python
# Test script (create if needed):
python3 tests/test_sdpa_kernel_correctness.py

# Expected results:
#   small:   ✅ PASS (max_err < 0.06)
#   mission: ✅ PASS (max_err < 0.06)  ← KEY FIX
#   long:    ✅ PASS (max_err < 0.06)
```

### Step 3: Benchmark Performance (10 min)
```python
# Benchmark mission shape (B=1, H=8, S=512, D=64)
python3 scripts/bench_sdpa.py --shape mission --iters 100

# Baseline expectation (scalar FP16, no WMMA):
#   Latency: ~800-1500 μs
#   vs PyTorch: ~18-33× slower (expected for unoptimized)
```

### Step 4: Document Results (5 min)
```
Create: FLASHCORE_L4_FP16_BASELINE.md
Include:
  - PTXAS stats
  - Correctness results (all shapes)
  - Performance (p50/p90 vs PyTorch)
  - Next optimization steps
```

---

## 📊 Expected Progression

### Phase 0: FP16 Baseline (Current)
```
Target: GET IT WORKING ✅
Latency: ~800-1500 μs (expected)
vs PyTorch: 18-33× slower
Correctness: ALL PASS
```

### Phase 1: WMMA Tensor Cores (Week 1)
```
Target: 10× speedup from baseline
Latency: ~80-150 μs
vs PyTorch: 2-3× slower (still needs fusion)
Technique: Replace Q@K^T and P@V with WMMA
```

### Phase 2: FlashAttention Fusion (Week 2)
```
Target: <60 μs (≥15× vs old PyTorch 870 μs) ✅ PROJECT GOAL
Latency: ~40-60 μs
vs PyTorch: 0.7-1.3× (comparable to SDPA!)
Technique: Tile-based fusion, online softmax, minimize global mem
```

### Phase 3: Advanced Opts (Stretch)
```
Target: Beat PyTorch SDPA
Latency: <45 μs
vs PyTorch: >1.0× (FASTER than SDPA!)
Technique: Warp specialization, persistent CTAs, XOR swizzle
```

---

## 🔍 Technical Notes

### Why FP8 Failed
```cuda
// FP8 E4M3 format:
// - Exponent: 4 bits → range [2^-6, 2^8] ≈ [0.015, 448]
// - Mantissa: 3 bits → ~1% precision

// Problem in online softmax:
float m_new = max(m_old, max(scores));  // OK
float rescale = exp(m_old - m_new);      // Can underflow if m_old << m_new
l_i *= rescale;                          // Accumulates error over 512 steps
o_i *= rescale;                          // NaN if rescale = 0

// On long sequences (S=512):
// - Many rescale operations compound
// - Eventually: rescale → 0 or Inf
// - Result: NaN propagation
```

### Why FP16 Works
```cuda
// FP16 format:
// - Exponent: 5 bits → range [2^-14, 2^15] ≈ [6e-5, 65504]
// - Mantissa: 10 bits → ~0.1% precision

// Same algorithm, but:
// - Wider range prevents underflow
// - Better precision reduces accumulation error
// - L4 Tensor Cores natively support FP16
```

---

## 🎯 Success Criteria

### Immediate (Phase 0 FP16 Baseline)
- ✅ Build: No errors, PTXAS clean
- ✅ Correctness: ALL shapes PASS (max_err < 0.06)
- ✅ Performance: Measured and documented (even if slow)
- ✅ Baseline: Establishes starting point for optimization

### Short-term (Phase 1 WMMA, 40 hours)
- ✅ Correctness: Maintained (all tests pass)
- ✅ Performance: <150 μs (10× improvement)
- ✅ PTXAS: TC utilization ≥50%

### Primary Goal (Phase 2 Fusion, 80 hours total)
- ✅ Performance: <60 μs (≥15× vs 870 μs old PyTorch)
- ✅ Correctness: All tests pass
- ✅ Documentation: Complete

---

## 📝 Next Immediate Actions

**RIGHT NOW (on L4)**:

1. Find/build FP16 minimal kernel
2. Run correctness tests
3. Measure baseline performance
4. Document results

**Then**: Start Phase 1 (WMMA) with clean FP16 baseline

---

**Status**: Findings documented, ready to execute FP16 baseline  
**Time on L4**: 15 minutes so far  
**GPU Cost**: $0.19 (at $0.75/hour)  
**Next**: Build FP16, get GREEN correctness ✅

