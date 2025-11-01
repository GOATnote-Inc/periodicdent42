# Honest Benchmark Status - November 1, 2025

**What We've Actually Tested vs What We Haven't**

---

## ✅ TESTED (Validated on H100)

| Comparison | Status | Results | Method | Date |
|------------|--------|---------|--------|------|
| **Custom vs CUTLASS 4.3** | ✅ YES | **610 vs 414 TFLOPS (+47%)** | CUDA Events, 100 runs | Nov 1, 2025 |
| **Custom vs cuBLAS ceiling** | ✅ YES | **610 vs 843 TFLOPS (72% efficiency)** | Hardware peak | Nov 1, 2025 |
| **Correctness vs PyTorch** | ✅ YES | **Max diff 0.002** | Numerical validation | Nov 1, 2025 |

---

## ❌ NOT TESTED (Critical Gaps)

| Comparison | Status | Why It Matters | Priority |
|------------|--------|----------------|----------|
| **Custom vs PyTorch sparse BSR** | ❌ NO | **If PyTorch is faster, no point in custom CUDA** | 🔴 CRITICAL |
| **Custom vs cuSPARSE** | ❌ NO | **NVIDIA's official sparse library - industry standard** | 🔴 CRITICAL |
| **Custom vs PyTorch dense (perf)** | ❌ NO | **Need to show sparsity exploitation** | 🟡 IMPORTANT |
| **Multiple matrix sizes** | ❌ NO | **Only tested 8K×8K - may be overfitted** | 🟡 IMPORTANT |
| **Different sparsity patterns** | ❌ NO | **Only tested topk=16 - not general** | 🟡 IMPORTANT |

---

## ⚠️ MISLEADING COMPARISONS (Need Clarification)

| Claim | Reality | Fix |
|-------|---------|-----|
| **"Beats FlashAttention-3"** | ❌ Never tested (different operation) | Remove or clarify "different op" |
| **"Faster than PyTorch"** | ⚠️ Only correctness, not performance | Clarify "matches correctness, perf unknown" |
| **"Production-ready"** | ❌ Single config, no error handling | Change to "research prototype" |

---

## 🎯 What We NEED to Test (This Week)

### Priority 1: Critical Baselines (REQUIRED)
```bash
# 1. PyTorch sparse BSR GEMM (THE critical baseline)
python benchmarks/compare_all_baselines.py

Expected: Should show if our kernel actually faster than PyTorch
Timeline: THIS WEEK (Nov 4-5)
```

```bash
# 2. cuSPARSE (NVIDIA's official sparse library)
./benchmarks/compare_cusparse.sh

Expected: Industry standard baseline
Timeline: THIS WEEK (Nov 4-5)
```

### Priority 2: Robustness Testing
```bash
# 3. Multiple matrix sizes (not just 8K×8K)
for size in 1024 2048 4096 8192 16384; do
    ./benchmark --M=$size --N=$size --K=$size
done

Expected: Performance curve vs matrix size
Timeline: Week of Nov 11
```

```bash
# 4. Different sparsity patterns
for topk in 4 8 16 32 64; do
    ./benchmark --topk=$topk
done

Expected: Performance vs sparsity
Timeline: Week of Nov 11
```

---

## 📊 Current Evidence Quality

| Claim | Evidence | Confidence | Notes |
|-------|----------|------------|-------|
| **"610 TFLOPS on H100"** | ✅ Measured (CUDA Events, 100 runs) | **HIGH** | Reproducible, <1% variance |
| **"+47% vs CUTLASS 4.3"** | ✅ Side-by-side test | **HIGH** | Same hardware, same compiler |
| **"72% hardware efficiency"** | ✅ vs cuBLAS ceiling | **HIGH** | Direct measurement |
| **"Faster than PyTorch"** | ❌ Not tested | **NONE** | No performance comparison done |
| **"Beats FlashAttention-3"** | ❌ Different operation | **NONE** | Apples-to-oranges comparison |
| **"Production-ready"** | ❌ Single config, no hardening | **LOW** | Research prototype only |

---

## 🔍 Why These Gaps Matter

### 1. PyTorch Sparse Baseline (CRITICAL)

**Why it matters:**
- PyTorch uses cuSPARSE under the hood
- If PyTorch sparse BSR is faster, custom kernel is pointless
- This is the #1 question reviewers will ask

**Current status:** ❌ Not tested
**Action:** Created `benchmarks/compare_all_baselines.py` to test

### 2. cuSPARSE Comparison (CRITICAL)

**Why it matters:**
- NVIDIA's official sparse library
- Industry standard baseline
- If we can't beat cuSPARSE, why use custom kernel?

**Current status:** ❌ Not tested
**Action:** Need to add cuSPARSE benchmark

### 3. Multiple Matrix Sizes (IMPORTANT)

**Why it matters:**
- May be overfitted to 8K×8K
- Real applications use many sizes
- Need to show general performance

**Current status:** ❌ Only tested 8K×8K
**Risk:** High - could be cherry-picked config

### 4. FlashAttention-3 Comparison (MISLEADING)

**Why it's problematic:**
- FA3 does **attention** (softmax + matmuls + normalization)
- Our kernel does **sparse GEMM** (one matrix multiply)
- Comparing apples to oranges
- Makes us look like we don't understand the domain

**Current status:** ❌ Should not be compared
**Action:** Remove or clarify "different operation"

---

## 🚦 Credibility Assessment

### What's Solid ✅
- CUTLASS comparison (same operation, validated)
- cuBLAS ceiling measurement (standard practice)
- Reproducibility (CUDA Events, SHA-256)
- Honest methodology (documented limitations)

### What's Questionable ⚠️
- Only one matrix size tested
- Only one sparsity pattern tested
- No PyTorch sparse comparison
- No cuSPARSE comparison

### What's Misleading ❌
- FlashAttention-3 comparison (different operation)
- "Production-ready" claim (research prototype)
- Implied "faster than PyTorch" (not tested)

---

## 📋 Action Plan

### This Week (Nov 4-8, 2025)

**Monday:**
- [ ] Run `compare_all_baselines.py` on H100
- [ ] Get PyTorch sparse BSR numbers
- [ ] Compare to custom kernel

**Tuesday:**
- [ ] Add cuSPARSE benchmark
- [ ] Run full comparison (PyTorch sparse vs cuSPARSE vs custom)

**Wednesday:**
- [ ] Test multiple matrix sizes (1K, 2K, 4K, 8K, 16K)
- [ ] Generate performance curves

**Thursday:**
- [ ] Test different sparsity patterns (topk=4,8,16,32,64)
- [ ] Analyze where kernel wins/loses

**Friday:**
- [ ] Update README with HONEST results
- [ ] Remove misleading comparisons
- [ ] Go/No-Go decision for open source

---

## 🎓 Expected Outcomes

### Best Case Scenario ✅
```
PyTorch sparse:  350 TFLOPS
cuSPARSE:        400 TFLOPS
CUTLASS 4.3:     414 TFLOPS
Our kernel:      610 TFLOPS  (+47% vs CUTLASS, +74% vs PyTorch)

Conclusion: Custom kernel JUSTIFIED
```

### Realistic Scenario ⚠️
```
PyTorch sparse:  500 TFLOPS
cuSPARSE:        450 TFLOPS
CUTLASS 4.3:     414 TFLOPS
Our kernel:      610 TFLOPS  (+47% vs CUTLASS, +22% vs PyTorch)

Conclusion: Custom kernel USEFUL but not amazing
```

### Worst Case Scenario ❌
```
PyTorch sparse:  650 TFLOPS  ← PyTorch wins!
cuSPARSE:        700 TFLOPS  ← cuSPARSE wins!
CUTLASS 4.3:     414 TFLOPS
Our kernel:      610 TFLOPS  

Conclusion: Custom kernel POINTLESS - use PyTorch/cuSPARSE
```

**Honest assessment:** We don't know which scenario is true until we run the benchmarks.

---

## 💡 Lessons for Future Work

1. **Test critical baselines FIRST**
   - Before claiming "faster", test against native libraries
   - PyTorch sparse and cuSPARSE are table stakes

2. **Don't compare different operations**
   - Sparse GEMM ≠ Attention
   - Makes us look uninformed

3. **Test multiple configurations**
   - One config = potential cherry-picking
   - Need robustness evidence

4. **Be honest about limitations**
   - "Research prototype" > "Production-ready" (when false)
   - Admit gaps, commit to fixing them

5. **Document methodology**
   - Clear what was tested
   - Clear what wasn't tested
   - Clear next steps

---

## 📧 Contact

**Author:** Brandon Dent, MD  
**Email:** b@thegoatnote.com  
**Purpose:** Honest engineering, not marketing

---

**Last Updated:** November 1, 2025  
**Status:** Benchmarks pending (THIS WEEK)  
**Next Update:** After PyTorch sparse / cuSPARSE comparison (Nov 5, 2025)

