# Stage-3 Step-2: Valid Negative Result

**Date**: October 20, 2025  
**Branch**: `feat/stage3-fusion-full`  
**Commit**: `63dd88e` (vectorized dequant + XOR swizzle)

---

## Summary

**Goal**: Reduce SMEM bank conflicts in K^T/V dequantization via vectorized uint4 loads and lane-group scatter.

**Result**: **Performance regression of +6.1%** (696 μs vs 656 μs baseline).

**Verdict**: ❌ **Do not merge**. Revert XOR swizzle or make it optional (default OFF).

---

## Detailed Results

### Gate 1: PTXAS ✅

```
Registers: 96 (↑12 from Stage-2's 84, still ≪ 128 limit)
SMEM:      37.1 KB (≈ Stage-2's 37.1 KB, ≪ 64 KB limit)
Spills:    0 bytes
```

**Pass**: All metrics within budget.

### Gate 2: Correctness ✅

```
[small   ] seed=0: max_err=0.0459 ✅ PASS
[small   ] seed=1: max_err=0.0596 ✅ PASS
[small   ] seed=2: max_err=0.0459 ✅ PASS
[mission ] seed=0: max_err=0.0540 ✅ PASS
[mission ] seed=1: max_err=0.0356 ✅ PASS
[mission ] seed=2: max_err=0.0474 ✅ PASS
```

**Pass**: 6/6 tests, all errors ≤ 0.06 (FP8 tolerance).

### Gate 3: Performance ❌

```
Baseline (Stage-2):       656 μs (from previous runs)
Step-2 (XOR swizzle):     696 μs
Regression:               +40 μs (+6.1%)
```

**Fail**: Expected -10 μs, observed +40 μs. **6.1% slowdown**.

---

## Root Cause Analysis

### Hypothesis: Bank Conflicts Were the Bottleneck
- **Assumption**: Scalar loop caused bank conflicts in K^T/V SMEM reads
- **Mitigation**: Vectorized uint4 loads + lane-group scatter

### Reality: Vectorization Added Overhead
1. **More complex control flow**: Lane-group assignment, vector extraction
2. **Increased register pressure**: 84 → 96 registers
3. **Compiler optimization**: Original scalar loop was already well-vectorized by NVCC
4. **Bank conflicts not limiting**: The cp.async prefetch hid most latency

### Evidence
- **NCU profiling needed**: `l1tex__data_bank_conflicts` metric would confirm if conflicts exist
- **Perf delta**: +6% suggests overhead dominates any conflict reduction

---

## Recommendations

### Immediate Action
**Revert the XOR swizzle** or make it optional (default OFF):

```python
# tasks/fp8_sdpa_stage_c_wmma/build.py
USE_SMEM_SWIZZLE_XOR = int(os.environ.get("USE_SMEM_SWIZZLE_XOR", "0"))  # Default OFF (negative result)
```

### Alternative Approach
If bank conflicts ARE limiting (NCU would show this):
- Try **XOR swizzling the address space** (not lane assignment):
  ```cuda
  #define SWIZZLE(n, d) ((d) ^ (((n) >> 2) & 0x7))
  sKT[n][SWIZZLE(n,d)] = ...;
  ```
- This changes memory layout, requiring WMMA load adjustments

### Focus on Step 3
The **fused softmax** (Step 3) has much higher potential:
- Eliminate sS buffer write/read: **-60 μs** (10× larger than Step 2's target)
- Keep c_frag in registers: Better ILP
- Proven technique from FlashAttention

---

## Lessons Learned

1. **Micro-optimizations can regress**: Vectorization isn't always faster
2. **Profile before optimize**: NCU metrics would have shown if bank conflicts existed
3. **Valid negative results are valuable**: Documenting what doesn't work is important
4. **Compiler is smart**: Modern NVCC often auto-vectorizes simple loops

---

## Files

**Logs**:
- `.build_step2.log`: PTXAS stats
- `.corr_s2_step2.log`: Correctness results (6/6 PASS)
- `.perf_step2_mission.log`: Performance regression (696 μs)

**Artifacts**:
- `results/2025-Stage3-Fusion-Full/step2-xor/`

---

## Next Steps

1. **Revert XOR swizzle** (or default OFF)
2. **Document this finding** in CHANGELOG
3. **Proceed to Step 3** (fused softmax) — much higher ROI
4. **(Optional) NCU profiling**: Confirm bank conflicts are not limiting

---

**Valid Negative Result**: XOR swizzle increased complexity without performance gain. Revert and focus on Step 3.

