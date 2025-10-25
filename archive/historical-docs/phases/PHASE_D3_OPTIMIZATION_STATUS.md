# Phase D.3 Optimization Status - Hour 9

**Time**: 9/14 hours used (extended budget)  
**Status**: Vectorization encountering alignment issues

---

## Progress Summary

### ✅ Completed: Correctness Baseline
- **Cycle 2**: Expert patch → 1596.75 μs, 100% correct
- Root causes fixed (row coverage, SMEM accumulator, broadcast)
- Validated with rigorous testing

### 🔧 In Progress: Cycle 2a Vectorization
- **Goal**: 400-500 μs (3-4× speedup)
- **Status**: Build successful, runtime error

**Error**: `RuntimeError: CUDA error: misaligned address`

**Root Cause**: int4 vectorized loads assume 16-byte alignment,  
but PyTorch tensors may not guarantee this at element level.

---

## Technical Analysis

**What Works**:
- ✅ Kernel compiles successfully
- ✅ 63 registers/thread (good occupancy potential)
- ✅ 20.5 KB SMEM (well under 48 KB limit)
- ✅ cp.async template syntax correct

**What's Broken**:
- ❌ int4 loads from global memory (`*reinterpret_cast<int4*>(&qrow[d0])`)
- ❌ Assumes 16-byte alignment at arbitrary D offsets

**Why**:
PyTorch tensors are allocated with stride alignment, but:
- Tensor start might be 16-byte aligned
- But `qrow + d0` where `d0 = lane * 16` may not be
- Especially when `lane > 0`, `d0` may not hit 16-byte boundary

---

## Fix Strategy

### Option A: Simpler Coalesced Loads (Recommended)
**Approach**: Scalar loads with good coalescing + cp.async
```cuda
// Each thread loads consecutive bytes (naturally coalesced)
for (int idx = tid; idx < rows_in_tile * D; idx += blockDim.x) {
    int r = idx / D;
    int d = idx % D;
    sQ[r][d] = Qbh[(q_start + r) * D + d];
}
```

**Benefits**:
- No alignment requirements
- Hardware naturally coalesces 32 consecutive loads
- cp.async still provides pipelining benefit
- Simple, robust, proven

**Expected**: 2-3× speedup (still good!) → 530-800 μs

### Option B: Careful Vectorization with Alignment Checks
**Approach**: Only vectorize when provably aligned
```cuda
if ((reinterpret_cast<uintptr_t>(qrow) % 16 == 0) && (d0 % 16 == 0)) {
    // Safe to use int4
} else {
    // Fall back to scalar
}
```

**Drawbacks**:
- Complex branch logic
- Warp divergence
- May not actually be faster

### Option C: Force Alignment in Python
**Approach**: Ensure tensors are 16-byte aligned before kernel launch
```python
Q_aligned = torch.empty(..., device='cuda').align_to(16)
```

**Issues**:
- PyTorch doesn't expose align_to() API
- Would need custom allocator
- Not portable

---

## Recommendation: Option A (Simpler Coalesced Loads)

**Why**:
1. **Robustness**: No alignment issues
2. **Performance**: Hardware coalescing is excellent on Ada
3. **cp.async**: Still get pipelining benefit (main win!)
4. **Simplicity**: Less complex, easier to debug
5. **Proven**: This is what production kernels do

**Realistic Target** with Option A:
- Baseline: 1596 μs
- With coalesced loads + cp.async: 600-800 μs (2-2.7× speedup)
- Still significant progress!

**Then**:
- Cycle 3: cp.async double-buffering → 300-400 μs
- **Total**: 4-5× speedup (still excellent!)

---

## Revised Timeline

```
Spent:    9 hours (correctness + vectorization attempt)
Remaining: 5 hours (of 14-hour extended budget)

Plan:
  Hour 10: Implement Option A (coalesced loads) [1 hour]
  Hour 11: Test + Debug [1 hour]
  Hour 12-13: cp.async double-buffering [2 hours]
  Hour 14: NCU validation + documentation [1 hour]

Expected Final Performance:
  Baseline: 1596 μs
  After Cycle 2a (coalesced): 600-800 μs
  After Cycle 3 (cp.async 2-stage): 300-400 μs
  
  Final: 4-5× speedup vs baseline
```

---

## Key Lesson

**Premature vectorization is the root of all evil!**

- int4 loads: Fast when aligned, crash when not
- Coalesced scalar loads: Always work, nearly as fast
- **Simplicity wins** on modern GPUs with good hardware coalescing

**Next Action**: Implement Option A (coalesced scalar loads + cp.async)

---

**Status**: Ready to fix and continue optimization! 💪

