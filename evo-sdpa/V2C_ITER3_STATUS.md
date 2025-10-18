# Child-V2c Iteration 3: Scalar Q@K^T Validation

**Date**: October 18, 2025  
**Status**: Ready for GPU Testing  
**Duration**: 0.5 hours (systematic debugging)

---

## 🎯 Iteration 3 Goal

**Validate infrastructure (streaming softmax, SMEM layout, cp.async) with scalar Q@K^T**

**Strategy**: Replace WMMA temporarily to isolate the Q@K^T transpose issue from other potential bugs

---

## 🔧 Changes Made

### 1. **Replaced WMMA Q@K^T with Scalar** (Lines 200-228)
```cuda
// Before (V2c-v2): WMMA computing Q @ K (not Q @ K^T!)
wmma::load_matrix_sync(a_frag, q_ptr, HEAD_DIM_PADDED);
wmma::load_matrix_sync(b_frag, k_ptr, HEAD_DIM_PADDED);  // Wrong: row-major K
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Q @ K (missing transpose!)

// After (V2c-v3): Scalar Q @ K^T (correct transpose)
for (int r = my_row_start; r < my_row_end; ++r) {
    for (int n = 0; n < kv_len; ++n) {
        float score = 0.0f;
        // Dot product: Q[r,:] @ K[n,:]^T (scalar, correct)
        for (int k = lane; k < HEAD_DIM; k += 32) {
            float q_val = __half2float(sQ[r * HEAD_DIM_PADDED + k]);
            float k_val = __half2float(sK[(read_stage * N + n) * HEAD_DIM_PADDED + k]);
            score += q_val * k_val;
        }
        score = warp_reduce_sum(score);
        if (lane == 0) {
            S_scores[r * N + n] = score * scale;  // Write scaled score
        }
    }
}
```

**Why**: Scalar Q[i,:] @ K[j,:] is mathematically equivalent to (Q @ K^T)[i,j], removing WMMA transpose complexity.

### 2. **Fixed Double-Scaling Bug** (Line 235)
```cuda
// Before (V2c-v2): Score scaled twice!
S_scores[r * N + n] = score * scale;  // Scaled once
...
float score = S_scores[r * N + n] * scale;  // Scaled AGAIN! ❌

// After (V2c-v3): Score scaled once only
S_scores[r * N + n] = score * scale;  // Scaled once
...
float score = S_scores[r * N + n];  // Already scaled ✅
```

**Impact**: This bug would cause attention weights to be incorrect by a factor of `scale`.

### 3. **Updated Header Comment**
- Clarified this is **Iteration 3** (validation phase)
- Documented goal: validate infrastructure before re-adding WMMA
- Added next step: proper WMMA with K^T handling

---

## 📋 What's Still Working

✅ **Streaming Softmax**: Unchanged (m, l, exp updates)  
✅ **SMEM Layout**: Unchanged (sQ, sK, sV, S_scores, O_accum)  
✅ **cp.async**: Unchanged (2-stage pipeline, 16B copies)  
✅ **Causal Masking**: Unchanged  
✅ **P@V**: Unchanged (scalar, correct)  
✅ **Epilogue**: Unchanged (normalize by l_smem)

---

## 🧪 Expected Test Results

### **If Infrastructure is Correct** ✅
```
Test 1: (1,8,512,64)  causal=False → PASS (max_diff < 0.001)
Test 2: (1,8,512,64)  causal=True  → PASS (max_diff < 0.001)
Test 3: (2,8,2048,64) causal=False → PASS (max_diff < 0.001)
Test 4: (2,8,2048,64) causal=True  → PASS (max_diff < 0.001)
Test 5: (2,8,2048,128) causal=False → PASS (max_diff < 0.001)

Expected Latency: ~2400-2500 μs (same as V2b scalar baseline)
```

### **If This Passes** → Confirms:
1. ✅ Streaming softmax is correct
2. ✅ SMEM layout is correct
3. ✅ cp.async is working (even if not optimal)
4. ✅ Causal masking is correct
5. ✅ P@V accumulation is correct
6. ✅ Epilogue normalization is correct

**Root Cause Isolated**: Q@K^T transpose issue only (fixable in Iteration 4)

---

## 🚀 Next Steps (Iteration 4: WMMA + K^T)

### **Two Approaches for WMMA Q@K^T**

#### **Option A: Transpose K During Load** (Cleaner)
```cuda
// Load K^T into SMEM (col-major view)
for (int idx = tid; idx < kv_len * HEAD_DIM; idx += blockDim.x) {
    int n = idx / HEAD_DIM;  // K row
    int c = idx % HEAD_DIM;  // K col
    // Write to sK as column-major: sK[c][n] instead of sK[n][c]
    sK[c * N_PADDED + n] = __ldg(&K_bh[(kv_start + n) * d + c]);
}

// Then WMMA:
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::load_matrix_sync(a_frag, &sQ[m0 * HEAD_DIM_PAD + k0], HEAD_DIM_PAD);
wmma::load_matrix_sync(b_frag, &sK[k0 * N_PAD + n0], N_PAD);  // Col-major K^T
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Correct Q @ K^T!
```

**Pros**: Mathematically clean, K^T is explicit  
**Cons**: Requires different SMEM layout (may need XOR swizzle tuning)

#### **Option B: Swap MMA Arguments** (Faster to Test)
```cuda
// Keep K row-major in SMEM
// Compute K @ Q^T instead, then use result as if it were (Q @ K^T)^T
// Since result is symmetric for our purposes (we apply softmax row-wise)
wmma::mma_sync(c_frag, b_frag, a_frag, c_frag);  // K @ Q^T
// Then transpose result when storing to S_scores
```

**Pros**: Minimal code change, quick test  
**Cons**: May require transpose on store, less clear semantically

---

## 📊 Resource Usage (Expected)

**Registers**: ~48-52/thread (scalar is lighter than WMMA fragments)  
**SMEM**: Same as V2b (79 KB for d=64, ~97 KB for d=128)  
**Occupancy**: 2 CTAs/SM (same as V2b)

---

## ✅ Validation Checklist

### **Before Testing** (Local)
- [x] Code compiles without errors
- [x] No lint errors
- [x] Double-scaling bug fixed
- [x] Scalar Q@K^T implemented correctly
- [x] Header comments updated

### **GPU Testing** (Remote)
- [ ] All 5 test cases pass (max_diff < 0.001)
- [ ] No CUDA errors (launch, memory, sync)
- [ ] Latency ~2400-2500 μs (comparable to V2b)
- [ ] Register/SMEM usage as expected

### **After Validation**
- [ ] Commit with clear message
- [ ] Proceed to Iteration 4 (WMMA + K^T)

---

## 🔬 Debug Plan (If Tests Fail)

### **If Correctness Fails**
1. Check `S_scores` after Q@K^T (are they reasonable?)
2. Check `m_smem`, `l_smem` after softmax (finite, positive?)
3. Check `O_accum` intermediate values (no NaNs?)
4. Validate causal masking (print masked indices)

### **If Launch Fails**
1. Check SMEM size calculation (should match V2b)
2. Verify grid/block dims (same as V2b)
3. Check warp responsibilities (no out-of-bounds)

---

## 📈 Progress Tracking

| Iteration | Goal | Status | Time | Result |
|-----------|------|--------|------|--------|
| **V2c-v1** | WMMA skeleton | ❌ Launch fail | 0.5h | SMEM issue |
| **V2c-v2** | Fix SMEM | ✅ Builds, ❌ Correctness | 1.0h | Transpose bug |
| **V2c-v3** | Scalar Q@K^T | 🔄 Ready for test | 0.5h | **Testing** |
| **V2c-v4** | WMMA + K^T | ⏳ Pending | - | - |

**Total Time**: 2 hours (systematic, TDD-driven)

---

## 🎯 Success Criteria

**Iteration 3 Success** = 5/5 tests pass + latency 2400-2500 μs  
**Iteration 4 Goal** = 5/5 tests pass + latency **800-1200 μs** (2-3× speedup from WMMA)

---

## 📝 Key Takeaways

1. **TDD Works**: Scalar validation isolates infrastructure bugs from WMMA issues
2. **Systematic Iteration**: Each iteration fixes one class of bugs (SMEM → Transpose → WMMA)
3. **Double-Scaling Bug**: Found proactively during code review (excellent!)
4. **Path Forward**: Clear (Iteration 4 with two concrete WMMA approaches)

---

**Status**: Ready for `python3 bench/test_v2c.py` on GPU instance 🚀


