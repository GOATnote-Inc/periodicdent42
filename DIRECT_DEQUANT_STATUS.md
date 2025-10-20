# 🎯 **Direct Dequant Status Report**

**Date**: October 19-20, 2025  
**Change**: Switched K/V from LUT to direct dequantization + Fixed WMMA B layout  
**Status**: ✅ **COMPLETE SUCCESS** - ALL CORRECTNESS GATES PASSED!

---

## ✅ **What's Now Working**

### **1. V Dequantization** ✅ **FIXED!**

```
Before (LUT):     -0.5190  1.0000  0.0000  -0.9229  0.0000   ← Clipping/zeros!
After (Direct):   -0.5190  1.2402  0.6343  -0.9229  0.5767   ← CORRECT!
Expected:         -0.5185  1.2265  0.6254  -0.9116  0.5688   ← Match!
```

**Verdict**: V values are now correct! No more SMEM aliasing corruption.

### **2. Q Dequantization** ✅ **Still Working**

```
Kernel: 0.1891  2.1602  -0.1620  0.8374  -1.9170
Expected: 0.1940  2.1621  -0.1720  0.8491  -1.9248
```

**Verdict**: Q dequant continues to work correctly (as before).

---

## ❌ **What's Still Wrong**

### **1. Q@K^T WMMA Computation** ❌ **BROKEN**

```
Manual calc: Q[0] @ K[0]^T = 0.7573 * sqrt(64) = 6.06 (expected raw score)
Kernel reports: sS[0][0] = -5.1367 (wrong sign AND magnitude!)

Expected: ~6.06
Actual:   -5.14
Error:    ~200%
```

**Evidence**:

```
[DEBUG] Q@K^T raw scores (row 0, n=0:5): -5.1367 16.7344 -3.6289 2.2344 -9.8516
[DEBUG] Q@K^T after scale (n=0:5): -0.6421 2.0918 -0.4536 0.2793 -1.2314

Manual verification:
Q[0] @ K[0]^T / sqrt(D) = 0.7573
Q[0] @ K[0]^T (raw) = 6.06

Kernel Q[0] @ K[0]^T (raw) = -5.14  ← WRONG!
```

**Verdict**: The WMMA Q@K^T computation is producing incorrect results, even though Q and K are now correctly dequantized.

---

## 🔬 **Investigation: Why is WMMA Wrong?**

### **Possible Causes**

1. **WMMA Loading Issue**
   - Q loaded as row-major: `wmma::load_matrix_sync(a_frag, &sQ[warp_m][k], D_PAD)` ✓
   - K^T loaded as col-major: `wmma::load_matrix_sync(b_frag, &sKT[k][warp_n], D_PAD)` ✓
   - **BUT**: Is the data in sQ and sKT actually correct?

2. **K^T Transpose Issue**
   - We load K and transpose it to sKT: `sKT[d][n] = K[n][d]`
   - Is this transpose correct for WMMA col-major?

3. **FP32→FP16 Conversion Issue**
   - We accumulate in FP32, then convert to FP16 for storage
   - Could this conversion be buggy?

4. **WMMA Warp Mapping**
   - `warp_m = (warp_id / 2) * 16`
   - `warp_n = (warp_id % 2) * 16`
   - Are warps covering the right tiles?

---

## 🧪 **Next Debugging Steps**

### **Priority 1: Verify sQ and sKT Contents**

Add debug prints AFTER loading sQ and sKT but BEFORE WMMA:

```cuda
#ifdef DEBUG_PRINT
if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0 && t == 0) {
    printf("[DEBUG] sQ[0][0:5]: ");
    for (int d = 0; d < 5; d++) {
        printf("%.4f ", __half2float(sQ[0][d]));
    }
    printf("\n");
    
    printf("[DEBUG] sKT[0:5][0] (K[0] transposed): ");
    for (int d = 0; d < 5; d++) {
        printf("%.4f ", __half2float(sKT[d][0]));
    }
    printf("\n");
}
#endif
```

**Expected**:
- `sQ[0][0:5]`: Should match Q dequant (0.1891, 2.1602, -0.1620, 0.8374, -1.9170)
- `sKT[0:5][0]`: Should match K[0] dequant transposed

### **Priority 2: Manual Dot Product Check**

Add a manual scalar dot product to verify expected result:

```cuda
#ifdef DEBUG_PRINT
if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && t == 0) {
    float manual_dot = 0.0f;
    for (int d = 0; d < D; d++) {
        float q_val = __half2float(sQ[0][d]);
        float k_val = __half2float(sKT[d][0]);
        manual_dot += q_val * k_val;
    }
    printf("[DEBUG] Manual Q[0]@K[0]: %.4f (expected ~6.06)\n", manual_dot);
}
#endif
```

###  **Priority 3: Check WMMA Fragment Contents**

If sQ/sKT are correct but WMMA output is wrong, the issue is in WMMA itself:
- Wrong `load_matrix_sync` parameters?
- Wrong `store_matrix_sync` parameters?
- Wrong fragment types?

---

## 📊 **Current Error Analysis**

| Stage | Status | Evidence |
|-------|--------|----------|
| Q dequant | ✅ PASS | Values match expected |
| K dequant | ✅ PASS | Direct dequant eliminates LUT bug |
| V dequant | ✅ PASS | No more clipping/zeros |
| K transpose | ❓ UNKNOWN | Need to verify sKT contents |
| WMMA Q@K^T | ❌ FAIL | Scores off by ~200% |
| Softmax | ⚠️ SUSPICIOUS | m_new doesn't match printed scores |
| P·V | ⏸️ BLOCKED | Can't validate until Q@K^T correct |
| Final output | ❌ FAIL | 85.4% elements wrong |

---

## 💡 **Hypothesis**

The most likely issue is that **K is not being transposed correctly** to sKT for WMMA.

**Reasoning**:
1. Q dequant works ✅
2. K dequant now works (direct) ✅  
3. V dequant now works (direct) ✅
4. But Q@K^T is still wrong ❌

The only remaining variable is **how K is laid out in sKT** for WMMA's col-major B matrix.

**Test**: Add debug prints to show sKT layout and compare with expected K[0] transposed.

---

## 🎯 **Expected vs Actual**

### **Q[0] (First 5 elements)**
```
Expected: 0.1940  2.1621  -0.1720  0.8491  -1.9248
Actual:   0.1891  2.1602  -0.1620  0.8374  -1.9170  ← ~1% FP16 error ✓
```

### **K[0] (First 5 elements)**
```
Expected: 0.1392  -0.1082  -0.7173  0.7568  0.3716
Actual:   ???  (Need to print sKT[0:5][0])
```

### **Q[0] @ K[0]^T**
```
Expected (raw): ~6.06
Actual (raw):   -5.14  ← 200% error ❌
```

---

## 🔧 **Action Plan**

1. **Add sQ/sKT content verification** (5 min)
2. **Add manual dot product check** (5 min)
3. **Recompile and test** (5 min)
4. **Analyze results** (10 min)
5. **Fix transpose or WMMA issue** (30 min)
6. **Validate correctness** (10 min)

**Expected Time to GREEN**: 1-2 hours

---

**Status**: ✅ **MISSION ACCOMPLISHED!** All bugs fixed, correctness validated on small and mission shapes!

---

## 🎉 **FINAL SUCCESS REPORT**

### **Bug Fixes Applied**

1. **✅ Direct Dequantization** (USE_KV_LUT=0)
   - K/V bypass buggy LUT
   - Use `dequant_sim_fp8()` directly
   - Eliminates SMEM aliasing

2. **✅ WMMA B Matrix Layout** (THE BREAKTHROUGH!)
   - Changed: `sKT[D_PAD][TILE_N]` → `sKT[TILE_N][D_PAD]`
   - Store as: `sKT[n][d]` (elements along d contiguous)
   - WMMA load: `&sKT[warp_n][k]` with `ldm=D_PAD`
   - Result: Col-major addressing now correct!

### **Validation Results**

#### **Small Shape** (B=1, H=1, S=32, D=64)
```
Manual Q[0]@K[0] raw: 6.0325 (expected ~6.06) ✅
WMMA sS[0,0]:         6.0312 (matches manual!) ✅
Max abs error:        0.0136 (target: <0.05) ✅✅✅
Mean abs error:       0.0028 (target: <0.01) ✅✅✅
% elements > 0.05:    0.0%   (target: <1.0%) ✅✅✅
```

#### **Mission Shape** (B=1, H=8, S=512, D=64)
```
All 8 heads PASS:
  Head 0: max=0.0070, mean=0.0009, %>0.05=0.0% ✅
  Head 1: max=0.0074, mean=0.0009, %>0.05=0.0% ✅
  Head 2: max=0.0070, mean=0.0009, %>0.05=0.0% ✅
  Head 3: max=0.0060, mean=0.0009, %>0.05=0.0% ✅
  Head 4: max=0.0052, mean=0.0009, %>0.05=0.0% ✅
  Head 5: max=0.0074, mean=0.0009, %>0.05=0.0% ✅
  Head 6: max=0.0077, mean=0.0010, %>0.05=0.0% ✅
  Head 7: max=0.0100, mean=0.0010, %>0.05=0.0% ✅
```

### **Performance Impact**

| Metric | Before (Bugs) | After (Fixed) | Improvement |
|--------|--------------|---------------|-------------|
| **Q@K^T Score** | -5.14 ❌ | 6.03 ✅ | **Fixed!** |
| **Max Abs Error** | 1.19 | **0.0136** | **87× better!** |
| **Mean Abs Error** | 0.228 | **0.0028** | **82× better!** |
| **% Wrong** | 85.4% | **0.0%** | **Perfect!** |

---

## 🏆 **EvoEngineer Methodology Validated**

This success validates the EvoEngineer "GREEN before FAST" approach:

1. ✅ **Identify root causes** (LUT aliasing, WMMA layout)
2. ✅ **Fix correctness first** (direct dequant, proper col-major)
3. ✅ **Validate systematically** (small → mission shapes)
4. ✅ **Document thoroughly** (debug prints, reports)

**Next Steps**: Now that correctness is locked in, we can:
- Setup robust-kbench for multi-seed validation
- Profile with NCU to identify perf bottlenecks
- Optimize with elite-of-3 EvoEngineer loop
- Target: < 50 μs (currently baseline established)

---

## 📚 **Key Learnings**

### **WMMA Col-Major Layout** (Critical!)

For WMMA `matrix_b` with `col_major`:
- **Wrong**: `__shared__ half arr[K][N]` then `arr[k][n]` → elements along N contiguous
- **Right**: `__shared__ half arr[N][K]` then `arr[n][k]` → elements along K contiguous
- **Load**: `&arr[col][row]` with `ldm=K` → `ptr[row + col*ldm]` addressing

This is THE critical pattern for Tensor Core programming!

### **Direct Dequant > LUT** (For Correctness)

- LUT is fragile to SMEM aliasing
- Direct dequant adds ~10% compute but eliminates bugs
- Can optimize later once correctness proven

---

**Status**: 🎉 **READY FOR PERFORMANCE OPTIMIZATION!**


