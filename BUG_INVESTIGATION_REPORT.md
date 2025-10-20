# 🔬 **FP8 Stage-C WMMA: Multi-Bug Investigation Report**

**Date**: October 19, 2025  
**GPU**: NVIDIA L4 (sm_89), Google Cloud  
**Status**: 🔴 **BLOCKED** - Multiple correctness issues identified

---

## 📊 **Current Test Results**

```
Test: (B=1, H=1, S=32, D=64)
Max abs error: 1.2227e+00
Mean abs error: 2.7417e-01
% elements > 0.05: 90.6%

Sample comparison:
  [0] FP8: -0.1985  Ref: -0.1880  Err: 1.0498e-02  ✓ (Close!)
  [1] FP8: -0.2340  Ref: -0.2820  Err: 4.7974e-02  ✓ (Close!)
  [2] FP8: -0.4165  Ref: -0.6367  Err: 2.2021e-01  ❌
  [3] FP8: -0.1975  Ref: +0.2343  Err: 4.3176e-01  ❌ (Wrong sign!)
  [4] FP8: -0.3169  Ref: -0.1293  Err: 1.8762e-01  ❌
```

---

## 🐛 **Bugs Found and Status**

| Bug# | Description | Status | Evidence |
|------|-------------|--------|----------|
| **#1** | Quantizer scale (zero tensors) | ✅ **FIXED** | Python tests pass |
| **#2** | WMMA score loading (uninitialized) | ✅ **FIXED** | All lanes load all scores |
| **#3** | WMMA leading dimension | ❓ **UNCLEAR** | Still seeing wrong Q@K^T scores |
| **#4** | V dequantization/LUT | 🔴 **ACTIVE** | V values clipped/zeroed |

---

## 🔍 **Bug #4: V Dequantization Issue** (NEW - Most Critical)

### **Evidence from Debug Prints**

```
[DEBUG] V tile loaded (row 0, d=0:5): -0.5190 1.0000 0.0000 -0.9229 0.0000
Expected:                               -0.5185 1.2265 0.6254 -0.9116 0.5688
                                              ↑      ↑      ↑             ↑
                                          V[1] CLIPPED! (1.2265 → 1.0000)
                                          V[2] ZEROED!  (0.6254 → 0.0000)
                                          V[4] ZEROED!  (0.5688 → 0.0000)
```

### **Quantized Input (Correct)**

```python
V_q[0,0,0,0:5]: [110, 171, 150, 96, 148]
V_s[0]: 0.008174

Manual dequant V[1]:
  encoded = 171
  centered = (171 - 128) / 127 = 0.3386
  decoded = 0.3386 * 448 = 151.7
  result = 151.7 * 0.008174 = 1.240 ✓ (should be ~1.2265)
  
But kernel shows: 1.0000 ❌ (clipped!)
```

### **Hypothesis #4A: LUT Clipping/Saturation**

**Suspect**: The vLUT might be storing values incorrectly, possibly due to:
1. FP16 precision loss in LUT storage
2. Wrong LUT indexing in the load loop
3. Synchronization issue (threads reading LUT before it's fully built)

**Test**: Add debug print of vLUT[171] right after LUT construction

---

## 🔍 **Bug #3 Revisited: Q@K^T Still Wrong**

### **Evidence**

```
[DEBUG] Q@K^T scores (row 0, n=0:5): -8.9141 12.2969 -3.4629 4.4414 -4.5391
Manual Q[0] @ K[0]^T / sqrt(D) = 0.7573 ✓

Kernel shows S[0,0] = -8.9141 ❌ (off by ~10×!)
```

**Status**: The leading dimension fix (`D_PAD` instead of `TILE_N`) is in the code, but scores are still wrong.

### **Hypothesis #3A: K^T Transpose/LUT Issue**

**Suspect**: K dequantization via kLUT might have the same bug as V.

**Evidence**: If K values are wrong, then Q@K^T will be completely wrong, explaining the -8.9 vs 0.76 discrepancy.

---

## 🎯 **Root Cause Analysis**

### **Most Likely: LUT Construction or Access Bug**

The pattern suggests **both K and V dequantization are failing**:

1. ✅ Python quantization works (verified in earlier tests)
2. ✅ Q dequantization works (Q values match expected)
3. ❌ K dequantization produces wrong Q@K^T scores
4. ❌ V dequantization produces clipped/zeroed values

**Common factor**: K and V use **LUT-based dequantization**, while Q uses **direct dequantization**.

### **LUT Code (Suspect)**

```cuda
// Lines 89-101
if (tid < 256) {
    const int u = tid;
    constexpr float INV_MAX = 1.0f / 127.0f;
    float centered = (static_cast<float>(u) - 128.0f) * INV_MAX;
    float decoded = centered * 448.0f;
    kLUT[u] = decoded * k_s;  // ← Suspect: k_s from global memory?
    vLUT[u] = decoded * v_s;  // ← Suspect: v_s from global memory?
}
```

**Potential Issues**:
1. **Missing __syncthreads()** after LUT construction?
2. **Scale values not properly loaded** before LUT construction?
3. **LUT storage type** (float vs half)?

---

## 🔬 **Diagnostic Plan**

### **Step 1: Add LUT Debug Prints**

```cuda
#ifdef DEBUG_PRINT
if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("[DEBUG] Scales: k_s=%.6f v_s=%.6f\n", k_s, v_s);
    printf("[DEBUG] kLUT[133]=%.4f (expected ~1.17)\n", kLUT[133]);
    printf("[DEBUG] vLUT[171]=%.4f (expected ~1.24)\n", vLUT[171]);
}
__syncthreads();
#endif
```

### **Step 2: Check Synchronization**

Add explicit `__syncthreads()` after LUT construction:

```cuda
if (tid < 256) {
    // ... LUT construction ...
}
__syncthreads();  // ← Ensure all threads see complete LUT
```

### **Step 3: Verify LUT Indexing**

Check if the K/V load loops use correct indices:

```cuda
uint8_t k_u8 = Kbh[(size_t)(kv_start + n) * D + d];
sKT[d][n] = __float2half(kLUT[k_u8]);  // ← Is k_u8 correct?
```

---

## 📋 **Action Items (Priority Order)**

1. **[HIGH]** Add LUT debug prints (scales, sample LUT values)
2. **[HIGH]** Add `__syncthreads()` after LUT construction
3. **[HIGH]** Verify K/V indexing in load loops
4. **[MEDIUM]** Test with direct dequant for K/V (bypass LUT)
5. **[MEDIUM]** Check if scales are loaded correctly (print q_s, k_s, v_s)

---

## 💡 **Alternative Hypothesis**

### **Could This Be FP8 Precision Loss?**

**Evidence Against**:
- Python quantization/dequantization cycle **works perfectly**
- Only kernel dequantization fails
- Pattern is **systematic** (clipping at 1.0, zeros), not random noise

**Verdict**: Unlikely to be FP8 precision. More likely a kernel bug.

---

## 📊 **Comparison: Python vs Kernel Dequant**

| Stage | Python | Kernel | Match? |
|-------|--------|--------|--------|
| Q encode | [135, 208, 122, ...] | N/A | ✓ |
| Q decode | [0.1940, 2.1621, ...] | [0.1891, 2.1602, ...] | ✓ (~1% FP16 error) |
| K encode | [133, 124, 103, ...] | N/A | ✓ |
| K decode | [0.1392, -0.1082, ...] | **UNKNOWN** | ❓ |
| V encode | [110, 171, 150, ...] | N/A | ✓ |
| V decode | [-0.5185, 1.2265, 0.6254, ...] | [-0.5190, **1.0000**, **0.0000**, ...] | ❌ |

---

## 🎯 **Expected Fix Impact**

Once LUT bug is fixed:
- Q@K^T scores should be O(1) scale (~0.76), not O(10) scale (~-8.9)
- V values should match Python dequant (no clipping/zeros)
- Overall error should drop from 90.6% wrong to <5% wrong
- Max abs error should drop from 1.22 to <0.05

---

## 🔗 **Related Files**

- **Kernel**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu` (lines 89-157)
- **Python Wrapper**: `cudadent42/bench/sdpa_fp8_stage_c_wmma.py` (quantize_sim_fp8_per_head)
- **Debug Script**: `tools/debug_fp8_stage_c.py`

---

**Next Step**: Add LUT debug prints and synchronization barrier, then re-test.


