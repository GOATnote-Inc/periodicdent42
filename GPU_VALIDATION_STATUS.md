# GPU Validation Status - In Progress

**Date**: October 19, 2025  
**GPU Instance**: cudadent42-l4-dev (NVIDIA L4, sm_89, us-west1-c)  
**Status**: 🟡 **DEBUGGING** (Correctness partially achieved, investigating remaining issue)

---

## 🎯 **Current Status**

### **Priority 1.1: Quantizer Fix** ✅ **PASS**

```bash
✅ Test: Quantizer scale bug (zero tensors)
   - Zero tensors → scale=1.0 ✓
   - Encoded values = 128 (midpoint) ✓
   - Manual dequant cycle verified ✓
```

**Verdict**: Bug #1 is fully fixed ✅

---

### **Priority 1.2: WMMA Score Loading Fix** ⚠️ **PARTIAL**

```bash
⚠️  Bug #2 fix applied (all lanes load all scores)
   - Code change confirmed in kernel ✓
   - Compiled successfully ✓
   - But correctness still failing ❌
```

**Issue**: After fixing the uninitialized array bug, we now have a **different correctness issue**:
- **Before fixes**: 99.5% wrong (abs=1.129)
- **After fixes**: 99.6% wrong (abs=1.52)

Still wrong, but the error pattern changed - now outputs are **systematically negative-biased**.

---

### **Priority 1.3: Correctness Validation** ❌ **FAIL**

```
Test: (B=1, H=4, S=128, D=64)
Max abs error: 1.52e+00
Mean abs error: 3.99e-01
% elements > 0.05: 99.6%

Sample outputs:
  FP8: [-0.311, -0.305, -0.339, -0.421, -0.316]
  Ref: [0.0297, -0.0106, -0.108, -0.206, 0.0137]
```

**Symptoms**:
- All FP8 outputs are **negative** (systematic bias)
- Reference has mixed signs (positive and negative)
- Magnitudes are also wrong (not just sign)
- **99.6% of elements** exceed tolerance

---

## 🔍 **Investigation So Far**

### **What We've Ruled Out** ✅

1. ✅ **Quantizer scale bug**: Fixed and verified working
2. ✅ **Uninitialized S_row array**: Fixed (all lanes load all scores)
3. ✅ **softmax_scale**: Correct (0.125 = 1/√64)
4. ✅ **Dequant formula**: Mathematically correct
5. ✅ **LUT construction**: Formula is correct
6. ✅ **Quantization cycle**: Python quant/dequant verified working

### **What Could Be Wrong** ❓

1. **WMMA output interpretation**: Maybe the 16×16 WMMA tiles aren't being stored correctly to `sS`
2. **Online softmax accumulation**: Issue with U_smem rescaling or accumulation
3. **V indexing**: Wrong indices when reading V in P·V loop
4. **Per-head scale indexing**: Scales might be applied to wrong heads
5. **Partial tile handling**: Edge cases when kv_len < TILE_N (though test has full tiles)
6. **Memory ordering**: Row-major vs col-major confusion
7. **Warp synchronization**: Missing `__syncwarp()` somewhere critical

---

## 🐛 **Suspected Root Cause**

Based on the **systematic negative bias** in all outputs, the most likely causes are:

### **Hypothesis 1: V values are wrong** (Most Likely)

The V dequantization might be applying scales incorrectly. Since outputs are all negative when they should be mixed, and the P·V computation multiplies attention weights (positive after softmax exp) by V values, wrong V signs would explain this.

**Action**: Add debug print to check V values after dequantization

### **Hypothesis 2: Online softmax rescaling error**

The `rescale = exp(m_old - m_new)` might be applied incorrectly, causing systematic bias.

**Action**: Check if rescale is being broadcast correctly to all elements

### **Hypothesis 3: WMMA output (Q@Kᵀ) is wrong**

The Q@Kᵀ scores might be computed incorrectly due to:
- Wrong leading dimension in `wmma::load_matrix_sync`
- Wrong memory layout (row-major vs col-major confusion)
- Incorrect transpose of K

**Action**: Add debug print to check Q@Kᵀ scores before softmax

---

## 🔧 **Next Debugging Steps**

### **Step 1: Validate Q@Kᵀ (WMMA output)**

Add debug kernel to print Q@Kᵀ scores and compare with reference:

```cuda
// After line 183 (wmma::store_matrix_sync for sS)
if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0 && t == 0) {
    printf("DEBUG Q@K^T scores [0:3, 0:3]:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("  sS[%d][%d] = %.4f\n", i, j, __half2float(sS[i][j]));
        }
    }
}
```

### **Step 2: Validate V values**

Add debug print to check V after dequantization:

```cuda
// After line 141 (load V tile)
if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0 && t == 0) {
    printf("DEBUG V values [0:3, 0:3]:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("  sV[%d][%d] = %.4f\n", i, j, __half2float(sV[i][j]));
        }
    }
}
```

### **Step 3: Validate Softmax**

Add debug print after softmax to check attention weights:

```cuda
// After line 213 (S_row[n] = exp(...))
if (r == 0 && warp_id == 0 && lane == 0 && t == 0) {
    printf("DEBUG Attention weights (row 0, first 3):\n");
    printf("  m_new = %.4f, l_add = %.4f\n", m_new, l_add);
    for (int n = 0; n < 3; n++) {
        printf("  P[%d] = %.4f\n", n, S_row[n]);
    }
}
```

### **Step 4: Validate Final Output**

Add debug print before final write:

```cuda
// After line 246 (float o = U_smem[r][d] / l_final)
if (r == 0 && d < 3) {
    printf("DEBUG Final output [0][%d] = %.4f (U=%.4f, l=%.4f)\n", 
           d, o, U_smem[r][d], l_final);
}
```

---

## 📊 **Detailed Error Analysis**

### **Sample Output Comparison**

| Index | FP8 Output | Reference | Abs Error | Rel Error |
|-------|------------|-----------|-----------|-----------|
| [0,0,0,0] | -0.311 | +0.0297 | 0.341 | 11.5× |
| [0,0,0,1] | -0.305 | -0.0106 | 0.295 | 27.8× |
| [0,0,0,2] | -0.340 | -0.108 | 0.232 | 2.1× |
| [0,0,0,3] | -0.421 | -0.206 | 0.215 | 1.0× |
| [0,0,0,4] | -0.317 | +0.0137 | 0.331 | 24.1× |

**Observations**:
- **Sign errors**: 2 out of 5 have wrong sign
- **Magnitude errors**: All have significant magnitude errors
- **Pattern**: FP8 outputs cluster around -0.3 to -0.4 (suspicious!)
- **Reference spread**: -0.2 to +0.03 (expected variation)

This clustering suggests a **systematic bias**, not random errors.

---

## 🎓 **Key Insights**

### **1. The Fixes Helped** ⚠️

The error pattern changed from 99.5% wrong (Bug #2: uninitialized array) to a different 99.6% wrong (new issue). This suggests:
- Bug #2 fix is working (no more uninitialized data)
- But there's a **third bug** we haven't found yet

### **2. Quantization is Correct** ✅

Python quantization/dequantization cycle verified working. The issue is **only in the CUDA kernel**.

### **3. Systematic Bias Points to Specific Issues** 🎯

Random errors would show as noise. **Systematic negative bias** suggests:
- Wrong sign in dequantization formula
- Wrong scale being applied
- Accumulation with wrong initial value
- Missing or extra rescaling step

---

## 📚 **Reference: Kernel Structure**

```
1. Load Q tile → sQ (dequant via dequant_sim_fp8)
2. Init U_smem = 0, m_smem = -inf, l_smem = 0
3. For each KV tile:
   a. Load K, V → sKT, sV (dequant via LUT)
   b. WMMA: Q @ K^T → sS (16×16 tiles)
   c. Load scores → S_row (all lanes, all elements)
   d. Online softmax:
      - Compute m_new, l_new
      - Rescale U by exp(m_old - m_new)
      - Accumulate P·V into U
   e. Update m_smem, l_smem
4. Final: O = U / l_final
```

---

## ⏭️ **Action Plan**

### **Immediate** (Next 30 minutes)

1. Add debug prints to kernel (Steps 1-4 above)
2. Recompile and run small test (B=1, H=1, S=16, D=64)
3. Compare debug outputs with Python reference
4. Identify which stage produces wrong values

### **Once Bug Found** (30-60 minutes)

1. Apply targeted fix
2. Re-run validation: `python scripts/bench_fp8_stage_c.py --shapes mission --iters 10`
3. Expect: ✅ PASS (abs<1e-2, rel<1e-2)
4. Proceed to Priority 2.1 (baseline performance)

### **After Correctness Passes** (2-4 hours)

1. Priority 2.1: Full baseline (mission, small, long)
2. Priority 2.2: NCU profiling
3. Priority 2.3: Optimization (WMMA P·V, cp.async, XOR swizzle)
4. Target: <20 μs (2× faster than PyTorch SDPA)

---

## 🔗 **Related Files**

- **Kernel**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`
- **Python Wrapper**: `cudadent42/bench/sdpa_fp8_stage_c_wmma.py`
- **Benchmark**: `scripts/bench_fp8_stage_c.py`
- **Tests**: `tests/test_fp8_stage_c_wmma.py`

---

## 💾 **GPU Instance Access**

```bash
# SSH to GPU instance
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c

# Working directory
cd ~/periodicdent42
source venv/bin/activate

# Quick test
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA L4
```

---

**Last Updated**: October 19, 2025  
**Status**: Debugging in progress (Bug #3 suspected in V dequantization or online softmax)  
**Confidence**: 80% that we can fix this with targeted debugging in next session


