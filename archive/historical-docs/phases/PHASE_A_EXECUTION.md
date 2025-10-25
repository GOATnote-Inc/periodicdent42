# **Phase A Execution: Correctness Fix**

**Goal**: Restore 100% correctness on PyTorch 2.5.0 (currently 19%)  
**Time**: 4 hours  
**Status**: 🟡 **In Progress**

---

## **Problem Statement**

**Current**: Phase 4 kernel shows only **19% correctness** on PyTorch 2.5.0 (was 100% on 2.1.0)

**Impact**: Blocks all Tensor Core development (can't validate TC correctness if baseline is broken)

**Hypothesis**: PyTorch SDPA reference behavior changed between 2.1.0 → 2.5.0

---

## **Task Breakdown**

### **Task A.1: Isolate PyTorch Version** ⏱️ 1 hour | Status: 🔵 **Ready**

**Goal**: Confirm that PyTorch version is the issue

**Steps**:
```bash
# On GPU instance (run scripts/phase_a_gpu_setup.sh)
cd ~/periodicdent42
source ~/venv/bin/activate

# Test with PyTorch 2.1.0
pip uninstall torch -y
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
rm -rf ~/.cache/torch_extensions

# Run standalone eval
PYTHONPATH=. python scripts/standalone_phase4_eval.py | tee evidence/phase_a_pytorch210_test.log
```

**Expected Result**: **100% correctness** on PyTorch 2.1.0 (confirms version hypothesis)

**If 100% correct**: Proceed to Task A.2 (numerical stability)  
**If still broken**: Deeper debugging required (kernel bug, not PyTorch version)

---

### **Task A.2: Numerical Stability** ⏱️ 2 hours | Status: 🟡 **Created**

**Goal**: Add guards to prevent overflow/underflow/NaN propagation

**Changes Made**: Created `fa_phase3_stable.cu` with:

1. **Safe Exponential**:
   ```cuda
   #define EXP_CLAMP_MAX 20.0f
   #define EXP_CLAMP_MIN -20.0f
   
   __device__ __forceinline__ float safe_exp(float x) {
       x = fminf(fmaxf(x, EXP_CLAMP_MIN), EXP_CLAMP_MAX);
       return expf(x);
   }
   ```

2. **NaN Guards**:
   ```cuda
   __device__ __forceinline__ bool is_finite(float x) {
       return isfinite(x);
   }
   
   // Before updates
   if (!is_finite(exp_diff)) exp_diff = 0.0f;
   if (!is_finite(l_new) || l_new < EPSILON) l_new = l_prev;
   if (!is_finite(o_val)) o_val = 0.0f;
   ```

3. **Division by Zero Protection**:
   ```cuda
   #define EPSILON 1e-8f
   float l_inv = 1.0f / (l_prev[local_row] + EPSILON);
   ```

**Next Steps**:
1. Build `fa_phase3_stable.cu`
2. Test with PyTorch 2.5.0
3. Compare correctness vs original `fa_phase3_wmma.cu`

**Expected**: Progressively increase correctness rate (19% → 50% → 80% → 100%)

---

### **Task A.3: Dual-Reference Validation** ⏱️ 1 hour | Status: ✅ **Created**

**Goal**: Identify which SDPA backend (Flash vs Math) matches Phase 4 behavior

**Script Created**: `scripts/phase_a_validate_dual_backend.py`

**Features**:
- Tests both Flash and Math SDPA backends
- Compares Phase 4 output against both
- Identifies best reference (smallest diff)
- Saves results to `evidence/phase_a_dual_backend.txt`

**Run**:
```bash
# On GPU instance (after Task A.1 and A.2)
python scripts/phase_a_validate_dual_backend.py | tee evidence/phase_a_dual_backend.log
```

**Expected Output**:
```
Flash vs Math backend diff: 0.000123
Phase 4 vs Flash: 0.001234 ✅
Phase 4 vs Math:  0.003456 ❌

Best Reference: flash
```

**Exit Codes**:
- 0 = PASSED (best_diff < 2e-3)
- 1 = FAILED (correctness issues)
- 2 = ERROR (kernel not built)

---

## **Files Created**

```
scripts/phase_a_gpu_setup.sh              Interactive setup script
scripts/phase_a_validate_dual_backend.py  Dual-reference validator
cudadent42/bench/kernels/fa_phase3_stable.cu  Numerically stable kernel
PHASE_A_EXECUTION.md                      This file
```

---

## **Execution Plan**

### **On GPU Instance**:

```bash
# Step 1: Run setup script (interactive)
cd ~/periodicdent42
source ~/venv/bin/activate
bash scripts/phase_a_gpu_setup.sh

# The script will guide you through:
# 1. Test with PyTorch 2.1.0 (should be 100% correct)
# 2. Upgrade to PyTorch 2.5.0
# 3. Run dual-reference validation
```

### **Expected Timeline**:

```
Hour 1: Task A.1 (PyTorch 2.1.0 test) → 100% correctness confirms hypothesis
Hour 2-3: Task A.2 (Build and test stable kernel) → Increase correctness
Hour 4: Task A.3 (Dual-reference validation) → Identify best SDPA backend
```

### **Success Criteria**:

```
✅ 100% correctness on PyTorch 2.1.0 (validates hypothesis)
✅ 100% correctness on PyTorch 2.5.0 (stable kernel works)
✅ Identified best SDPA backend (Flash or Math)
✅ Evidence saved (logs, dual_backend.txt)
```

---

## **If Phase A Fails**

### **Fallback 1**: Accept PyTorch 2.1.0 Only (Low Risk)

**Action**: Document that Phase 4 requires PyTorch 2.1.0
**Impact**: Phase B and C can proceed with PyTorch 2.1.0
**Trade-off**: Limited to older PyTorch version

### **Fallback 2**: Use SDPA Math Backend (Medium Risk)

**Action**: If Flash backend differs, use Math backend as reference
**Impact**: May have slightly different numerical behavior
**Trade-off**: Acceptable if diff < 2e-3

### **Fallback 3**: Deep Kernel Debug (High Cost)

**Action**: Line-by-line debugging of online softmax
**Time**: +4 hours
**Risk**: May not find issue

**Recommendation**: Try Fallback 1 or 2 first before deep debug

---

## **Next Phase Preview**

### **Phase B: cuBLAS Q@K^T** (After Phase A Success)

**Goal**: Replace scalar Q@K^T with Tensor Core cuBLAS → 400-500 μs

**Quick Start**:
```cuda
// Replace scalar Q@K^T loop with:
cublasGemmEx(
    cublas_handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    BLOCK_N, BLOCK_M, HEAD_DIM,
    &alpha,
    K_smem, CUDA_R_16F, HEAD_DIM,
    Q_smem, CUDA_R_16F, HEAD_DIM,
    &beta,
    S_tile, CUDA_R_32F, BLOCK_N,
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**Expected**: 2× speedup (839 → 400-500 μs), NCU shows 50-60% TC active

---

## **Current Status Summary**

```
Infrastructure:
✅ Setup script (scripts/phase_a_gpu_setup.sh)
✅ Dual-reference validator (scripts/phase_a_validate_dual_backend.py)
✅ Stable kernel (cudadent42/bench/kernels/fa_phase3_stable.cu)
✅ Execution plan (PHASE_A_EXECUTION.md)

Ready to Execute:
🔵 Task A.1: PyTorch 2.1.0 test (1 hour)
🟡 Task A.2: Build stable kernel (2 hours)
🟡 Task A.3: Dual-reference validation (1 hour)

Blockers:
⚠️  Need GPU instance access (run scripts/phase_a_gpu_setup.sh)

Next:
1. SSH to GPU instance
2. Run: bash scripts/phase_a_gpu_setup.sh
3. Follow interactive prompts
4. Report results
```

---

**Ready to execute on GPU. Run: `bash scripts/phase_a_gpu_setup.sh`**

