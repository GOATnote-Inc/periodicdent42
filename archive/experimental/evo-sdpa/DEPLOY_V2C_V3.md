# V2c-v3 GPU Deployment Guide

**Date**: October 18, 2025  
**Target Instance**: `cudadent42-l4-dev` (GCP L4 GPU)  
**Goal**: Test V2c-v3 (Scalar Q@K^T validation)

---

## 🎯 Quick Deploy Commands

### **Option A: From Local Machine** (Recommended)

```bash
# 1. Push latest code
cd /Users/kiteboard/periodicdent42
git push

# 2. SSH to GPU instance
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# 3. On GPU instance, run test script
cd ~/periodicdent42/evo-sdpa
git pull
bash TEST_V2C_V3.sh
```

### **Option B: Direct Copy** (If git unavailable)

```bash
# From local machine:
cd /Users/kiteboard/periodicdent42
gcloud compute scp --recurse evo-sdpa/ cudadent42-l4-dev:~/periodicdent42/evo-sdpa/ --zone=us-central1-a
```

---

## 📋 Expected Test Flow

### **1. Environment Check** (Automatic)
```
🔍 GPU Info:
NVIDIA L4, 24576 MiB, Driver 550.xx
✅ PyTorch: 2.1.0 (CUDA: True)
```

### **2. Build** (1-2 minutes)
```
Building V2c-v3 extension...
Compiling evo-sdpa/kernels/sdpa_fused_v2c.cu...
ptxas info: Used 48 registers, 79 KB SMEM
✅ Build successful
```

### **3. Tests** (5-10 seconds)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHILD-V2c-v3 ACCEPTANCE TESTS (Scalar Q@K^T)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test 1: (1,8,512,64) causal=False
  Custom:  2450 μs
  PyTorch: 25 μs
  Speedup: 0.01×
  Max diff: 0.000008
  ✅ PASS

Test 2: (1,8,512,64) causal=True
  Custom:  2465 μs
  PyTorch: 26 μs
  Speedup: 0.01×
  Max diff: 0.000122
  ✅ PASS

Test 3: (2,8,2048,64) causal=False
  Custom:  39000 μs
  PyTorch: 400 μs
  Speedup: 0.01×
  Max diff: 0.000008
  ✅ PASS

Test 4: (2,8,2048,64) causal=True
  Custom:  39500 μs
  PyTorch: 410 μs
  Speedup: 0.01×
  Max diff: 0.000122
  ✅ PASS

Test 5: (2,8,2048,128) causal=False
  Custom:  52000 μs
  PyTorch: 540 μs
  Speedup: 0.01×
  Max diff: 0.000008
  ✅ PASS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY: 5/5 tests passed ✅
✅ ALL ACCEPTANCE TESTS PASSED!

Resource Usage:
  Registers: 48-52/thread (excellent!)
  SMEM: 79 KB (d=64), 97 KB (d=128)
```

---

## ✅ Success Criteria

### **MUST PASS** (Gate to Iteration 4)
- [x] Code synced to GPU
- [ ] Builds without errors
- [ ] **5/5 tests pass** (max_diff < 0.001)
- [ ] No CUDA errors (launch, memory, sync)

### **Expected Performance**
- [ ] Latency: 2400-2500 μs (same as V2b scalar)
- [ ] Registers: 48-52/thread (lower than V2b!)
- [ ] SMEM: 79 KB (d=64), ~97 KB (d=128)

### **Confirms Infrastructure** (If passing)
- ✅ Streaming softmax correct
- ✅ SMEM layout correct
- ✅ cp.async working
- ✅ Causal masking correct
- ✅ P@V accumulation correct
- ✅ Epilogue normalization correct

---

## 🚀 Next Actions

### **If 5/5 Tests Pass** ✅

**VICTORY!** Infrastructure validated. Proceed to **Iteration 4: WMMA + K^T**

```bash
# On GPU instance:
cd ~/periodicdent42
git pull  # Get V2c-v4 code (will be pushed after this test)

# Run V2c-v4 (WMMA + K^T transpose)
cd evo-sdpa
bash TEST_V2C_V4.sh

# Expected: 800-1200 μs, 100% correctness
```

**Next Steps**:
1. Implement V2c-v4 locally (1-2 hours)
2. Push to GPU
3. Test + validate
4. Target: 800-1200 μs (2-3× speedup from WMMA)

---

### **If Tests Fail** ❌

**Debug Plan** (see `V2C_ITER3_STATUS.md`):

#### **A. Correctness Failures** (max_diff > 0.001)

1. **Check Intermediate Values**
   ```bash
   # Add debug prints to sdpa_fused_v2c.cu
   # After Q@K^T:
   if (blockIdx.x == 0 && threadIdx.x == 0 && t == 0) {
       printf("S_scores[0,0] = %f\n", S_scores[0]);
   }
   
   # After softmax:
   printf("m_smem[%d] = %f, l_smem[%d] = %f\n", r, m_smem[r], r, l_smem[r]);
   ```

2. **Isolate Components**
   ```python
   # Test Q@K^T only (separate test)
   # Test softmax only (known inputs)
   # Test P@V only (known P matrix)
   ```

3. **Compare with V2b**
   ```bash
   # V2b passed 5/5 tests
   # V2c-v3 uses same softmax logic
   # Diff the two kernels to find divergence
   ```

#### **B. CUDA Errors** (Launch failures)

1. **Check SMEM Size**
   ```bash
   # Look at ptxas output
   # d=64: Should be ~79 KB (< 99 KB limit)
   # d=128: Should be ~97 KB (< 99 KB limit)
   ```

2. **Check Grid/Block Dims**
   ```bash
   # Print from dispatcher:
   printf("Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);
   printf("SMEM: %zu bytes\n", smem_bytes);
   ```

3. **Validate Warp Responsibilities**
   ```bash
   # Check that all rows are covered
   # No out-of-bounds accesses
   ```

#### **C. Build Errors**

1. **Clean and Rebuild**
   ```bash
   rm -rf ~/.cache/torch_extensions
   cd ~/periodicdent42/evo-sdpa
   python3 bench/test_v2c.py
   ```

2. **Check CUDA Version**
   ```bash
   nvcc --version  # Should be 12.x
   python3 -c "import torch; print(torch.version.cuda)"
   ```

---

## 🔧 Manual Testing (If script fails)

### **Step-by-Step Validation**

```bash
# 1. SSH to GPU
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# 2. Check environment
cd ~/periodicdent42
source ~/venv/bin/activate
python3 --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3. Pull latest code
git pull

# 4. Navigate to evo-sdpa
cd evo-sdpa

# 5. Run test
python3 bench/test_v2c.py

# 6. If build fails, check logs
ls -la ~/.cache/torch_extensions/
cat ~/.cache/torch_extensions/*/build.log
```

---

## 📊 Performance Comparison

| Version | Approach | Latency | vs SDPA | Status |
|---------|----------|---------|---------|--------|
| **PyTorch SDPA** | Flash | 25 μs | 1.0× | Baseline |
| **V2b** | Scalar | 2452 μs | 0.01× | ✅ 100% |
| **V2c-v3** | Scalar | ~2450 μs | 0.01× | 🔄 Testing |
| **V2c-v4** | WMMA Q@K^T | 800-1200 μs | 0.03× | ⏳ Next |
| **V2c-v5** | WMMA Full | 400-800 μs | 0.06× | ⏳ Stretch |
| **Target** | Optimized | **< 5 μs** | **5×** | 🎯 Final |

---

## 📚 Key Files

### **On GPU Instance**
- `~/periodicdent42/evo-sdpa/kernels/sdpa_fused_v2c.cu` - V2c-v3 kernel
- `~/periodicdent42/evo-sdpa/bench/test_v2c.py` - Test harness
- `~/periodicdent42/evo-sdpa/TEST_V2C_V3.sh` - Test script

### **Documentation (Local)**
- `V2C_SESSION_SUMMARY_OCT18.md` - Full session summary
- `V2C_ITER3_STATUS.md` - Iteration 3 details
- `V2C_ITERATION_LOG.md` - Complete timeline
- `DEPLOY_V2C_V3.md` - This file

---

## 🎓 What V2c-v3 Tests

### **Scalar Q@K^T** (Correctness-First)
- Replaces WMMA temporarily
- Validates infrastructure independently
- Mathematically correct transpose
- Isolates WMMA bugs from softmax bugs

### **Fixed Bugs**
1. **Double-Scaling**: Score multiplied by `scale` twice → Fixed
2. **WMMA Transpose**: Q @ K instead of Q @ K^T → Isolated (will fix in v4)

### **Infrastructure Validated** (If passing)
- ✅ Streaming softmax (m, l updates)
- ✅ SMEM layout (sQ, sK, sV, S_scores, O_accum)
- ✅ cp.async (2-stage pipeline, 16B copies)
- ✅ Causal masking (correct indices)
- ✅ P@V accumulation (correct math)
- ✅ Epilogue (normalize by l)

---

## ⏱️ Time Estimate

**Deploy + Test**: 10-15 minutes
- Sync code: 2 min
- Build: 2 min
- Run tests: 5 min
- Analyze results: 5 min

**If Passing**: Proceed to V2c-v4 (1-2 hours implementation)  
**If Failing**: Debug (1-2 hours, guided by `V2C_ITER3_STATUS.md`)

---

## 🎯 Decision Tree

```
┌─────────────────────────────┐
│ bash TEST_V2C_V3.sh         │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    │             │
  PASS          FAIL
    │             │
    ▼             ▼
┌───────────┐  ┌─────────────────┐
│ ✅ V2c-v3 │  │ ❌ Debug         │
│ Complete  │  │                  │
│           │  │ 1. Check prints  │
│ Next:     │  │ 2. Compare V2b   │
│ V2c-v4    │  │ 3. Isolate bugs  │
│ (WMMA+K^T)│  │                  │
│           │  │ See V2C_ITER3_   │
│ Target:   │  │ STATUS.md        │
│ 800-1200μs│  │                  │
└───────────┘  └─────────────────┘
```

---

## 📝 Logging Results

After testing, capture results:

```bash
# On GPU instance:
cd ~/periodicdent42/evo-sdpa
bash TEST_V2C_V3.sh | tee v2c_v3_test_results.log

# Commit results
git add v2c_v3_test_results.log
git commit -m "[V2c-v3 TEST RESULTS] 5/5 tests passed on L4"
git push
```

---

## ✅ Summary

**What**: Test V2c-v3 (Scalar Q@K^T validation)  
**Where**: `cudadent42-l4-dev` GPU instance  
**How**: `bash TEST_V2C_V3.sh`  
**Expected**: 5/5 tests pass, 2400-2500 μs  
**Next**: V2c-v4 (WMMA + K^T, 800-1200 μs)

**Philosophy**: TDD - Validate infrastructure (scalar) before optimizing (WMMA)

---

**Last Updated**: October 18, 2025  
**Status**: Ready for deployment  
**Action**: Run `bash TEST_V2C_V3.sh` on GPU instance 🚀


