# FlashCore Fused Kernel - Quick Start Guide

**Phase 2 Implementation Complete!** ✅  
**Ready to test on GCP L4**

---

## 🚀 Quick Test (5 minutes)

### 1. SSH to GCP L4 Instance
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c
```

### 2. Copy Files to Instance
```bash
# From local machine (separate terminal)
cd /Users/kiteboard/periodicdent42

# Copy entire flashcore directory
gcloud compute scp --recurse flashcore/ cudadent42-l4-dev:~/flashcore/ --zone=us-west1-c
```

### 3. Run Test on Instance
```bash
# On GCP instance
cd ~/flashcore
python3 test_fused.py
```

**Expected output**:
- ✅ Build successful (with PTXAS resource usage)
- ✅ Correctness tests (max_err < 0.06)
- ✅ Performance tests (p50 latency in μs)
- ✅ Summary report

---

## 📊 What to Look For

### Correctness ✅ CRITICAL
```
  Correctness:
    max_err:  0.XXXXXX  <-- Must be < 0.06
    mean_err: 0.XXXXXX
  ✅ PASS (correctness)
```

**If max_err > 0.06**: See debugging section below.

### Performance 🎯 TARGET
```
  Performance:
    p50: XX.XX μs         <-- Target: <100 μs (good), <50 μs (excellent), <40 μs (stretch)
    p90: XX.XX μs
    Speedup vs baseline: X.XX×  <-- Target: ≥6× (good), ≥13× (excellent)
```

### PTXAS Resource Usage 📊
```
ptxas info    : Used XX registers, YY+ZZ bytes smem
```

**Targets**:
- Registers: ≤96 per thread (ideally <80)
- SMEM: ~18 KB static (4608 bytes = 18 KB)
- **Spills**: MUST be 0 (any spills = performance killer)

---

## 🎯 Success Criteria

| Result | Interpretation | Next Step |
|--------|----------------|-----------|
| ✅ Correctness + p50 < 50 μs | **EXCELLENT!** | Proceed to Phase 4 (optimize to <40 μs) |
| ✅ Correctness + p50 < 100 μs | **GOOD!** | Proceed to Phase 4 (expand to 64×64 tiles) |
| ✅ Correctness + p50 < 200 μs | **OK** | Debug performance (likely register spills) |
| ❌ Correctness (max_err > 0.06) | **NEEDS DEBUG** | Fix fragment LUT or softmax logic |
| ❌ Build fails | **NEEDS DEBUG** | Check CUDA version, compiler flags |

---

## 🐛 Quick Debugging

### If Correctness Fails (max_err > 0.06)

**Most likely cause**: WMMA fragment coordinate mapping is incorrect.

**Quick fix**: Replace `get_wmma_frag_coord` with reference LUT:

```bash
# Check if LUT file exists
ls ~/periodicdent42/cudadent42/bench/kernels/wmma16x16_accum_lut.h

# If exists, copy the correct mapping from sdpa_fp8_stage_c_wmma.cu
# Lines 118-126 (fragment layout)
```

**Or try simplified fragment access** (less efficient but correct):
1. Store c_frag_qk to shared memory using `wmma::store_matrix_sync`
2. Read back values for softmax computation
3. Verify correctness first, optimize later

---

### If Performance Is Slow (>200 μs)

**Check PTXAS output for**:
```
ptxas info    : Used XX registers, YY+ZZ bytes smem
ptxas info    : Compiling entry function '...' for 'sm_89'
```

**Common issues**:
1. **Register spills** (`XX bytes stack frame` in PTXAS):
   - **Fix**: Add `#pragma unroll 1` to reduce unrolling
   - Or use more shared memory, fewer local variables

2. **Low occupancy** (< 20% from NCU):
   - **Fix**: Check if SMEM or registers are limiting
   - May need to reduce tile size temporarily

3. **WMMA not being used** (scalar fallback):
   - **Fix**: Verify `sm_89` in build flags
   - Check for alignment issues (all pointers 16-byte aligned)

---

### If Build Fails

**Common errors**:
```
error: identifier "wmma" is undefined
```
→ Check CUDA version: `nvcc --version` (need ≥12.0 for sm_89)

```
error: unrecognized option '-arch=sm_89'
```
→ Upgrade CUDA toolkit: `sudo apt-get install cuda-toolkit-12-0`

```
torch._C._LinkerError: ...
```
→ Ensure all `.cu` files (not `.cpp`) are compiled with NVCC

---

## 📈 Performance Tuning (After Correctness Passes)

### Quick Profile with NCU
```bash
ncu --set full --target-processes all --launch-count 5 \
    --kernel-name flashcore_fused_wmma_kernel \
    python3 -c "
from build_fused import build_fused
import torch
ext = build_fused()
Q = torch.randn(1,8,512,64, dtype=torch.float16, device='cuda')
K = torch.randn(1,8,512,64, dtype=torch.float16, device='cuda')
V = torch.randn(1,8,512,64, dtype=torch.float16, device='cuda')
ext.forward(Q,K,V,0.125)
" 2>&1 | tee ncu_fused.txt

# Key metrics to check
grep "Compute \(SM\) Throughput" ncu_fused.txt
grep "Memory Throughput" ncu_fused.txt
grep "Warp Occupancy" ncu_fused.txt
```

---

## 🎉 Expected Results

### Conservative Estimate (32×32 tiles, no cp.async)
- **Correctness**: ✅ Should pass (algorithm is proven)
- **Latency**: 50-100 μs
- **Speedup**: 6-13× from 634 μs baseline

### If We Hit These Numbers...

| p50 Latency | Speedup | Grade | Next Step |
|-------------|---------|-------|-----------|
| **40-50 μs** | ~13-16× | **A+** 🎉 | Already met stretch goal! |
| **50-70 μs** | ~9-13× | **A** ✅ | Expand to 64×64 tiles |
| **70-100 μs** | ~6-9× | **B+** ✅ | Add cp.async + 64×64 |
| **100-150 μs** | ~4-6× | **B** | Optimize atomics, profile |
| **150-200 μs** | ~3-4× | **C** | Debug register usage |
| **>200 μs** | <3× | **D** | Significant debugging needed |

---

## 📞 Reporting Results

Please share:
1. **Correctness**: max_err values for all 3 shapes
2. **Performance**: p50 latency for mission shape (B=1, H=8, S=512, D=64)
3. **PTXAS output**: Registers and SMEM usage
4. **Any errors**: Full error messages if build or test fails

---

## 🚀 If Everything Works...

**Celebrate!** 🎉 Then proceed to:

### Phase 4A: Expand to 64×64 tiles
- Change `TILE_M`, `TILE_N` to 64 in kernel
- Request 100 KB SMEM with `cudaFuncSetAttribute`
- Expected: 40-50 μs

### Phase 4B: Add 2-stage cp.async
- Double-buffer K/V loads
- Overlap memory and compute
- Expected: 35-45 μs

### Phase 4C: Final tuning
- NCU-guided optimization
- **Target: <40 μs achieved!** 🎯

---

**Ready to test? Let's see those numbers!** 🔥

```bash
# One command to rule them all
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c --command="cd ~/flashcore && python3 test_fused.py"
```

