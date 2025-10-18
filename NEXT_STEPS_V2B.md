# 🚀 Next Steps: Child-V2b GPU Testing

**Created**: Oct 18, 2025  
**Status**: Implementation complete, awaiting GPU validation  

---

## 📦 What's Ready

### ✅ Complete Implementation

- `evo-sdpa/kernels/sdpa_fused_v2b.cu` - Correctness-first kernel
- `evo-sdpa/kernels/runtime.hpp` - Updated dispatcher
- `evo-sdpa/kernels/sdpa_fused_bindings.cpp` - PyTorch bindings
- `evo-sdpa/bench/test_v2b.py` - Acceptance test suite
- `evo-sdpa/bench/parse_ptxas.py` - Resource usage parser
- `evo-sdpa/DEPLOY_V2B.md` - Deployment guide
- `evo-sdpa/V2B_STATUS.md` - Implementation summary

---

## 🎯 Action Items (GPU Required)

### **STEP 1: Deploy Code**

```bash
# On GPU instance (cudadent42-l4-dev):
cd ~/periodicdent42
git pull  # Get latest V2b code

# Verify files present:
ls -la evo-sdpa/kernels/sdpa_fused_v2b.cu
ls -la evo-sdpa/bench/test_v2b.py
```

### **STEP 2: Activate Environment**

```bash
# Assuming venv setup exists:
source ~/periodicdent42/venv/bin/activate  # Adjust path if different
cd ~/periodicdent42/evo-sdpa/bench
```

### **STEP 3: Run Acceptance Tests**

```bash
python test_v2b.py 2>&1 | tee v2b_results.txt
```

**What to look for**:
- ✅ "Build successful" message
- ✅ "5/5 tests passed"
- ❌ Any CUDA errors or correctness failures

### **STEP 4: Check Resource Usage**

```bash
python parse_ptxas.py v2b_results.txt
```

**Validate**:
- Registers ≤ 72/thread
- SMEM ≤ 96 KB/CTA

---

## 🔀 Decision Tree

### ✅ Scenario A: ALL TESTS PASS (5/5 correctness)

**Victory!** Streaming softmax is correct.

**Next**: Implement Child-V2c (Full WMMA)
```bash
# Expected speedup: 3-7× from Tensor Cores
# Timeline: 4-6 hours
# Goal: 200-400 μs on (1,8,512,64)
```

**Levers**:
1. Replace scalar Q@K^T with WMMA 16×16×16
2. Replace scalar P@V with WMMA 16×16×16
3. Keep streaming softmax structure (verified correct)

---

### ⚠️ Scenario B: SOME TESTS FAIL (1-4/5)

**Debug needed**. Which shapes fail?

#### B.1: d=128 fails, d=64 passes
**Hypothesis**: SMEM overflow or indexing bug for d=128

**Debug**:
```cuda
// Add to sdpa_fused_v2b.cu:
if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("d=%d, M=%d, N=%d, SMEM calculated=%zu\\n", 
           d, M, N, SmemLayout<HEAD_DIM, STAGES>::total_bytes);
}
```

**Fix**: Reduce `N` for d=128 or increase SMEM budget

#### B.2: causal=True fails, causal=False passes
**Hypothesis**: Causal mask logic bug

**Debug**:
```cuda
// Check mask application:
if (causal && k_pos > q_pos) {
    dot = -FLT_MAX;  // Verify this is executed
}
```

**Fix**: Ensure mask applied before `exp`

#### B.3: Large L fails (2048), small L passes (512)
**Hypothesis**: K-tile iteration bug or numerical instability

**Debug**:
```cuda
// Track (m,l) evolution:
if (lane == 0 && r == 0 && t % 10 == 0) {
    printf("tile %d: m=%.6f, l=%.6f\\n", t, m_smem[r], l_smem[r]);
}
```

**Fix**: Verify streaming softmax invariants hold

---

### ❌ Scenario C: CUDA LAUNCH ERROR

**Common errors**:

#### C.1: `invalid configuration argument`
**Cause**: SMEM > 96 KB or grid/block mismatch

**Fix**:
```cpp
// Check dispatcher in runtime.hpp:
printf("Grid: (%d,%d), Block: (%d), SMEM: %zu KB\\n",
       grid.x, grid.y, block.x, smem_bytes / 1024);
```

Reduce `N` if SMEM too large.

#### C.2: `misaligned address`
**Cause**: cp.async not 16B aligned

**Fix**:
```cuda
// Verify alignment in kernel:
assert((size_t)&sK[smem_idx] % 16 == 0);
assert((size_t)&K_bh[global_offset] % 16 == 0);
```

Add scalar fallback for unaligned cases.

#### C.3: `unspecified launch failure`
**Cause**: OOB access, race, or assert

**Debug**: Build with `-G` (device debug):
```python
# In bench_sdpa.py:
extra_cuda_cflags=["-G", "-arch=sm_89", ...]
```

Run with `cuda-gdb` or add `printf` debugging.

---

## 📊 Performance Expectations

### V2b (Current - Scalar Path)

| Shape | Expected μs | vs PyTorch | Notes |
|-------|-------------|------------|-------|
| (1,8,512,64) | 800-1200 | 0.03-0.04× | Scalar, correctness-first |
| (2,8,2048,64) | 12000-18000 | Similar | Scales with L² |

**Why slow?** No WMMA yet, focus on correctness.

### V2c (Next - Full WMMA)

| Shape | Expected μs | vs PyTorch | Speedup |
|-------|-------------|------------|---------|
| (1,8,512,64) | 200-400 | 0.1-0.2× | 3-5× from V2b |
| (2,8,2048,64) | 3000-5000 | Similar | Tensor Cores |

**Target**: Beat 1000 μs on mission shape.

### V2d+ (Future - NCU Tuned)

| Shape | Expected μs | vs PyTorch | Speedup |
|-------|-------------|------------|---------|
| (1,8,512,64) | 50-100 | 0.5-1.0× | 4-8× from V2c |

**Target**: Approach PyTorch parity (~31 μs).

---

## 🎓 What We Learned (So Far)

### From V1 → V2b Journey

1. **Correctness is non-negotiable**
   - V2 had "framework" but 0% correctness
   - V2b rebuilds from scratch, correctness-first

2. **Single-warp ownership crucial**
   - Streaming softmax requires per-row state
   - Inter-warp races killed V2

3. **cp.async is strict**
   - Must use 4/8/16 byte copies
   - Must align addresses properly
   - Must commit/wait in correct order

4. **SMEM is the bottleneck**
   - 96 KB fills fast with double-buffering
   - Need careful budget management
   - Padding helps but costs space

5. **EvoEngineer works**
   - Structured approach found bugs
   - Test-driven mindset paid off
   - Iterative refinement (V2→V2b→V2c) is right

---

## 📚 Reference Documents

- `evo-sdpa/DEPLOY_V2B.md` - Full deployment guide
- `evo-sdpa/V2B_STATUS.md` - Implementation details
- `evo-sdpa/00_task.md` - Task context (I1)
- `evo-sdpa/01_generate.md` - EvoEngineer-Free prompt
- `cudadent42/PHASE_D3_FINAL_RESULTS.md` - FP8 journey learnings

---

## ⏱️ Time Budget

### Already Spent
- Phase A (PyTorch 2.1.0 correctness): 4 hours ✅
- Phase B (cuBLAS hybrid): 6 hours ✅
- Phase C (Backend testing): 8 hours ✅
- EvoEngineer V1: 2 hours ✅
- EvoEngineer V2b: 3 hours ✅
- **Total: 23 hours**

### Remaining Budget (if continuing)
- V2b validation: 1 hour
- V2c (WMMA): 4-6 hours
- V2d (NCU + I3): 2-3 hours
- Elite loop: 3-4 hours
- **Total needed: ~10-15 hours**

### To reach < 100 μs
**Realistic**: 10-15 more hours (total ~35-40 hours)  
**Optimistic**: Catch a lucky break at 5-8 hours  
**Pessimistic**: Weeks (research needed)

---

## ✅ Success Criteria

### V2b (Current Phase)

**MUST HAVE**:
- [x] Code complete and committed
- [ ] Builds without errors on GPU
- [ ] 5/5 acceptance tests pass
- [ ] No CUDA runtime errors

**NICE TO HAVE**:
- [ ] Faster than V1 baseline (1378 μs)
- [ ] Resource usage optimal (regs ≤64, SMEM ≤80KB)

### V2c (Next Phase)

**MUST HAVE**:
- [ ] WMMA implemented (Q@K^T, P@V)
- [ ] Maintains 100% correctness
- [ ] 3× faster than V2b (≤400 μs)

**NICE TO HAVE**:
- [ ] 10× faster than V1 baseline (≤140 μs)
- [ ] Tensor core utilization >50%

---

## 🚀 Quick Commands (Copy-Paste)

### Deploy + Test (GPU)
```bash
cd ~/periodicdent42 && \
git pull && \
source venv/bin/activate && \
cd evo-sdpa/bench && \
python test_v2b.py 2>&1 | tee v2b_results.txt && \
python parse_ptxas.py v2b_results.txt
```

### Debug Single Shape
```bash
python -c "
from bench_sdpa import build_ext, run_case
import torch
mod = build_ext()
run_case(mod, B=1, H=8, L=512, d=64, causal=False, verbose=True, iters=10)
"
```

### Check GPU Status
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

**Last Update**: Oct 18, 2025  
**Ready**: Yes ✅  
**Blocking**: GPU access only  

**When tests complete, report back results and we'll proceed to V2c (WMMA) or debug as needed.**



