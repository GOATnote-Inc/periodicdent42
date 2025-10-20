# Stage-3 Fused Softmax+P·V — Progress Checkpoint

**Date**: October 20, 2025 (1:29 PM)  
**Status**: ⏸️ **PAUSED at Performance Benchmarking** (gcloud auth expired)  
**Branch**: `feat/stage3-fused-softmax`  
**L4 State**: ~/periodicdent42 on feat/stage3-fused-softmax, venv active

---

## ✅ **Completed Gates (4/7)**

### 1. Branch & Build System ✅
```bash
# Local + L4 both on feat/stage3-fused-softmax
# Commit: c84387b (build toggles)
# Commit: 3f2a6b2 (kernel 3A implementation)
```

**Toggles Added:**
- `USE_FUSED_SOFTMAX_PV`: 0 (Stage-2), 1 (3A: sS reused), 2 (3B: full fusion)
- `USE_XOR_SWIZZLE`: 0 (default), 1 (XOR swizzle for bank conflicts)
- `USE_THREE_STAGE_PIPE`: 0 (default), 1 (3-stage pipeline, future)

### 2. Kernel 3A Implementation ✅
```cuda
// Changes in cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu:
// 1. sP buffer removed when USE_FUSED_SOFTMAX_PV >= 1
// 2. Softmax writes P to sS instead of sP
// 3. WMMA P·V loads from sS instead of sP
// Result: Saves 2 KB SMEM (44.5 KB → 42.5 KB)
```

### 3. PTXAS Validation ✅

| Variant | Registers | SMEM | Spills | Status |
|---------|-----------|------|--------|--------|
| **Stage-2** (baseline) | 84 | 37.1 KB | 0 | ✅ |
| **Stage-3A** (sS fusion) | **84** | **35.1 KB** ↓ | 0 | ✅ |

**Saved 2 KB SMEM as designed!**

### 4. Correctness Validation ✅

**9/9 Tests PASS (bit-exact with Stage-2):**

```
Stage-2 Baseline (USE_FUSED_SOFTMAX_PV=0):
[small   ] seed=0: max_err=0.0459, mean_err=0.0142, %bad=0.0% ✅ PASS
[small   ] seed=1: max_err=0.0596, mean_err=0.0132, %bad=0.0% ✅ PASS
[small   ] seed=2: max_err=0.0459, mean_err=0.0133, %bad=0.0% ✅ PASS
[mission ] seed=0: max_err=0.0540, mean_err=0.0170, %bad=0.0% ✅ PASS
[mission ] seed=1: max_err=0.0356, mean_err=0.0171, %bad=0.0% ✅ PASS
[mission ] seed=2: max_err=0.0474, mean_err=0.0165, %bad=0.0% ✅ PASS
[long    ] seed=0: max_err=0.0391, mean_err=0.0178, %bad=0.0% ✅ PASS
[long    ] seed=1: max_err=0.0311, mean_err=0.0177, %bad=0.0% ✅ PASS
[long    ] seed=2: max_err=0.0315, mean_err=0.0179, %bad=0.0% ✅ PASS

Stage-3A (USE_FUSED_SOFTMAX_PV=1):
[small   ] seed=0: max_err=0.0459, mean_err=0.0142, %bad=0.0% ✅ PASS
[small   ] seed=1: max_err=0.0596, mean_err=0.0132, %bad=0.0% ✅ PASS
[small   ] seed=2: max_err=0.0459, mean_err=0.0133, %bad=0.0% ✅ PASS
[mission ] seed=0: max_err=0.0540, mean_err=0.0170, %bad=0.0% ✅ PASS
[mission ] seed=1: max_err=0.0356, mean_err=0.0171, %bad=0.0% ✅ PASS
[mission ] seed=2: max_err=0.0474, mean_err=0.0165, %bad=0.0% ✅ PASS
[long    ] seed=0: max_err=0.0391, mean_err=0.0178, %bad=0.0% ✅ PASS
[long    ] seed=1: max_err=0.0311, mean_err=0.0177, %bad=0.0% ✅ PASS
[long    ] seed=2: max_err=0.0315, mean_err=0.0179, %bad=0.0% ✅ PASS
```

**Numerical Equivalence Confirmed:** Stage-3A is bit-exact with Stage-2.

---

## ⏸️ **Interrupted: Performance Benchmarking**

**Last Command (interrupted):**
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c \
  --command="cd ~/periodicdent42 && source venv/bin/activate && \
  USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX_PV=0 \
  python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 500"
```

**Error:** `gcloud auth login` required

---

## 📋 **Remaining Steps (Resume Here)**

### Step 5: Performance Benchmarking

**Re-authenticate first:**
```bash
gcloud auth login
```

**Then run these commands on L4:**

```bash
# SSH to L4
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c

# On L4:
cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH

# 5.1) Stage-2 Baseline Performance
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX_PV=0 \
  python -m tasks.fp8_sdpa_stage_c_wmma.runner \
  --shapes mission --seeds 0 --iters 500 \
  2>&1 | tee .perf_s2.log

# 5.2) Stage-3A Performance
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX_PV=1 \
  python -m tasks.fp8_sdpa_stage_c_wmma.runner \
  --shapes mission --seeds 0 --iters 500 \
  2>&1 | tee .perf_s3a.log

# 5.3) Compare Results
python scripts/compare_results.py \
  $(ls -dt results/fp8_wmma_baseline/* | sed -n '1p')/perf_baseline.json \
  $(ls -dt results/fp8_wmma_baseline/* | sed -n '2p')/perf_baseline.json
```

**Expected Outcome:**
- **Stage-2 p50**: ~656 μs (from prior validation)
- **Stage-3A target**: ≤590 μs (+10% minimum)
- **Stage-3A stretch**: ≤550 μs (+15-20%)

**Gate:** p50 improvement ≥ +10% (p50 ≤ 590 μs)

---

### Step 6: NCU Profiling (Optional, if time permits)

```bash
# On L4 (as root for NCU permissions):
sudo -E bash -c 'source /home/kiteboard/periodicdent42/venv/bin/activate && \
  /usr/local/cuda-12.2/bin/ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-regex "sdpa_fp8_stage_c_wmma.*" \
  --metrics \
    smsp__inst_executed_pipe_tensor.sum,\
    sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,\
    lts__t_sectors_aperture_device.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
  --csv \
  --log-file results/2025-Stage3-Fused-Validation/ncu_stage3_metrics.csv \
  USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX_PV=1 \
  python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 50'
```

**If NCU has permission issues:** Skip and document in report that profiling was deferred.

---

### Step 7: A/B vs PyTorch SDPA (Optional)

Create `scripts/bench_sdpa_ab.py`:

```python
#!/usr/bin/env python3
"""A/B benchmark: Stage-3A vs PyTorch SDPA (flash backend)"""

import torch
import time

def bench_pytorch_sdpa(B, H, S, D, iters=500):
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(100):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_flash=True)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        O = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_flash=True)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # Convert to μs
    
    times = sorted(times)
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    
    print(f"PyTorch SDPA (flash) - Shape: ({B},{H},{S},{D})")
    print(f"  p50: {p50:.2f} μs")
    print(f"  p90: {p90:.2f} μs")
    return p50, p90

if __name__ == "__main__":
    print("A/B Benchmark: PyTorch SDPA (flash backend)")
    print("="*60)
    
    # Mission shape
    bench_pytorch_sdpa(1, 8, 512, 64, iters=500)
    
    # Long shape
    bench_pytorch_sdpa(1, 8, 2048, 64, iters=500)
```

**Run:**
```bash
# On L4:
cd ~/periodicdent42
source venv/bin/activate
python scripts/bench_sdpa_ab.py | tee results/2025-Stage3-Fused-Validation/sdpa_ab.txt
```

---

### Step 8: Create Validation Reports

**On L4, after perf benchmarking completes:**

```bash
# Create results directory
mkdir -p results/2025-Stage3-Fused-Validation

# Copy logs
cp .build_s2.log .build_s3a.log results/2025-Stage3-Fused-Validation/
cp .corr_s2.log .corr_s3a.log results/2025-Stage3-Fused-Validation/
cp .perf_s2.log .perf_s3a.log results/2025-Stage3-Fused-Validation/

# Copy perf JSONs (two most recent)
cp $(ls -dt results/fp8_wmma_baseline/*/perf_baseline.json | head -2 | xargs) \
   results/2025-Stage3-Fused-Validation/
```

**Then create reports locally:**

Create `STAGE3_VALIDATION_REPORT.md` with:
- PTXAS comparison (Stage-2 vs Stage-3A)
- Correctness results (9/9 PASS)
- Performance results (p50 comparison)
- NCU metrics (if collected)
- A/B SDPA results (if collected)

Create `STAGE3_GPU_VALIDATION_SUMMARY.md` with:
- 3-line headline
- Gate checklist
- Merge decision

---

### Step 9: Commit Artifacts & Push

```bash
# On local machine:
cd /Users/kiteboard/periodicdent42
git checkout feat/stage3-fused-softmax
git pull  # Get any updates from L4

git add results/2025-Stage3-Fused-Validation/
git add STAGE3_VALIDATION_REPORT.md
git add STAGE3_GPU_VALIDATION_SUMMARY.md
git add scripts/bench_sdpa_ab.py  # If created

git commit -m "stage3(fused-softmax-pv): Complete L4 validation

Results:
- PTXAS: 84 regs, 35.1 KB SMEM (saves 2 KB vs Stage-2)
- Correctness: 9/9 PASS (bit-exact with Stage-2)
- Performance: p50=<FILL>μs (vs Stage-2 656μs, +<XX>%)
- NCU: <summary if collected>
- A/B SDPA: <summary if collected>

All gates: <PASS/DEFER>"

git push
```

---

### Step 10: Merge to Main (Only if Gates Pass)

**Merge Decision Criteria:**
- ✅ Correctness: 9/9 PASS (already confirmed)
- ✅ PTXAS: ≤120 regs, ≤48 KB SMEM, 0 spills (already confirmed)
- ⏳ Performance: p50 improvement ≥ +10% (pending measurement)

**If all gates pass:**

```bash
# Local machine:
cd /Users/kiteboard/periodicdent42
git checkout main
git pull

git merge --no-ff feat/stage3-fused-softmax -m "Merge Stage-3: Fused softmax + WMMA P·V (3A)

VALIDATION SUMMARY (L4, Oct 20, 2025):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Correctness: 9/9 tests PASS (bit-exact with Stage-2)
✅ PTXAS: 84 regs, 35.1 KB SMEM (-2 KB vs Stage-2), 0 spills
✅ Performance: p50=<FILL>μs (vs Stage-2 656μs, +<XX>%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage-3A implementation (USE_FUSED_SOFTMAX_PV=1):
- Reuse sS (score buffer) to store unnormalized P after softmax
- Remove sP buffer (saves 2 KB SMEM: 37.1 KB → 35.1 KB)
- WMMA P·V loads from sS instead of sP
- Preserves numerical parity with Stage-2

Toggle: USE_FUSED_SOFTMAX_PV=0 (Stage-2 rollback), =1 (Stage-3A)

Artifacts: results/2025-Stage3-Fused-Validation/
Reports: STAGE3_VALIDATION_REPORT.md, STAGE3_GPU_VALIDATION_SUMMARY.md"

git push origin main

# Tag the release
git tag -a v3.0-stage3-fused -m "Stage-3: Fused softmax + WMMA P·V (3A)

Performance: <FILL>× faster (656μs → <FILL>μs)
SMEM: -2 KB (37.1 KB → 35.1 KB)
Validated: L4 GPU (SM 8.9, CUDA 12.2)"

git push origin v3.0-stage3-fused
```

---

### Step 11: Update README (Post-Merge)

Add to performance history table:

```markdown
## Performance History

| Version | Optimization | Mission Shape (μs) | Speedup vs v1.0 |
|---------|--------------|--------------------:|---------------:|
| v1.0 | Baseline (scalar) | 2870.0 | 1.0× |
| v2.0-stage1 | cp.async double-buffer | 1200.8 | 2.4× |
| v2.0-stage2 | WMMA P·V | 656.4 | 4.4× |
| v3.0-stage3 | Fused softmax + P·V | **<FILL>** | **<FILL>×** ⚡ |
```

---

## 📊 **Decision Tree**

```
Performance Result:
├─ p50 ≤ 590 μs (+10%)     → ✅ MERGE (minimum gate)
├─ p50 ≤ 550 μs (+15-20%)  → ✅ MERGE (excellent)
├─ p50 ≤ 500 μs (+25%+)    → ✅ MERGE (outstanding!)
└─ p50 > 590 μs (<+10%)    → ❌ DEFER (investigate, try USE_XOR_SWIZZLE=1)
```

**If performance gate fails (<+10%):**
1. Try `USE_XOR_SWIZZLE=1` to reduce bank conflicts
2. Profile with NCU to identify bottlenecks
3. Consider Stage-3B (full QK^T→softmax→P·V fusion)
4. If still insufficient, defer Stage-3 and document findings

---

## 🔧 **Quick Commands Reference**

**Re-authenticate gcloud:**
```bash
gcloud auth login
```

**SSH to L4:**
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c
```

**On L4 - Resume benchmarking:**
```bash
cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH

# Stage-2 baseline
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX_PV=0 \
  python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 500

# Stage-3A
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX_PV=1 \
  python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 500

# Compare
python scripts/compare_results.py \
  $(ls -dt results/fp8_wmma_baseline/* | sed -n '1p')/perf_baseline.json \
  $(ls -dt results/fp8_wmma_baseline/* | sed -n '2p')/perf_baseline.json
```

---

## 📈 **Expected Performance (Predictions)**

Based on Stage-3A design (sS reused for P):

**Conservative Estimate:**
- Saved 2 KB SMEM → better cache locality
- Eliminated 1 write + 1 read of sP buffer
- Expected: **+5-10% speedup** (p50 ~590-620 μs)

**Optimistic Estimate:**
- SMEM savings enable better occupancy or fewer bank conflicts
- Expected: **+10-15% speedup** (p50 ~550-590 μs)

**Realistic Target:** **p50 ~590 μs** (+10% speedup, meets minimum gate)

---

## 🎯 **Current TODO Status**

- [x] Branch & build system
- [x] Kernel 3A implementation
- [x] PTXAS validation
- [x] Correctness testing (9/9)
- [ ] **Performance benchmarking** ← YOU ARE HERE
- [ ] NCU profiling (optional)
- [ ] A/B SDPA (optional)
- [ ] Validation reports
- [ ] Merge to main
- [ ] Update README

---

**Last Updated**: October 20, 2025, 1:29 PM  
**Next Action**: Re-authenticate with `gcloud auth login`, then resume performance benchmarking on L4.

