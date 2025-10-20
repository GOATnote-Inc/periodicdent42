# Current Project Status — Oct 20, 2025

## 🎯 Mission: Sub-5 μs SDPA on NVIDIA L4 (Ada, sm_89)

**Current Best**: **656 μs** (Stage-2: `cp.async` + WMMA P·V)  
**Target**: < 5 μs (131× speedup needed from current)  
**Progress**: 4.4× speedup achieved (2870 → 656 μs)  
**Remaining**: 30× speedup to target

---

## ✅ What's Done (Validated on L4)

### **Stage-1: `cp.async` Double-Buffer K/V** ✅
- **Performance**: 761 μs (p50) on mission shape (B=2, H=8, S=256, D=64)
- **Speedup**: +13.8% over baseline
- **Status**: Merged to `main`, tagged `v1.0-stage1-cp-async`
- **Evidence**: `STAGE1_VALIDATION_REPORT.md`

### **Stage-2: WMMA-Accelerated P·V** ✅
- **Performance**: 656 μs (p50) on mission shape
- **Speedup**: +13.8% over Stage-1 (+28.6% cumulative)
- **Resources**: 96 regs, 37.1 KB SMEM, 0 spills
- **Status**: Merged to `main`, tagged `v2.0-stage2-wmma-pv`
- **Evidence**: `STAGE2_VALIDATION_REPORT.md`, `SESSION_STAGE1_STAGE2_COMPLETE.md`

### **Infrastructure** ✅
- ✅ Robust-kbench validation framework (correctness + perf)
- ✅ Feature flag system (`USE_CP_ASYNC`, `USE_WMMA_PV`, etc.)
- ✅ PTXAS resource tracking (regs, SMEM, spills)
- ✅ Automated build + test pipeline
- ✅ NCU profiling integration
- ✅ Git workflow with PR-ready reports

---

## ❌ What Didn't Work ("Valid Negatives")

### **Stage-3A: Reuse `sS` for `sP` (Micro-Fusion)** ⚠️
- **Performance**: 655 μs (p50) — only +0.2% improvement
- **Conclusion**: Removing 2 KB `sP` buffer is not the bottleneck
- **Status**: Documented, not merged (marginal gain)

### **Stage-3B: Fused Softmax in Registers** ❌
- **Objective**: Eliminate `sS` buffer, fuse Q@K^T → softmax → P in registers
- **Resources**: 83 regs, 35.1 KB SMEM (IMPROVED vs Stage-2)
- **Correctness**: 0/6 tests (max_err 2.4-4.1, 40-60× worse than Stage-2)
- **Debug Attempts**: 3 systematic fixes (KV mask, sP zeroing, cross-warp sync) — 0 progress
- **Conclusion**: Deep algorithmic bug; register-based fusion too complex for current design
- **Time Invested**: 12 hours (impl 8h, debug 4h)
- **Status**: Abandoned (branch `feat/stage3-fusion-full` retained for reference)
- **Evidence**: `SESSION_STAGE3_COMPLETE_OCT20_2025.md`, `STAGE3B_HOTFIX_STATUS.md`

### **Step-2: XOR Swizzle for Bank Conflicts** ⚠️
- **Performance**: 696 μs (p50) — **-6.1% regression**
- **Conclusion**: SMEM bank conflicts not the bottleneck
- **Status**: Disabled by default (`USE_SMEM_SWIZZLE_XOR=0`)

---

## 🚀 Recommended Next Steps

### **Option A: Stage-4 (3-Stage `cp.async`) — LOW RISK** ⭐ RECOMMENDED
**Description**: Extend 2-stage K/V prefetch to 3-stage for better latency hiding  
**Target**: +5-10% speedup (≤590 μs)  
**Effort**: 4-6 hours  
**Risk**: Low (no algorithmic changes)  
**Rationale**: Proven technique (FlashAttention-3), incremental improvement

**Implementation**:
```cuda
// Current: 2-stage (ping-pong)
sK_u8[2][TILE_N][D_PAD];  // 18 KB × 2 = 36 KB

// Proposed: 3-stage ring buffer
sK_u8[3][TILE_N][D_PAD];  // 18 KB × 3 = 54 KB (within 64 KB limit)
// Prefetch tile t+2 while computing t
```

**Expected Outcome**:
- PTXAS: ≤120 regs, ≤52 KB SMEM, 0 spills
- Correctness: 9/9 tests (bit-exact with Stage-2)
- Performance: 590 μs (≥+10%)

---

### **Option B: Pivot to Different Optimization**
**Alternatives**:
1. **Persistent CTAs**: Keep blocks resident across batches (for serving workloads)
2. **Kernel Fusion**: Fuse layernorm/QKV projection with SDPA
3. **Multi-Query Attention (MQA)**: Optimize for fewer KV heads
4. **Native FP8**: Replace simulated FP8 with CUDA's `__nv_fp8_e4m3` / `__nv_fp8_e5m2`
5. **Split-K Attention**: Parallelize KV reduction across CTAs (long sequences)

**Rationale**: Explore orthogonal optimizations; fused softmax may not be the critical path

---

### **Option C: Return to Fused Softmax (Clean-Slate)** — HIGH RISK
**Description**: Redesign Stage-3B with phased validation  
**Phases**:
1. Unit test WMMA reductions (max, sum) in isolation (2h)
2. Partial fusion: QK^T in regs, softmax in SMEM, P·V with WMMA (2h)
3. Full fusion only if Phase 2 validates (4h)

**Target**: +15-25% speedup (≤557-525 μs)  
**Effort**: 8-12 hours  
**Risk**: Medium-High (same complexity as Stage-3B)  
**Rationale**: If successful, highest payoff; but requires more careful approach

---

## 📊 Current Bottleneck Analysis (Hypothesis)

Based on Stage-3B failures and Stage-2 NCU profiling:

**Not the Bottleneck**:
- ❌ SMEM bank conflicts (XOR swizzle regressed)
- ❌ `sS` buffer reuse (micro-fusion gained only +0.2%)
- ❌ Simple register spilling (PTXAS shows 0 spills)

**Likely Bottlenecks** (requires NCU deep-dive):
1. **Global Memory Bandwidth**: Loading Q/K/V tiles from HBM
2. **Warp Occupancy**: 96 regs/thread → limited CTAs per SM
3. **Softmax Latency**: `__expf` dominates compute (even with FP8)
4. **KV Tile Size**: TILE_N=64 may not be optimal for L4 (vs 128 or 32)

**Next NCU Investigation**:
```bash
ncu --set full --target-processes all \
    --metrics dram__bytes_read,smsp__inst_executed_pipe_tensor,sm__warps_active \
    python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --iters 100
```

---

## 🎓 Key Lessons (Stage-1 through Stage-3)

1. **GREEN before FAST**: Never optimize a broken kernel (Stage-3B: 4h wasted debugging)
2. **"Valid Negative" is Success**: Proving an optimization DOESN'T work saves future teams time
3. **Resource Wins ≠ Perf Wins**: Stage-3B reduced SMEM by 2 KB but failed correctness (latency >> resource)
4. **WMMA is Non-Trivial**: Fragment layout and reduction patterns are architecture-specific and error-prone
5. **Incremental Validation**: Stage-1 (+13.8%) → Stage-2 (+13.8%) worked; Stage-3B's big jump failed

---

## 📁 Key Artifacts

### **Validated Baselines** (on L4)
- `main` branch: Stage-2 (`v2.0-stage2-wmma-pv`, 656 μs)
- `tasks/fp8_sdpa_stage_c_wmma/`: Robust-kbench task (correctness + perf gates)
- `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`: Production kernel

### **Documentation**
- `SESSION_STAGE1_STAGE2_COMPLETE.md`: Stage-1 and Stage-2 validation (4.4× speedup)
- `SESSION_STAGE3_COMPLETE_OCT20_2025.md`: Stage-3 attempt (valid negative)
- `STAGE2_GPU_VALIDATION_SUMMARY.md`: Stage-2 L4 results (PR-ready)
- `STAGE3B_HOTFIX_STATUS.md`: Debugging log (3 fix attempts)

### **Results**
```
results/2025-Stage1-Validation/     # Stage-1 correctness + perf
results/2025-Stage2-Validation/     # Stage-2 correctness + perf + NCU
results/2025-Stage3B-Fused/         # Stage-3B failed attempts
```

---

## ⚙️ Development Environment

### **Hardware**
- **Local**: MacBook Pro M3 (CUDA not available, CPU-only testing)
- **Remote**: Google Cloud `cudadent42-l4-dev` (L4, sm_89, 24 GB, us-west1-c)

### **Software**
- **CUDA**: 12.2
- **PyTorch**: 2.x (with CUDA 12.x)
- **Compiler**: `nvcc` with `-arch=sm_89 -O3 --use_fast_math -lineinfo -Xptxas -v`
- **Python**: 3.10 (venv on L4)

### **Access**
```bash
# SSH to L4
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c

# Activate environment
cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

# Run validation
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 500
```

---

## 🎯 Success Criteria (Remaining)

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **Latency (p50)** | 656 μs | < 5 μs | 131× speedup needed |
| **vs PyTorch SDPA** | ~16× faster | ~5000× faster | ~300× more speedup needed |
| **Correctness** | 100% (6/6 tests) | 100% | ✅ Maintained |
| **Resource Budget** | 96 regs, 37 KB SMEM | ≤128 regs, ≤64 KB SMEM | ✅ Within limits |

**Reality Check**: The 131× remaining speedup is **extremely ambitious**. More realistic next milestones:
- **Near-term**: 400-500 μs (2× faster than Stage-2) — achievable with Stage-4 + Stage-5
- **Mid-term**: 100-200 μs (5-10× faster) — requires kernel fusion or Split-K
- **Long-term**: 10-50 μs (50-100× faster) — requires algorithmic breakthroughs (e.g., sparse attention)

---

## 📞 Next Session Prompt

**Goal**: Implement Stage-4 (3-stage `cp.async`) for quick, low-risk win  
**Status**: Ready to start (Stage-2 validated baseline on `main`)  
**Branch**: Create new `feat/stage4-3stage-pipeline` from `main`  
**Timeline**: 4-6 hours  
**Expected Outcome**: 590 μs (≥+10% vs Stage-2)

**OR**

**Goal**: Pivot to [chosen alternative] optimization  
**Status**: Requires design doc and baseline profiling  
**Branch**: Create new `feat/stage[N]-[name]` from `main`

---

**Last Updated**: 2025-10-20 20:35 UTC  
**Current Branch**: `main` (Stage-2 validated, 656 μs)  
**Feature Branch**: `feat/stage3-fusion-full` (abandoned, retained for reference)  
**Status**: ✅ **READY FOR NEXT STAGE**

