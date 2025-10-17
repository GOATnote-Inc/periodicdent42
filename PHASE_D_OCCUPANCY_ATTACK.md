# Phase D: Occupancy Attack - Register Pressure Reduction

**Date**: Oct 17, 2025  
**Mission**: Fix low occupancy (9.28% → 33%+) by attacking register pressure  
**Current**: xFormers @ 33.19 μs, 9.28% occupancy, 0.27 eligible warps/sched  
**Target**: < 5 μs with 33%+ occupancy, 2+ eligible warps/sched

---

## **🎯 Strategy: Stand on Giants + Lift Eligible Warps**

### **Phase D.1: Import Best-in-Class Baselines (1-2 hours)**

**FlashAttention-2** (Dao-AILab):
- ✅ Proven on sm_89 (Ada/L4)
- ✅ Battle-tested occupancy tuning
- ✅ Will use as reference implementation

**CUTLASS FMHA** (NVIDIA):
- ✅ Example 77 FMHA kernels
- ✅ Autotuning hooks built-in
- ✅ Official NVIDIA support for Ada

**Integration**:
```bash
git submodule add https://github.com/Dao-AILab/flash-attention third_party/flash-attention
git submodule add https://github.com/NVIDIA/cutlass third_party/cutlass
```

---

### **Phase D.2: Register Pressure Triage (2-3 hours)**

**Root Cause** (from NCU):
```
Theoretical Occupancy: 33.33% (limited by REGISTERS)
Achieved Occupancy: 9.28%
Block Limit (Registers): 4 blocks ← BOTTLENECK
```

**Attack Plan**:

1. **Cap registers**: `-maxrregcount={64,72,80,88,96}`
2. **Smaller CTAs**: `BLOCK_THREADS={128,192,256}`
3. **Move temporaries to SMEM**: Big per-thread arrays → shared memory
4. **De-inline helpers**: Drop `__forceinline__` where it bloats regs
5. **Add `__restrict__`**: Reduce aliasing pressure
6. **Prune unrolls**: Replace `#pragma unroll` with bounded loops
7. **Two-stage softmax**: Scope per-row stats carefully

**Nsight Acceptance Gates**:
- ✅ Eligible warps/sched ≥ 2
- ✅ Issue-slot util ≥ 60%
- ✅ Theoretical occ ≥ 33%
- ✅ Achieved occ ≥ 20%

---

### **Phase D.3: Systematic Parameter Sweep (2-3 hours)**

**Dimensions**:
- `REGCAP`: {64, 72, 80, 88, 96}
- `THREADS`: {128, 192, 256}
- `TILE_M/N`: {32, 64}
- `K_TILE`: {16, 32, 64}

**Expected Combinations**: 5 × 3 × 2 × 3 = 90 configs

**Automation**: `tools/sweep_occupancy.sh`

---

### **Phase D.4: EvoEngineer Integration (1-2 hours)**

**Solution-Guiding Layer**:
- Historical bests: fa2, cutlass, custom_v3
- NCU insights: Register pressure, low eligible warps
- Mutation space: REGCAP, block size, tile dims

**Validation**: KernelBench + robust-kbench (anti-reward-hacking)

---

## **📋 Implementation Checklist**

### **Step 1: Submodules & Build Infrastructure**
- [ ] Add FlashAttention-2 submodule
- [ ] Add CUTLASS submodule
- [ ] Update build system for sm_89
- [ ] Add REGCAP environment variable
- [ ] Add launch bounds toggle
- [ ] Test FA-2 build
- [ ] Test CUTLASS FMHA build

### **Step 2: Baseline Measurements**
- [ ] Benchmark FA-2 on L4
- [ ] Benchmark CUTLASS FMHA on L4
- [ ] Document as new baselines
- [ ] Verify correctness (all ≤ 2e-3)

### **Step 3: Register Pressure Fixes**
- [ ] Add `-maxrregcount` to build
- [ ] Move large temporaries to SMEM
- [ ] Add `__restrict__` to pointers
- [ ] De-inline heavy helpers
- [ ] Prune aggressive unrolls
- [ ] Add launch bounds
- [ ] Test single config (REGCAP=80, THREADS=192)

### **Step 4: Occupancy Sweep**
- [ ] Create `tools/sweep_occupancy.sh`
- [ ] Run 90-config sweep
- [ ] Log NCU metrics for each
- [ ] Identify Pareto frontier
- [ ] Select best config

### **Step 5: Validation**
- [ ] Run NCU on best config
- [ ] Verify: eligible warps ≥ 2
- [ ] Verify: issue-slot util ≥ 60%
- [ ] Verify: achieved occ ≥ 20%
- [ ] Benchmark latency
- [ ] Verify correctness

---

## **⏱️ Timeline**

```
Phase D.1 (Submodules):       1-2 hours
Phase D.2 (Register Fixes):   2-3 hours
Phase D.3 (Sweep):            2-3 hours
Phase D.4 (EvoEngineer):      1-2 hours
──────────────────────────────────────
Total: 6-10 hours
```

---

## **🎯 Success Criteria**

**Minimum (Tier 1)**:
- ✅ Achieved occupancy ≥ 20%
- ✅ Eligible warps ≥ 2
- ✅ Latency < 33.19 μs (beat xFormers)
- ✅ Correctness: max_diff ≤ 2e-3

**Target (Tier 2)**:
- ✅ Achieved occupancy ≥ 25%
- ✅ Eligible warps ≥ 3
- ✅ Latency < 20 μs (1.65× vs xFormers)
- ✅ Issue-slot util ≥ 60%

**Ambitious (Tier 3)**:
- ✅ Achieved occupancy ≥ 30%
- ✅ Eligible warps ≥ 4
- ✅ Latency < 10 μs (3.3× vs xFormers)
- ✅ Approach FlashAttention-2 performance

**Stretch (Tier 4)**:
- ✅ Achieved occupancy ≥ 33%
- ✅ Latency < 5 μs (6.6× vs xFormers)
- ✅ Beat FlashAttention-2 on L4

---

## **📚 References**

1. [NVIDIA Nsight Compute: Warp Eligibility](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
2. [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
3. [FlashAttention-2 GitHub](https://github.com/Dao-AILab/flash-attention)
4. [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
5. [KernelBench Paper](https://arxiv.org/abs/2502.10517)
6. [NVIDIA Issue Efficiency](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/issueefficiency.htm)
7. [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

---

**Status**: ACTIVE - Phase D.1 starting now  
**Philosophy**: No quitting. Standing on giants AND optimizing.

