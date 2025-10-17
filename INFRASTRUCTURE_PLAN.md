# Infrastructure Plan - Pragmatic Approach
**Date**: Oct 17, 2025  
**Mission**: Build comparison & profiling infrastructure, not reinvent TC from scratch  
**Status**: 🟢 **IN PROGRESS**

---

## Philosophy Shift

### Previous Approach ❌
- Try to implement full TC from scratch (WMMA/CUTLASS)
- 4 hours → both approaches blocked
- High complexity, uncertain outcome

### New Approach ✅
- **Build on Phase 4** (working, 1028 μs, correct)
- **Add infrastructure** for profiling & comparison
- **Use CUTLASS as reference** baseline, not main implementation
- **Systematic optimization** with hardware feedback
- **EvoEngineer-guided** with proper seeding

---

## Components

### A) Nsight Compute Profiling ✅ COMPLETE
**Status**: Installed, version 2023.2.0  
**Path**: `/usr/local/cuda/bin/ncu`

**Usage**:
```bash
# Brief metrics (fast, <10s overhead):
ncu --target-processes all --replay-mode kernel \
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv -o evidence/ncu_profile.csv \
  python bench/run_kernel.py
```

**Metrics**:
- `sm__warps_active`: Warp occupancy (target: >60%)
- `sm__pipe_tensor_active`: Tensor Core utilization
- `dram__throughput`: Memory bandwidth usage

---

### B) Microbench Infrastructure 🟡 IN PROGRESS
**Goal**: Fast, clock64()-based ranking of tile configurations

**Files to create**:
1. `bench/micro/bench_many.cu` - Synthetic tile kernels
2. `bench/micro/build_micro.sh` - Compile script
3. `bench/micro/run_micro.py` - Rank & output Top-K

**Output**: `evidence/micro_best.json` (Top-8 configs)

**Integration**: Seed EvoEngineer Generation 0 with Top-K

---

### C) CUTLASS Minimal Baseline ⏸️ PENDING
**Goal**: Working GEMM for comparison, NOT full implementation

**Approach**:
- Single-file test: `bench/cutlass/cutlass_gemm_sm89.cu`
- Use Sm80 config (works on Sm89/Ada)
- Measure as reference baseline only
- **Not integrated into main kernel** (comparison tool)

**Expected**:
- Compiles ✅
- Runs QK^T GEMM correctly ✅
- Provides performance target

---

### D) Phase 4 Light-Barrier Path ✅ EXISTS
**Status**: Already implemented in `fa_phase3_wmma.cu`

**Enable**:
```bash
SYNC_POLICY=2 REDUCE_WARP=1 VEC_LOAD=1 NUM_WARPS=4 BLOCK_M=32
```

**Current**: 1028 μs (proven correct)

---

### E) EvoEngineer Seeding 🟡 IN PROGRESS
**Goal**: Intelligent Generation 0 from microbench

**Flow**:
1. Microbench ranks 20-30 tile configs → Top-8
2. EvoEngineer Generation 0 starts from Top-8
3. Evolves with measured fitness (speedup vs SDPA)

**Already exists**: `bench/evo/sweep.py` (needs seeding hook)

---

### F) Profiling Scripts ⏸️ PENDING
**Scripts to add**:
1. `scripts/profile_ncu.sh` - Wrapper for ncu with standard metrics
2. `scripts/compare_backends.py` - A/B test custom vs CUTLASS vs cuBLAS
3. `scripts/collect_evidence.sh` - Run full benchmark suite + profiling

---

## Implementation Order

### Phase 1: Microbench (1 hour) 🟡 CURRENT
- [x] Verify ncu installed
- [ ] Create `bench/micro/bench_many.cu`
- [ ] Create `bench/micro/build_micro.sh`
- [ ] Create `bench/micro/run_micro.py`
- [ ] Test: generates `evidence/micro_best.json`

### Phase 2: EvoEngineer Seeding (30 mins)
- [ ] Modify `bench/evo/sweep.py` to read `micro_best.json`
- [ ] Seed Generation 0 with Top-K configs
- [ ] Test: sweep starts from good configs

### Phase 3: CUTLASS Baseline (1 hour)
- [ ] Create `bench/cutlass/cutlass_gemm_sm89.cu`
- [ ] Compile & test (standalone)
- [ ] Benchmark QK^T GEMM performance
- [ ] Document as reference baseline

### Phase 4: Profiling Infrastructure (30 mins)
- [ ] Create `scripts/profile_ncu.sh`
- [ ] Create `scripts/compare_backends.py`
- [ ] Run on Phase 4 kernel
- [ ] Collect metrics in `evidence/`

### Phase 5: Integrated Run (30 mins)
- [ ] Full EvoEngineer sweep with seeding
- [ ] Profile best candidates with ncu
- [ ] Compare vs CUTLASS baseline
- [ ] Document findings

**Total**: ~3.5 hours for complete infrastructure

---

## Success Criteria

### Immediate (Phase 1-2)
- ✅ Nsight Compute installed & working
- ✅ Microbench produces Top-8 configs
- ✅ EvoEngineer starts from intelligent seeds
- ✅ Faster convergence to good configs

### Short-term (Phase 3-4)
- ✅ CUTLASS baseline compiles & runs
- ✅ Profiling shows bottlenecks (DRAM vs compute vs sync)
- ✅ Evidence collected in standardized format
- ✅ Can compare multiple approaches objectively

### Long-term (Phase 5)
- ✅ EvoEngineer finds better configs than manual tuning
- ✅ Hardware counters guide optimization priorities
- ✅ Reproducible, documented methodology
- ✅ Portfolio-ready optimization process

---

## What This Gives Us

### Technical Value
1. **Systematic optimization**: Hardware feedback loop
2. **Reproducible process**: EvoEngineer + profiling
3. **Objective comparison**: vs CUTLASS, cuBLAS baselines
4. **Portfolio piece**: Demonstrates ML+systems engineering

### Learning Value
1. **Profiling methodology**: Nsight Compute workflow
2. **EvoEngineer framework**: LLM-guided optimization
3. **Hardware insights**: What matters on L4/Ada
4. **Pragmatic engineering**: When to use libraries vs custom

### Realistic Outcomes
- **Phase 4**: 1028 μs (current, correct) ✅
- **With seeding**: 800-900 μs (better configs) - likely
- **vs CUTLASS**: Know gap to production TC - valuable
- **Portfolio**: Shows systematic approach - impressive

---

## Key Insight

**Don't need to beat CUTLASS** - showing systematic optimization process with:
- Hardware profiling (Nsight)
- Intelligent search (EvoEngineer)
- Objective comparison (vs production baselines)
- Documented methodology

...is MORE impressive than a half-working TC implementation.

---

**Status**: Nsight ✅ | Microbench 🟡 | CUTLASS ⏸️ | Profiling ⏸️  
**Next**: Create microbench infrastructure (Phase 1)  
**Timeline**: 3.5 hours for complete infrastructure

