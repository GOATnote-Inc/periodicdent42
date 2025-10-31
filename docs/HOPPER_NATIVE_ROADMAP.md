# Hopper-Native FlashAttention Roadmap (FA3-Style)

**Status**: Phase 1 - Skeleton Implementation  
**Target**: Beat FA3 on H100 (DRAM ≥85%, SM ≥70%, 1.5-2× over FA2)  
**Strategy**: Standing on giants' shoulders (PyTorch FA3, CUTLASS 3.x, NVIDIA TMA/WGMMA)

---

## 🎯 Measurable Targets

### 1. **Bandwidth Roofline**
- **Target**: ≥85-90% of theoretical DRAM bandwidth on memory-bound phases
- **Metric**: `dram__throughput.avg.pct_of_peak_sustained_elapsed` (NSight Compute)
- **Baseline**: FA3 achieves 85-90% on H100

### 2. **Compute Overlap**
- **Target**: Keep Tensor Core lanes busy via warp specialization + WGMMA
- **Metric**: `sm__throughput.avg.pct_of_peak_sustained_elapsed` ≥70%
- **Baseline**: FA3 achieves 70-80% SM utilization

### 3. **Long-Context Scaling**
- **Target**: Flat or gently sloped throughput out to 16K-32K tokens
- **Metric**: Tokens/sec at 8K, 16K, 32K (no "context collapse")
- **Baseline**: FA3 shows flat scaling with TMA-fed pipelines

### 4. **Safety**
- **Target**: Deterministic toggles, sanitizer-clean, alignment-correct
- **Validation**: `compute-sanitizer` (memcheck, racecheck, synccheck) all pass
- **API**: Shape/dtype/stride validation before launch

---

## 🏗️ Implementation Phases

### **Phase 1: Skeleton (Current)**
**Goal**: Compile, run, validate correctness

- [x] Warp specialization structure (loader vs compute warps)
- [x] Double-buffered shared memory for K/V
- [x] Online softmax in registers (FA2/FA3 algorithm)
- [x] Barrier-based synchronization (mbarrier)
- [x] Scalar fallback for GEMMs (TODO: upgrade to WGMMA)
- [ ] **Validation**: Correctness vs PyTorch SDPA (max_diff < 2e-3)

**Expected**: Compiles, runs, correct output (performance TBD)

### **Phase 2: TMA Integration**
**Goal**: Async memory copy with high bandwidth

- [ ] TMA descriptors for K/V tiles (2D/3D tensor maps)
- [ ] `cp.async.bulk.tensor` for GMEM → SMEM transfers
- [ ] Double buffering with mbarrier arrive/wait
- [ ] Multi-CTA cluster multicast (shared K/V blocks)
- [ ] **Validation**: DRAM throughput ≥85% (NSight Compute SOL)

**Expected**: 2-3× speedup from memory bandwidth alone

### **Phase 3: WGMMA Integration**
**Goal**: Hopper tensor cores for Q·K^T and P·V

- [ ] WGMMA instructions (`wgmma.mma_async`) for FP16/BF16
- [ ] Multiple GMMA ops in flight per warp-group
- [ ] Fence only as required (minimize barriers)
- [ ] **Validation**: SM throughput ≥70%, compute-bound phases

**Expected**: 3-5× speedup from Tensor Core utilization

### **Phase 4: 2-Stage Pipeline (FA3 Trick)**
**Goal**: Interleave WGMMA ↔ softmax to hide softmax latency

- [ ] Producer-consumer pipeline (WGMMA in stage N, softmax in stage N-1)
- [ ] Warp-level synchronization (lightweight flags, not global barriers)
- [ ] Overlap epilogue stores with next tile loads
- [ ] **Validation**: Warp stall analysis (NCU), reduced idle time

**Expected**: 1.3-1.5× speedup from latency hiding

### **Phase 5: Auto-Tuner**
**Goal**: Optimal tiling per head_dim and sequence length

- [ ] Search space: (BM, BN, BK, NUM_STAGES, WARPS_LOADER:WARPS_COMPUTE)
- [ ] Constraints: SMEM ≤227KB, alignment (128-byte for WGMMA)
- [ ] Heuristics: hdim=64 → 128×256×128, hdim=128 → 64×256×128, hdim=256 → 64×128×64
- [ ] **Validation**: Auto-tune beats hand-tuned on diverse shapes

**Expected**: 1.1-1.2× final speedup from optimal configs

### **Phase 6: Long Context (16K-32K)**
**Goal**: Flat scaling without "context collapse"

- [ ] Dynamic re-tiling (reduce BN, increase stage depth as S grows)
- [ ] Selective recompute (instead of spilling intermediates)
- [ ] Cluster multicast for K (amortize loads across CTAs)
- [ ] **Validation**: Tokens/sec flat from 2K → 32K

**Expected**: Enable production use at long contexts

### **Phase 7: Production Hardening**
**Goal**: CI gates, determinism, safety

- [ ] Compute-sanitizer in CI (gate merges on clean runs)
- [ ] Host-side API with bounds/dtype/stride validation
- [ ] Determinism mode (optional flag, slight perf tax)
- [ ] Alignment checks (catch WGMMA footguns early)
- [ ] **Validation**: CI green, deterministic mode passes

**Expected**: Production-ready, merge to main

---

## 📊 Validation Workflow

### **1. Correctness** (every commit)
```bash
./build_cuda_simple.sh
# Output: max_diff < 2e-3 vs PyTorch SDPA
```

### **2. Performance** (after each phase)
```bash
./build_cuda_simple.sh
# Output: Median latency, TFLOPS, comparison to FA3
```

### **3. NSight Compute** (Phase 2+)
```bash
./tools/ncu_validate.sh ./build/bin/test_hopper
# Output: DRAM ≥85%, SM ≥70%, roofline plots
```

### **4. Compute-Sanitizer** (every phase)
```bash
RUN_SANITIZER=1 ./tools/run_debug_profile.sh
# Output: memcheck, racecheck, synccheck all PASS
```

### **5. FA3 Benchmark** (Phase 7)
```bash
python flashcore/benchmark/fa3_comparison.py
# Output: Match/beat FA3 on (a) memory throughput (b) tokens/s @8-32K (c) numerical stability
```

---

## 🔬 Key Metrics Per Phase

| Phase | DRAM (%) | SM (%) | Latency (μs) | TFLOPS | vs FA3 | Status |
|-------|----------|--------|--------------|--------|--------|--------|
| **Baseline (PyTorch SDPA)** | ~60% | ~50% | 25.94 | ~130 | 0.5× | ✅ Measured |
| **Phase 1: Skeleton** | TBD | TBD | TBD | TBD | TBD | 🚧 In Progress |
| **Phase 2: TMA** | ≥85% | ~50% | ~10 | ~260 | 1.0× | ⏳ Pending |
| **Phase 3: WGMMA** | ≥85% | ≥70% | ~5 | ~520 | 1.5× | ⏳ Pending |
| **Phase 4: Pipeline** | ≥85% | ≥75% | ~3 | ~870 | 2.0× | ⏳ Pending |
| **Phase 5-7: Tuned** | ≥90% | ≥80% | < 3 | ≥870 | **≥2.0×** | 🎯 Target |

---

## 📚 References (Standing on Shoulders)

### **Primary Sources**
1. **FA3 Paper** (arXiv:2510.03760v1): Warp specialization, 2-stage pipeline, TMA integration
2. **PyTorch FA3** (GitHub): Reference implementation, benchmarks, tile configs
3. **NVIDIA TMA Docs**: Tensor Memory Accelerator programming guide
4. **NVIDIA WGMMA Docs**: Warp-group matrix multiply instructions
5. **CUTLASS 3.x**: SM90 GEMM kernels, alignment requirements, tiling heuristics

### **Validation Standards**
- **NSight Compute**: SOL metrics, roofline analysis, memory/compute workload
- **Compute-Sanitizer**: memcheck, racecheck, synccheck (NVIDIA validation toolkit)
- **FA3 Baseline**: 85-90% DRAM, 70-80% SM, 1.5-2× over FA2

---

## 🚀 Quick Start

### **Build & Test**
```bash
# Compile Hopper-native kernel (requires H100, sm_90a)
./build_cuda_simple.sh

# Output:
# - Correctness: max_diff vs SDPA
# - Performance: Median latency, TFLOPS
```

### **Profile (NSight Compute)**
```bash
# Quick SOL check (DRAM, SM throughput)
./tools/ncu_validate.sh ./build/bin/test_hopper

# Open UI for detailed analysis
ncu-ui build/ncu_reports/sol.ncu-rep
```

### **Validate (Sanitizer)**
```bash
# Run memory/race/sync checks
RUN_SANITIZER=1 ./tools/run_debug_profile.sh
```

---

## 🎓 Lessons from WMMA Pivot

**Why we pivoted from WMMA → Hopper-native**:
1. WMMA (sm_70-80 API) has stack footprint issues on sm_90
2. WGMMA (sm_90 API) is purpose-built for Hopper, more efficient
3. TMA is the "right way" for async memory on H100
4. FA3 paper proves this path achieves SOTA performance

**Standing on giants**: Use NVIDIA's battle-tested TMA/WGMMA APIs instead of fighting with WMMA stack limits.

---

**Last Updated**: Oct 27, 2025  
**Next Milestone**: Phase 1 correctness validation

