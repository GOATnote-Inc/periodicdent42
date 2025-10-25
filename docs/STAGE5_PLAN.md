# Stage-5: Warp Specialization + Persistent CTAs — Implementation Plan

**Date**: October 20-21, 2025  
**Branch**: `feat/stage5-warp-spec-persistent`  
**Target**: ≥15× vs PyTorch SDPA, +10% vs Stage-2 (656 μs)  
**Strategy**: Compute-side acceleration + EvoEngineer-Full search

---

## 🎯 Motivation

**Stage-4 Result**: 3-stage cp.async → +0.7% speedup (far below target)  
**Conclusion**: Kernel is **compute-bound**, NOT memory-bound  
**Next Wins**: Overlap load/compute at warp level + amortize overhead

---

## 📋 What's Changing

### 1. **Warp Specialization** (`USE_WARP_SPECIALIZATION=1`)
- **Producer warps** (warp_id < `NUM_PRODUCER_WARPS`):
  - Issue `cp.async` for K/V tiles
  - Perform u8→half dequantization
  - Write to shared memory
- **Consumer warps** (warp_id >= `NUM_PRODUCER_WARPS`):
  - Compute Q@K^T, softmax, P·V
  - WMMA matrix multiplications
- **Synchronization**: Lightweight flags + `__threadfence_block()` instead of full `__syncthreads()`

### 2. **Persistent CTAs** (`USE_PERSISTENT_CTA=1`)
- Each CTA processes multiple `q_block` indices via atomic work queue
- Amortizes:
  - Q tile loading
  - Stats initialization (`m_smem`, `l_smem`, `U_smem`)
  - Launch overhead

### 3. **Fast Exp Approximation** (`USE_FAST_EXP=1`)
- 5th-order polynomial approximation for softmax exponentials
- Replaces `__expf` with `fast_expf`
- Trade-off: ~1e-3 relative error for 2-3× speedup in exp operations
- **Default OFF** to protect correctness gate

### 4. **Robust Benchmarking**
- 100-run medians (p50/p90/p99), 20 warmup
- Modular evaluation: compile → correctness → performance
- PyTorch SDPA baseline comparison
- JSON output for reproducibility

### 5. **EvoEngineer-Full Autotune**
- Elite preservation (K=3) over configuration grid
- Knobs: `USE_WARP_SPECIALIZATION`, `NUM_PRODUCER_WARPS`, `USE_PERSISTENT_CTA`, `USE_FAST_EXP`, tile sizes
- Re-checks correctness after each mutation
- Retains top-3 configs by p50 latency

---

## ✅ Acceptance Gates (HARD FAIL if violated)

### Gate 1: PTXAS
- **Registers**: ≤120 per thread
- **SMEM**: ≤64 KB per CTA
- **Spills**: 0 bytes

**Why**: Resource constraints ensure occupancy; spills kill performance

### Gate 2: Correctness
- **Metric**: `max_err ≤ 0.06` on small/mission/long shapes
- **Method**: 5 random inputs per shape, compare vs PyTorch SDPA (FP16 reference)
- **Tolerance**: FP8-appropriate (0.06 accounts for quantization noise)

**Why**: No optimization at the cost of correctness ("GREEN before FAST")

### Gate 3: Performance (Mission Shape)
- **Target 1**: ≥**15× vs PyTorch SDPA** (mission shape, B=2, H=8, S=512, D=64)
- **Target 2**: ≥**+10% vs Stage-2** (p50 ≤ 590 μs, Stage-2 baseline = 656 μs)

**Why**: Meaningful speedup over both PyTorch and previous best

### Gate 4: NCU Profiling Sanity
- **Tensor Core utilization**: `sm__pipe_tensor_cycles_active ≥ 50%` **OR**
- **DRAM throughput**: `dram__throughput < 50%` peak

**Why**: Confirms compute-bound hypothesis; if DRAM throughput high, revisit memory pipeline

---

## 🗂 File Structure

```
feat/stage5-warp-spec-persistent/
├── cudadent42/bench/kernels/
│   └── sdpa_fp8_stage_c_wmma.cu        # Kernel with WS + Persistent CTA logic
├── scripts/
│   ├── bench_sdpa.py                    # Robust benchmarking (100-run medians)
│   └── ncu_sdpa.sh                      # One-click NCU profiling
├── kbench/
│   ├── autotune_evo_full.py            # EvoEngineer-Full search (elite K=3)
│   ├── results_stage5.json             # Benchmark output
│   └── elite.json                       # Top-3 configs from autotune
├── docs/
│   ├── STAGE5_PLAN.md                   # This file
│   ├── ROBUST_KBENCH.md                 # Benchmarking methodology
│   └── EVOLUTION_NOTES.md               # EvoEngineer design notes
└── tools/ncu/                           # NCU reports
```

---

## 🔨 Implementation Checklist

### Phase 1: Foundational Changes ✅
- [x] Add Stage-5 toggles (`USE_WARP_SPECIALIZATION`, etc.)
- [x] Add sync helpers (`stage_store_release`, `stage_spin_acquire`)
- [x] Add `fast_expf` function (guarded by `USE_FAST_EXP`)
- [x] Create benchmarking script (`scripts/bench_sdpa.py`)
- [x] Create NCU profiling script (`scripts/ncu_sdpa.sh`)
- [x] Create documentation (`docs/STAGE5_PLAN.md`, etc.)

### Phase 2: Kernel Modifications (IN PROGRESS)
- [ ] Add warp role detection (`is_producer` flag)
- [ ] Add producer/consumer handshake flags (`kv_ready`, `kv_consumed`)
- [ ] Implement producer warp logic (async load + dequant)
- [ ] Implement consumer warp logic (compute Q@K^T, softmax, P·V)
- [ ] Replace `__syncthreads()` with lightweight flags in WS path
- [ ] Add persistent CTA work queue (atomic `q_block` allocation)
- [ ] Replace `__expf` with `fast_expf` in softmax (when `USE_FAST_EXP=1`)

### Phase 3: Validation on L4
- [ ] Build Stage-2 control (record PTXAS baseline)
- [ ] Build Stage-5 with WS (record PTXAS + compare)
- [ ] Run correctness tests (compare Stage-2 vs Stage-5)
- [ ] Run performance benchmarks (mission shape, 100 iters)
- [ ] Run NCU profiling (confirm compute-bound)
- [ ] Check all gates (PTXAS, correctness, performance, NCU)

### Phase 4: EvoEngineer-Full Search
- [ ] Create autotune script (`kbench/autotune_evo_full.py`)
- [ ] Define configuration grid (WS, producers, persistent, fast-exp, tiles)
- [ ] Run search (elite K=3, ~50-100 configs)
- [ ] Select best config (lowest p50, passes correctness)
- [ ] Re-validate with 500-iter benchmark

### Phase 5: Documentation & Merge
- [ ] Update `STATUS_CURRENT.md` with Stage-5 results
- [ ] Create session summary (`SESSION_STAGE5_COMPLETE.md`)
- [ ] Merge to `main` if all gates pass
- [ ] Tag `v3.0-stage5-warp-spec` (if successful)

---

## 🧪 Testing Protocol

### Local (Mac, CPU-only)
```bash
# Compile check (won't run kernels)
python -m tasks.fp8_sdpa_stage_c_wmma.build
```

### Remote (L4 GPU)
```bash
# SSH to L4
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c

# Setup
cd ~/periodicdent42
git fetch -p && git checkout feat/stage5-warp-spec-persistent && git pull
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

# Gate 1: PTXAS (Stage-2 baseline)
USE_CP_ASYNC=1 USE_WMMA_PV=1 python -m tasks.fp8_sdpa_stage_c_wmma.build 2>&1 | grep -E "Used|spill|smem"

# Gate 1: PTXAS (Stage-5 with WS)
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_WARP_SPECIALIZATION=1 NUM_PRODUCER_WARPS=1 \
python -m tasks.fp8_sdpa_stage_c_wmma.build 2>&1 | grep -E "Used|spill|smem"

# Gate 2+3: Correctness + Performance
python scripts/bench_sdpa.py --iters 100 --warmup 20 --shapes small,mission,long

# Gate 4: NCU Profiling
bash scripts/ncu_sdpa.sh

# EvoEngineer-Full Search
python kbench/autotune_evo_full.py
```

---

## 📊 Expected Results

### Optimistic (All optimizations work)
```
Mission Shape (B=2, H=8, S=512, D=64):
  Stage-2:             656 μs  (baseline)
  Stage-5 (WS only):   ~590 μs  (+10% ✅, but needs WS kernel impl)
  Stage-5 (WS+Persist):~550 μs  (+19%)
  Stage-5 (WS+P+Fast): ~500 μs  (+31%, if fast-exp passes correctness)
  
vs PyTorch SDPA:       ~25× faster  (meets ≥15× gate ✅)
```

### Realistic (Conservative estimate)
```
Mission Shape:
  Stage-2:             656 μs
  Stage-5 (WS):        ~600 μs  (+9%, close to +10% gate)
  Stage-5 (WS+Persist):~580 μs  (+13%)
  
vs PyTorch SDPA:       ~18× faster  (meets ≥15× gate ✅)
```

### Pessimistic (WS overhead dominates)
```
Mission Shape:
  Stage-2:             656 μs
  Stage-5 (WS):        ~670 μs  (-2%, regression due to sync overhead)
  
Result: FAIL performance gate → document as valid negative
```

---

## 🚨 Risk Mitigation

### Risk 1: WS Synchronization Overhead
**Symptom**: Stage-5 slower than Stage-2  
**Mitigation**: Profile with NCU, check for barrier stalls → switch to `cuda::barrier` if needed  
**Fallback**: Document as valid negative, revert to Stage-2

### Risk 2: Persistent CTA Correctness Issues
**Symptom**: Correctness failures when `USE_PERSISTENT_CTA=1`  
**Mitigation**: Add extensive debug prints, validate per-CTA state  
**Fallback**: Disable persistent CTAs (`USE_PERSISTENT_CTA=0`), keep WS only

### Risk 3: Fast Exp Accuracy Loss
**Symptom**: `max_err > 0.06` when `USE_FAST_EXP=1`  
**Mitigation**: Keep `USE_FAST_EXP=0` by default, only enable if autotune validates  
**Fallback**: Never use fast-exp in production builds

### Risk 4: Register Spilling
**Symptom**: PTXAS shows spills when WS enabled  
**Mitigation**: Reduce `NUM_PRODUCER_WARPS` to 1, unroll loops less aggressively  
**Fallback**: Reject config if spills > 0 bytes

---

## 🎓 Key Design Principles

1. **"GREEN before FAST"**: All toggles default OFF; enable only after validation
2. **Modular Evaluation**: Compile → Correctness → Performance (explicit gates)
3. **Reproducibility**: 100-run medians, JSON logs, git SHAs captured
4. **EvoEngineer Alignment**: Elite preservation, two-layer traverse, task context
5. **Fail Fast**: If any gate fails, document and pivot (valid negatives are valuable)

---

## 📖 References

- **EvoEngineer Paper** (arXiv:2510.03760v1): Elite preservation, two-layer traverse (Table 3, Sec. 4.1-4.3)
- **Stage-4 Valid Negative**: 3-stage cp.async → +0.7% (proved memory not bottleneck)
- **FlashAttention-2**: Warp specialization for producer/consumer overlap
- **CUTLASS**: Multi-stage pipelining, persistent kernels

---

**Status**: Phase 1 COMPLETE ✅, Phase 2 IN PROGRESS  
**Next Action**: Implement warp specialization kernel logic (see `docs/WS_IMPLEMENTATION_GUIDE.md`)  
**ETA**: 6-8 hours for full implementation + validation

