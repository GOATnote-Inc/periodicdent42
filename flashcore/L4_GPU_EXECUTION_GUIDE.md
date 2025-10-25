# L4 GPU Execution Guide

**Status**: ⚠️ Currently on local Mac - Must SSH to L4 GPU instance for execution  
**Date**: October 21, 2025

---

## 🚨 Important: We're Currently on Local Mac

The preflight checks revealed we're running on a **local Mac environment**, not the L4 GPU instance. To execute FlashCore:

### Option 1: SSH to L4 Instance (Recommended)

```bash
# SSH to your GCP L4 instance
gcloud compute ssh <instance-name> --zone=<zone>

# Once on L4, navigate to project
cd /path/to/flashcore  # or clone from git

# Run preflight
bash scripts/preflight.sh

# If preflight passes, proceed with Phase 1
```

### Option 2: Clone to L4 and Start Fresh

```bash
# On L4 instance
git clone <flashcore-repo-url>
cd flashcore

# Install dependencies
pip install -r requirements.txt

# Run preflight
bash scripts/preflight.sh

# Build baseline
python build.py

# Run tests
pytest tests/test_correctness.py -v

# Benchmark
python benchmarks/benchmark_latency.py --shape mission --iters 100
```

---

## 📋 Pre-Flight Check Results (Local Mac)

| Check | Status | Notes |
|-------|--------|-------|
| nvidia-smi | ❌ Not found | No NVIDIA GPU on Mac |
| nvcc | ❌ Not found | CUDA toolkit not installed on Mac |
| ncu | ❌ Not found | Nsight Compute not on Mac |
| PyTorch CUDA | ❌ Sandbox blocked | Python library access restricted |
| git | ❌ Sandbox blocked | Library dependencies blocked |

**Conclusion**: Cannot execute GPU work on local Mac. Must use L4 instance.

---

## ✅ What's Ready for L4 Execution

### Phase 0 Infrastructure (Complete)

All files are ready to transfer to L4:

```
flashcore/
├── kernels/
│   ├── flashcore_baseline.cu     ✅ Ready (from fa_minimal.cu)
│   └── bindings.cpp               ✅ Ready
│
├── tests/
│   └── test_correctness.py       ✅ Ready (15 test cases)
│
├── benchmarks/
│   └── benchmark_latency.py      ✅ Ready (100-run medians)
│
├── scripts/
│   ├── env_cuda_l4.sh            ✅ Ready (CUDA environment setup)
│   ├── preflight.sh              ✅ Ready (GPU validation)
│   └── keepalive.sh              ✅ Ready (tmux + nvidia-smi dmon)
│
├── docs/
│   ├── ARCHITECTURE.md           ✅ Ready
│   ├── GETTING_STARTED.md        ✅ Ready
│   └── PHASE1_WMMA_GUIDE.md      ✅ Ready
│
├── build.py                       ✅ Ready
├── requirements.txt               ✅ Ready
└── AGENTS.md                      ✅ Ready
```

---

## 🚀 Execution Plan on L4

### Step 1: Transfer Files to L4

**Option A: Git (Recommended)**
```bash
# On local Mac - commit and push
cd /Users/kiteboard/periodicdent42/flashcore
git add .
git commit -m "feat: FlashCore Phase 0 complete - ready for L4 validation"
git push origin main

# On L4 - pull
cd /path/to/flashcore
git pull
```

**Option B: SCP**
```bash
# From local Mac
cd /Users/kiteboard/periodicdent42
tar czf flashcore.tar.gz flashcore/
gcloud compute scp flashcore.tar.gz <instance>:~ --zone=<zone>

# On L4
tar xzf flashcore.tar.gz
cd flashcore
```

### Step 2: Setup Environment on L4

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify CUDA
source scripts/env_cuda_l4.sh

# Run preflight (should pass on L4)
bash scripts/preflight.sh
```

**Expected Output**:
```
==============================================================================
FlashCore GPU Preflight Check
==============================================================================

== GPU Hardware ==
NVIDIA L4, 535.xx.xx, 24576 MiB

== CUDA Toolkit ==
release 12.2, V12.2.xxx
[OK] nvcc found: /usr/local/cuda-12.2/bin/nvcc

== Nsight Compute ==
[OK] ncu found: /usr/local/cuda-12.2/bin/ncu

== PyTorch CUDA ==
PyTorch: 2.x.x
CUDA Version: 12.1
CUDA Available: True
Device: NVIDIA L4
Compute Capability: 8.9
[OK] L4 GPU detected (sm_89)

==============================================================================
Preflight Summary
==============================================================================
Warnings: 0
Errors: 0
[PASS] All preflight checks passed ✅
```

### Step 3: Validate Baseline (2 hours)

```bash
# Build baseline kernel
python build.py

# Run correctness tests (15 cases)
pytest tests/test_correctness.py -v

# Benchmark performance
python benchmarks/benchmark_latency.py --shape mission --iters 100 --out results/baseline.json

# View results
cat results/baseline.json | jq '.results.mission'
```

**Expected Results**:
```json
{
  "config": {"B": 1, "H": 8, "S": 512, "D": 64},
  "flashcore": {
    "p50": 1500.0,  // ~1500 µs (baseline)
    "p90": 1520.0,
    "p99": 1550.0
  },
  "pytorch": {
    "p50": 25.9     // PyTorch SDPA reference
  },
  "speedup": 0.017  // 58× slower (expected for scalar baseline)
}
```

### Step 4: Begin Phase 1 (WMMA) (30-40 hours)

Follow `docs/PHASE1_WMMA_GUIDE.md`:

```bash
# 1. Create WMMA kernel from baseline
cp kernels/flashcore_baseline.cu kernels/flashcore_wmma.cu

# 2. Edit flashcore_wmma.cu (add WMMA for Q@K^T and P@V)
# ... (see PHASE1_WMMA_GUIDE.md for detailed steps)

# 3. Build WMMA kernel
python build.py  # update to build flashcore_wmma

# 4. Test incrementally
pytest tests/test_correctness.py::test_correctness[tiny-0] -v     # Start with tiny
pytest tests/test_correctness.py::test_correctness[small-0] -v    # Then small
pytest tests/test_correctness.py::test_correctness[mission-0] -v  # Then mission
pytest tests/test_correctness.py -v                               # All 15 tests

# 5. Benchmark
python benchmarks/benchmark_latency.py --shape mission --iters 100 --out results/wmma.json

# 6. Profile with NCU
ncu --set full --launch-skip 10 --launch-count 1 \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    python benchmarks/benchmark_latency.py --shape mission --iters 1 \
    > logs/ncu_wmma.txt

# 7. Check Tensor Core utilization
grep "sm__pipe_tensor_cycles_active" logs/ncu_wmma.txt
# Target: >50%
```

---

## 🎯 Success Criteria Recap

### Phase 0 (Baseline Validation)
- ✅ All 15 tests pass (max_err ≤ 0.06)
- ✅ Baseline latency measured (~1500 µs)
- ✅ Infrastructure working (build, test, bench)

### Phase 1 (WMMA)
- ✅ PTXAS: ≤120 regs, ≤64 KB SMEM, 0 spills
- ✅ Correctness: All 15 tests pass
- ✅ Performance: <200 µs (≥7× vs baseline)
- ✅ NCU: TC utilization ≥50%

### Phase 2 (Fusion) - **PRIMARY GOAL**
- ✅ Performance: <58 µs (≥15× vs 870 µs)
- ✅ Correctness: All 15 tests pass
- ✅ NCU: DRAM throughput >50% of peak

---

## 📊 Estimated Timeline on L4

| Activity | Time | Cumulative |
|----------|------|------------|
| Transfer files + setup | 0.5h | 0.5h |
| Baseline validation | 1.5h | 2h |
| Phase 1 WMMA implementation | 30h | 32h |
| Phase 1 testing + profiling | 8h | 40h |
| **Phase 1 Complete** | | **40h** |
| Phase 2 Fusion implementation | 30h | 70h |
| Phase 2 testing + optimization | 10h | 80h |
| **Phase 2 Complete (Goal)** | | **80h** |

**Critical Path**: Phase 2 completion = **PROJECT SUCCESS** ✅

---

## 🔧 Troubleshooting on L4

### Issue: Preflight Fails

**Symptoms**: `scripts/preflight.sh` exits with errors

**Fixes**:
```bash
# Fix CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Re-source environment
source scripts/env_cuda_l4.sh

# Re-run preflight
bash scripts/preflight.sh
```

### Issue: Build Fails

**Symptoms**: `python build.py` errors

**Common Fixes**:
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check nvcc
which nvcc
nvcc --version

# Try verbose build
VERBOSE=1 python build.py
```

### Issue: Tests Fail (Correctness)

**Symptoms**: `pytest` shows failures, high max_err

**Debug**:
```bash
# Run single test with output
pytest tests/test_correctness.py::test_correctness[tiny-0] -v -s

# Check for NaN/Inf
# ... (kernel will print debug info if DEBUG=1)

# Enable debug build
DEBUG=1 python build.py
pytest tests/test_correctness.py::test_correctness[tiny-0] -v -s
```

### Issue: Performance Regression

**Symptoms**: WMMA slower than baseline

**Debug**:
```bash
# Profile with NCU
ncu --set full python benchmarks/benchmark_latency.py --shape mission --iters 1

# Check metrics
grep "sm__pipe_tensor_cycles_active" <ncu_output>
grep "dram__throughput" <ncu_output>

# Compare against baseline
python benchmarks/benchmark_latency.py --all --out results/comparison.json
```

---

## 📞 Next Steps

1. **Transfer FlashCore to L4 instance** (git push/pull or scp)
2. **Run preflight on L4** (`bash scripts/preflight.sh`)
3. **Validate baseline** (2 hours: build + test + bench)
4. **Begin Phase 1 WMMA** (30-40 hours, follow PHASE1_WMMA_GUIDE.md)

---

## ✅ Readiness Checklist

- [ ] L4 GPU instance accessible (SSH or GCP console)
- [ ] FlashCore files transferred to L4
- [ ] Python environment set up (`pip install -r requirements.txt`)
- [ ] Preflight passes on L4 (`bash scripts/preflight.sh`)
- [ ] Baseline validated (tests + benchmark)
- [ ] Ready to begin Phase 1 WMMA

---

**Status**: Infrastructure complete, ready for L4 execution 🚀  
**Current Location**: Local Mac (development)  
**Execution Location**: L4 GPU instance (validation + optimization)  
**Next Action**: SSH to L4, run preflight, validate baseline

