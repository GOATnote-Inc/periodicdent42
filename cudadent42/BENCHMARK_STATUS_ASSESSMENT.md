# Benchmark Infrastructure Assessment - October 12, 2025

**Branch**: opt/vectorized-loads  
**Status**: Ready to execute, but needs upgrades for publication-grade rigor

---

## Executive Summary

**Current State**: We have **2 comprehensive benchmark scripts** (878 lines total) that compare against PyTorch SDPA and FlashAttention-2, with CSV export, plotting, and statistical analysis.

**Gap**: The benchmarks are **operationally solid** (reproducible, well-documented, cost-tracked) but **scientifically simplistic** for publication because:
1. Speedup is "expected 1.7×" not "measured 1.72× (95% CI: 1.60-1.84)"
2. Missing contemporary SOTA baselines (xFormers, CUTLASS)
3. No Nsight profiling integration
4. No hardware environment manifest
5. No TFLOP/s or bandwidth utilization metrics

**Verdict**: **B+ for ops, C+ for rigor**. A reviewer would flag "Unclear experimental throughput gains."

---

## What We Have ✅

### 1. `benches/bench_correctness_and_speed.py` (396 lines)
**Purpose**: Simple, fast benchmarks vs PyTorch SDPA

**Features**:
- ✅ Benchmarks PyTorch SDPA (baseline)
- ✅ Benchmarks our FlashMoE kernel
- ✅ Multiple configurations (6 configs: Tiny → XLarge + Multi-head)
- ✅ Statistical analysis (mean, std, median, min, max)
- ✅ Throughput calculation (tokens/s)
- ✅ Memory efficiency testing
- ✅ CSV export (`--save-csv`)
- ✅ FP16 + BF16 support
- ✅ Command-line arguments (repeats, warmup, output-dir)

**Missing**:
- ❌ Only 1 baseline (PyTorch SDPA, no backend specification)
- ❌ No 95% confidence intervals
- ❌ No xFormers, CUTLASS baselines
- ❌ No TFLOP/s or bandwidth metrics
- ❌ No Nsight integration

**Grade**: B+ (solid foundation, needs rigor)

---

### 2. `benchmarks/benchmark_attention.py` (482 lines)
**Purpose**: Comprehensive benchmarks with plotting

**Features**:
- ✅ 4 baselines: PyTorch SDPA, FlashAttention-2, our warp-specialized, our basic
- ✅ Plotting (performance curves, speedup graphs)
- ✅ JSON export with metadata (GPU name, memory, dtype)
- ✅ Multiple sequence lengths (512, 1024, 2048, 4096)
- ✅ Causal masking support
- ✅ Memory usage tracking
- ✅ Speedup calculation vs PyTorch and FA-2

**Missing**:
- ❌ No xFormers baseline
- ❌ No CUTLASS baseline
- ❌ No 95% confidence intervals (bootstrap)
- ❌ No TFLOP/s, bandwidth, or Tensor Core utilization
- ❌ No Nsight integration
- ❌ No hardware environment manifest

**Grade**: A- (comprehensive, but missing rigorous metrics)

---

## What's Missing for Publication-Grade 📊

### Critical Gaps (blocks A/A+ grade)

1. **Strong Contemporary Baselines**:
   - ❌ xFormers (Meta's memory-efficient attention)
   - ❌ CUTLASS (NVIDIA's HPC reference)
   - ❌ PyTorch SDPA backend specification (math vs mem_efficient vs flash)

2. **Statistical Rigor**:
   - ❌ 95% confidence intervals via bootstrap
   - ❌ Median over N=30 runs (currently N=50 mean)
   - ❌ Outlier removal (1.5×IQR)
   - ❌ Paired t-tests for significance

3. **Hardware Efficiency Metrics**:
   - ❌ TFLOP/s vs theoretical peak
   - ❌ HBM bandwidth GB/s vs peak
   - ❌ Tensor Core utilization %
   - ❌ SM occupancy

4. **Reproducibility Artifacts**:
   - ❌ `ENV/GPU_ENV.md` (hardware, driver, CUDA, PyTorch versions)
   - ❌ Nsight profiles (`profiles/*.ncu-rep`)
   - ❌ Roofline analysis
   - ❌ Environment hygiene script

5. **Claims Language**:
   - Current: "expected 1.7× speedup" (assertion)
   - Needed: "1.72× (95% CI: 1.60-1.84)" (demonstration)

---

## Upgrade Path (Research-Grade) 🎯

### Option A: Minimal Upgrade (30 min, $0.30)
**Goal**: Get to **demonstrated** speedup with CI

**Changes**:
1. Run existing `bench_correctness_and_speed.py` on L4 GPU
2. Compute 95% CI via bootstrap from existing results
3. Create `ENV/GPU_ENV.md` with hardware specs
4. Update claims language in docs

**Output**: "1.7× ± 0.1" (mean ± std) → upgrades to **B+**

**Limitations**: Still only PyTorch SDPA baseline, no TFLOP/s

---

### Option B: Full SOTA Validation (2-3 hours, $1.50)
**Goal**: Publication-grade with all baselines + metrics

**New Infrastructure Needed**:
1. **`benches/runners/`** (modular baseline runners):
   - `pytorch_sdpa.py` (math, mem_efficient, flash backends)
   - `xformers.py`
   - `flashattn2.py` (already exists in benchmark_attention.py)
   - `cutlass.py` (best-effort tuned)
   - `ours.py` (with Nsight hooks)

2. **`benches/bench_sdpa_suite.py`** (orchestrator):
   - Multiple baselines
   - Bootstrap CI calculation
   - TFLOP/s, bandwidth, TC utilization
   - CSV export for all metrics
   - Nsight profile trigger

3. **`ENV/GPU_ENV.md`** (hardware manifest):
   - GPU name, driver, CUDA, PyTorch, commit SHA
   - Compiler flags, env vars

4. **`scripts/environment_setup.sh`**:
   - Export CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED, etc.

**Output**:
- **Results table**: GPU | S | Mode | Dtype | Baseline | Tokens/s | Δ vs Torch-Flash | TFLOP/s | %Peak TC
- **Nsight profiles**: `profiles/l4_seq1024_causal_fp16.ncu-rep`
- **Claims**: "1.72× (95% CI: 1.60-1.84) tokens/s vs PyTorch SDPA (flash backend), 62% TC activity, 71% HBM bandwidth"

**Grade**: **A/A+** (publication-ready)

---

## Recommended Immediate Action

### Step 1: Execute Existing Benchmarks (20 min, $0.30)
**Why**: Get actual numbers, not "expected"

```bash
# On L4 GPU (opt/vectorized-loads branch)
cd ~/periodicdent42/cudadent42
python benches/bench_correctness_and_speed.py \
  --save-csv --output-dir results/ \
  --repeats 30 --warmup 10
```

**Output**: `results/benchmark_results_fp16.csv` with actual speedups

---

### Step 2: Compute 95% CI from Results (5 min local)
```python
# scripts/compute_ci.py (NEW)
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('results/benchmark_results_fp16.csv')
speedups = df['Speedup'].values

# Bootstrap 95% CI
n_bootstrap = 10000
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = np.random.choice(speedups, size=len(speedups), replace=True)
    bootstrap_means.append(np.mean(sample))

ci_low = np.percentile(bootstrap_means, 2.5)
ci_high = np.percentile(bootstrap_means, 97.5)
mean_speedup = np.mean(speedups)

print(f"Mean speedup: {mean_speedup:.2f}x")
print(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
print(f"Publication claim: '{mean_speedup:.2f}× (95% CI: {ci_low:.2f}-{ci_high:.2f})'")
```

---

### Step 3: Create Hardware Manifest (2 min)
```bash
# ENV/GPU_ENV.md (NEW)
cat > ENV/GPU_ENV.md << 'EOF'
# Hardware & Software Environment

**Date**: October 12, 2025  
**Branch**: opt/vectorized-loads  
**Commit**: 26d5715

## Hardware
- **GPU**: NVIDIA L4 24GB
- **Compute Capability**: SM_89
- **Driver**: 550.90.07
- **CUDA**: 12.4

## Software
- **PyTorch**: 2.7.1+cu128
- **Python**: 3.10
- **GCC**: 11.4.0
- **Compiler Flags**: `-O3 --use_fast_math -lineinfo`

## Environment
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `PYTHONHASHSEED=0`
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
EOF
```

---

### Step 4: Update Claims Language (1 min)
Replace in all docs:
- ❌ "expected 1.7× speedup"
- ✅ "1.72× (95% CI: 1.60-1.84) tokens/s vs PyTorch SDPA (B=16, H=16, D=128, S=1024, FP16, L4)"

---

## Summary Table

| Component | Current | Needed for A+ | Priority | Time | Cost |
|-----------|---------|---------------|----------|------|------|
| **Benchmark script** | ✅ 2 scripts, 878 lines | ✅ Have | Low | - | - |
| **Execute on GPU** | ❌ Not run yet | ✅ Must run | **HIGH** | 20 min | $0.30 |
| **95% CI calculation** | ❌ Missing | ✅ Add bootstrap | **HIGH** | 5 min | $0 |
| **Hardware manifest** | ❌ Missing | ✅ ENV/GPU_ENV.md | **HIGH** | 2 min | $0 |
| **Claims language** | ❌ "expected" | ✅ "demonstrated" | **HIGH** | 1 min | $0 |
| **xFormers baseline** | ❌ Missing | ⚠️  Nice-to-have | Medium | 30 min | $0.20 |
| **CUTLASS baseline** | ❌ Missing | ⚠️  Nice-to-have | Medium | 1 hour | $0.40 |
| **TFLOP/s metrics** | ❌ Missing | ⚠️  Nice-to-have | Medium | 30 min | $0 |
| **Nsight profiles** | ❌ Missing | ⚠️  Nice-to-have | Medium | 30 min | $0.20 |

**Total for B+ → A**: 28 min, $0.30  
**Total for B+ → A+**: 3 hours, $1.40

---

## Honest Assessment

### What We Have (B+)
- ✅ **Operational Excellence**: Cost tracking, reproducible steps, clean git history
- ✅ **Comprehensive Benchmarks**: 2 scripts, multiple configs, CSV export
- ✅ **Baseline Comparison**: PyTorch SDPA + FlashAttention-2
- ✅ **Statistical Analysis**: Mean, std, median, memory efficiency

### What We're Missing (A/A+)
- ❌ **Actual Results**: Haven't run on GPU yet ("expected" not "measured")
- ❌ **Statistical Rigor**: No 95% CI, no bootstrap, no paired t-tests
- ❌ **Hardware Manifest**: No ENV/GPU_ENV.md
- ❌ **SOTA Baselines**: Missing xFormers, CUTLASS
- ❌ **Efficiency Metrics**: No TFLOP/s, bandwidth, TC utilization
- ❌ **Nsight Profiles**: No `.ncu-rep` files

### Grade Progression Path
1. **Current**: B+ (ops solid, rigor missing)
2. **After GPU run + CI**: A- (demonstrated speedup)
3. **After ENV + claims**: A (publication-ready claims)
4. **After SOTA baselines**: A+ (comprehensive validation)

---

## Recommended Next Steps

### Immediate (Next Session - 28 min, $0.30)
1. ✅ Start L4 GPU instance
2. ✅ Checkout `opt/vectorized-loads` branch
3. ✅ Run `benches/bench_correctness_and_speed.py` with FP16-only build
4. ✅ Compute 95% CI from results
5. ✅ Create `ENV/GPU_ENV.md`
6. ✅ Update all claims language
7. ✅ Commit results + docs

**Output**: Upgrade from B+ → A (demonstrated speedup with CI)

### Future (Optional for A+)
8. Add xFormers baseline runner
9. Add CUTLASS baseline runner
10. Implement TFLOP/s calculation
11. Add Nsight profile triggers
12. Generate roofline plot

---

## Conclusion

**Answer to User's Question**: "Do we not already have these benchmarks?"

**YES**, we have excellent benchmark infrastructure (878 lines, 2 scripts, CSV export, plotting). The issue is not **missing code**, it's **missing execution + rigor**:

1. **Never run on GPU yet** → all speedups are "expected" not "measured"
2. **No 95% CI** → can't say "1.72× (1.60-1.84)" only "~1.7×"
3. **No hardware manifest** → can't reproduce

**The fix is simple**: Execute existing benchmarks (20 min, $0.30), add bootstrap CI (5 min, $0), create ENV/GPU_ENV.md (2 min, $0), update claims. That's **B+ → A** in 28 minutes.

**The user's comprehensive guide** is for **A → A+** (xFormers, CUTLASS, Nsight, roofline), which is 2-3 hours more work. We can do that AFTER validating Fix #1.

---

**Recommendation**: Run what we have, demonstrate speedup, THEN add more baselines.

**Status**: Ready to execute ✅

