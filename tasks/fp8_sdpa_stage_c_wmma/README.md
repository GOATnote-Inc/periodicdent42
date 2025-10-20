# FP8 SDPA Stage-C WMMA — Robust-kbench Task

**Correctness-gated validation and reproducible performance baseline for FP8 SDPA kernel**

---

## 📋 **Overview**

This task provides systematic validation for the FP8 SDPA Stage-C WMMA kernel optimized for NVIDIA L4 (Ada, sm_89). It implements:

- **Correctness Gates**: Hard failure on numerical errors exceeding thresholds
- **Performance Baseline**: Reproducible latency measurements with CUDA events
- **Build Toggles**: Control USE_KV_LUT and DEBUG_PRINT at compile time
- **Metadata Capture**: Git SHA, device info, PTXAS stats for reproducibility

---

## 🚀 **Quick Start**

### **Run Correctness Validation**

```bash
# Test small and mission shapes with 3 seeds each
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2
```

**Expected Output**:
```
[small   ] seed=0: max_err=0.0136, mean_err=0.0028, %bad=0.0% ✅ PASS
[small   ] seed=1: max_err=0.0142, mean_err=0.0029, %bad=0.0% ✅ PASS
[small   ] seed=2: max_err=0.0138, mean_err=0.0027, %bad=0.0% ✅ PASS
[mission ] seed=0: max_err=0.0100, mean_err=0.0010, %bad=0.0% ✅ PASS
[mission ] seed=1: max_err=0.0098, mean_err=0.0009, %bad=0.0% ✅ PASS
[mission ] seed=2: max_err=0.0097, mean_err=0.0010, %bad=0.0% ✅ PASS

✅ ALL CORRECTNESS CHECKS PASSED!
```

### **Run Performance Baseline**

```bash
# Mission shape with 500 timed iterations
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 500
```

**Expected Output**:
```
[mission ] seed=0: p50=XXX.XXμs, p90=XXX.XXμs, std=X.XXμs
```

---

## 📊 **Correctness Gates**

All three gates must pass for each (shape, seed) pair:

| Gate | Threshold | Description |
|------|-----------|-------------|
| **Gate 1** | `max_abs_err ≤ 0.06` | Maximum absolute error vs PyTorch SDPA (FP8-tuned) |
| **Gate 2** | `mean_abs_err ≤ 0.02` | Mean absolute error (strict) |
| **Gate 3** | `% bad ≤ 1.0%` | Percentage of elements with `|err| > 0.06` |

**Failure Handling**: If any gate fails, the runner exits with code 1 and prints gate details.

---

## 🔧 **Build Toggles**

Control kernel compilation via environment variables:

### **USE_KV_LUT** (default: `0`)

```bash
# Direct dequant (default, safe, correct) ✓
USE_KV_LUT=0 python -m tasks.fp8_sdpa_stage_c_wmma.runner ...

# LUT path (experimental, requires debugging)
USE_KV_LUT=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner ...
```

### **DEBUG_PRINT** (default: `0`)

```bash
# Quiet build (production) ✓
DEBUG_PRINT=0 python -m tasks.fp8_sdpa_stage_c_wmma.runner ...

# Verbose debug prints (development)
DEBUG_PRINT=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner ...
```

### **USE_CP_ASYNC** (default: `1`)

```bash
# cp.async double-buffering (Stage 1 optimization) ✓
USE_CP_ASYNC=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner ...

# Disable for baseline comparison
USE_CP_ASYNC=0 python -m tasks.fp8_sdpa_stage_c_wmma.runner ...
```

### **TORCH_CUDA_ARCH_LIST** (default: `8.9`)

```bash
# Override for different architectures
TORCH_CUDA_ARCH_LIST="8.0" python -m tasks.fp8_sdpa_stage_c_wmma.runner ...
```

---

## 📁 **Output Structure**

Results are saved to `results/fp8_wmma_baseline/<timestamp>/`:

```
results/fp8_wmma_baseline/20251020-143022/
├── build_meta.json             # Git SHA, device, compile flags, PTXAS stats
├── correctness_summary.json    # Per-shape/seed correctness metrics
└── perf_baseline.json          # Per-shape/seed latency (p50/p90/std)
```

### **build_meta.json**

```json
{
  "timestamp": "2025-10-20T14:30:22",
  "build": {
    "USE_KV_LUT": 0,
    "DEBUG_PRINT": 0,
    "arch": "sm_89",
    "flags": ["-O3", "--use_fast_math", "-lineinfo"]
  },
  "git": {
    "sha": "1e272a1f",
    "branch": "main",
    "dirty": false
  },
  "device": {
    "name": "NVIDIA L4",
    "cuda_version": "12.8",
    "sm_version": "8.9"
  }
}
```

### **correctness_summary.json**

```json
{
  "results": [
    {
      "shape": "mission",
      "B": 1, "H": 8, "S": 512, "D": 64,
      "seed": 0,
      "max_abs_err": 0.0100,
      "mean_abs_err": 0.0010,
      "max_rel_err": 0.1234,
      "pct_bad": 0.0,
      "gates": {
        "max_abs_err_pass": true,
        "mean_abs_err_pass": true,
        "pct_bad_pass": true
      },
      "pass": true
    }
  ],
  "all_pass": true
}
```

### **perf_baseline.json**

```json
{
  "results": [
    {
      "shape": "mission",
      "B": 1, "H": 8, "S": 512, "D": 64,
      "seed": 0,
      "warmup_iters": 100,
      "timed_iters": 500,
      "p50_us": 245.32,
      "p90_us": 248.17,
      "mean_us": 245.89,
      "std_us": 1.23,
      "iters_per_sec": 4066.89
    }
  ]
}
```

---

## 🧪 **Shapes Configuration**

Shapes are defined in `config_forward.json`:

| Name | B | H | S | D | Purpose |
|------|---|---|-----|---|---------|
| **small** | 1 | 1 | 32 | 64 | Quick correctness check |
| **mission** | 1 | 8 | 512 | 64 | Primary optimization target |
| **long** | 1 | 8 | 2048 | 64 | Scaling validation |

Add custom shapes by editing `config_forward.json`.

---

## 🔬 **Advanced Usage**

### **Skip Rebuild (Use Cached Extension)**

```bash
# Useful for rapid iteration
python -m tasks.fp8_sdpa_stage_c_wmma.runner --no-build --shapes mission --seeds 0
```

### **Single Shape + Performance**

```bash
# Correctness + performance for mission shape
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0,1,2 --iters 500
```

### **Integrate with CI/CD**

```bash
# Exit code 0 = all pass, 1 = any fail
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2
if [ $? -ne 0 ]; then
  echo "❌ Correctness gates failed!"
  exit 1
fi
```

---

## 📚 **Module Structure**

```
tasks/fp8_sdpa_stage_c_wmma/
├── __init__.py              # Package metadata
├── README.md                # This file
├── config_forward.json      # Shapes, seeds, tolerances
├── build.py                 # Extension builder with toggles
├── func_forward.py          # forward_ref, forward_kernel, validate
└── runner.py                # CLI entry point
```

---

## 🎯 **Next Steps**

After correctness is locked in:

1. **NCU Profiling**: `scripts/profile_ncu.sh mission`
2. **Optimization Loop**: See `docs/PERF_PLAN.md` for staged perf improvements
3. **Regression Testing**: Add to CI with `--shapes small --seeds 0 --iters 10`

---

## 📖 **References**

- **Kernel**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`
- **Bindings**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma_bindings.cpp`
- **EvoEngineer Methodology**: "GREEN before FAST" gating strategy

---

**Status**: ✅ Correctness validated (small + mission shapes, seeds 0-2)  
**Baseline**: TBD (run with `--iters 500` to establish)

