# CUDA Performance Feedback Loop System

**Status**: Production-ready iterative optimization framework  
**Author**: Brandon Dent (b@thegoatnote.com)  
**Date**: October 13, 2025

---

## TL;DR

Your CI now **iteratively improves CUDA performance** via three closed-loop hooks:

1. **Performance Ratcheting**: Tracks best-known results, fails on regression, updates baseline on improvement
2. **Auto-Profiling**: Runs Nsight on regressions/large gains, provides root cause evidence
3. **Auto-Tuning** (manual dispatch): Searches parameter space, suggests better kernel settings

**Result**: Every PR either maintains or improves performance. The baseline continuously ratchets forward.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PR Commit                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Build + Benchmark                                  │
│  → Compile kernel                                           │
│  → Run bench/integrated_test.py                            │
│  → Output: results/current.json                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Performance Ratchet                                │
│  → Compare current vs baseline (best-ever per config)      │
│  → Decision:                                                │
│    • Regression (>3% slower) → FAIL + profile              │
│    • Improvement (>5% faster) → Update baseline + profile  │
│    • Unchanged → PASS                                       │
│  → Output: results/ratchet_report.md                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Auto-Profile (conditional)                         │
│  → If regression or large improvement:                      │
│    • Run Nsight Compute                                     │
│    • Generate .ncu-rep + PNG one-pager                     │
│    • Attach to PR artifacts                                 │
│  → Output: results/nsight/*.ncu-rep                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Comment PR                                         │
│  → Post ratchet report as PR comment                        │
│  → Include:                                                 │
│    • Summary (wins/losses/unchanged)                        │
│    • Regression details (if any)                            │
│    • Link to profiling artifacts                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Update Baseline (on main push)                    │
│  → If improvements found:                                   │
│    • Commit updated baseline.json                           │
│    • Push with [skip ci] tag                                │
│  → Baseline continuously ratchets forward                   │
└─────────────────────────────────────────────────────────────┘
```

### Manual Optimization Loop (Weekly/On-Demand)

```
┌─────────────────────────────────────────────────────────────┐
│  Manual Dispatch: Autotune Workflow                         │
│  → Time budget: 20 minutes (configurable)                   │
│  → Search parameter space:                                  │
│    • BLOCK_M, BLOCK_N (tile sizes)                         │
│    • NUM_STAGES (pipelining)                                │
│    • Other compile-time params                              │
│  → Output: tuning/suggestions.md                           │
│  → Human reviews and applies best params                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage

### 1. Performance Ratcheting (Automatic on Every PR)

**Workflow**: `.github/workflows/cuda_benchmark_ratchet.yml`

**Triggers**:
- On PR touching `cudadent42/**/*.cu`, `bench/**`, `setup.py`
- On push to `main`
- Manual dispatch

**What it does**:
1. Builds kernel
2. Runs benchmark
3. Compares to baseline (best-known per config)
4. Fails if regression >3%
5. Updates baseline if improvement >5%
6. Auto-profiles if regression or large improvement
7. Comments results on PR

**Example PR comment**:
```markdown
## 📊 Performance Ratchet Report

**Commit**: abc123d  
**Hardware**: L4 GPU  

### Summary
- **Total configs**: 1
- **Regressions**: 0 ❌
- **Improvements**: 1 ✅
- **Unchanged**: 0

### ✅ Improvements (Baseline Updated)
| Config | Baseline | Current | Change |
|--------|----------|---------|--------|
| training_512 | 0.0612 ms | 0.0530 ms | **+13.4%** |

📊 Configs to profile: training_512

---
*Automated by CUDA Performance Ratchet*
```

### 2. Auto-Profiling (Automatic on Performance Changes)

**When triggered**:
- Regression detected (>3% slower)
- Large improvement detected (>5% faster)

**What it does**:
- Runs `ncu --set full` on affected configs
- Generates `.ncu-rep` file (open with Nsight UI)
- Uploads as artifact to workflow run

**How to view**:
1. Go to Actions tab → Click workflow run
2. Scroll to bottom → Download "benchmark-ratchet-results"
3. Extract and open `.ncu-rep` files with `nv-nsight-cu-cli` or Nsight UI

**What to look for**:
- Memory coalescing efficiency (<80% → problem)
- SM utilization (<60% → occupancy issue)
- Stall reasons (memory throttle → bandwidth-bound)
- L2 cache hit rate (<80% → locality issue)

### 3. Auto-Tuning (Manual Dispatch)

**When to run**: Weekly or after major kernel changes

**How to run**:
```bash
# Via GitHub Actions (recommended)
# 1. Go to Actions tab
# 2. Select "CUDA Autotune" workflow
# 3. Click "Run workflow"
# 4. Enter config name (e.g., "training_512")
# 5. Enter time budget (default: 20 minutes)
# 6. Click "Run"

# Or locally
cd cudadent42/bench
python3 autotune.py --config training_512 --time-budget 20
```

**What it does**:
1. Tests all combinations of tunable parameters
2. Benchmarks each combination (3 iterations)
3. Ranks by speedup vs baseline
4. Generates `tuning/suggestions.md` with best params

**Example output** (`tuning/suggestions.md`):
```markdown
# Autotune Report: training_512

## Best Configuration Found
**Speedup**: 1.23×
**Latency**: 0.0431 ms (baseline: 0.0530 ms)
**Parameters**:
- `BLOCK_M = 128`
- `BLOCK_N = 128`
- `NUM_STAGES = 2`

## How to Apply
### Option 1: Compile-time flags
```cmake
target_compile_definitions(kernel PRIVATE BLOCK_M=128 BLOCK_N=128 NUM_STAGES=2)
```

### Option 2: nvcc flags
```bash
nvcc -DBLOCK_M=128 -DBLOCK_N=128 -DNUM_STAGES=2 flash_attention.cu
```
```

**Human decision**: Review report, test params, apply if beneficial.

---

## Configuration

### Ratchet Thresholds

Edit in `.github/workflows/cuda_benchmark_ratchet.yml`:

```yaml
env:
  REGRESSION_THRESHOLD: -3.0  # Fail if >3% slower
  IMPROVEMENT_THRESHOLD: 5.0  # Profile if >5% faster
```

**Tuning guidance**:
- `-3.0`: Strict (catches small regressions, may have false positives)
- `-5.0`: Relaxed (only catches significant regressions)
- `+5.0`: Standard (profiles meaningful improvements)
- `+10.0`: Conservative (only profiles large gains)

### Baseline Location

```yaml
env:
  BASELINE_PATH: cudadent42/bench/results/baseline.json
```

**Structure**:
```json
{
  "configs": {
    "training_512": {
      "median_ms": 0.0530,
      "ci_95_low": 0.0512,
      "ci_95_high": 0.0548,
      "throughput_gflops": 20.3,
      "bandwidth_gb_s": 311.9,
      "git_commit": "abc123d",
      "timestamp": "2025-10-13T15:30:00Z"
    }
  }
}
```

### Autotuning Parameters

Edit in `bench/autotune.py`:

```python
def _get_tunable_params(self) -> List[TuneParam]:
    return [
        TuneParam(
            name="BLOCK_M",
            values=[64, 128, 256],  # Add more values
            current_default=64
        ),
        TuneParam(
            name="BLOCK_N",
            values=[64, 128, 256],
            current_default=64
        ),
        TuneParam(
            name="NUM_STAGES",
            values=[1, 2, 3, 4],  # More pipeline stages
            current_default=1
        ),
        # Add new parameters:
        TuneParam(
            name="NUM_WARPS",
            values=[4, 8, 16],
            current_default=4
        ),
    ]
```

---

## Workflow Files

### 1. `cuda_benchmark_ratchet.yml` (Main CI)

**Purpose**: Automatic ratcheting on every PR

**Key steps**:
- Lock GPU clocks (if supported)
- Build kernel
- Run benchmark
- Compare to baseline
- Auto-profile on changes
- Comment PR with results
- Update baseline on main push

### 2. `cuda_autotune.yml` (Manual Dispatch)

**Purpose**: Periodic parameter search

**Key steps**:
- Run autotune script with time budget
- Generate suggestions report
- Upload as artifact
- Optionally open draft PR with suggestions

---

## Best Practices

### 1. Baseline Hygiene

**DO**:
- ✅ Commit baseline updates separately (`perf: Ratchet baseline forward`)
- ✅ Include commit SHA in baseline for audit trail
- ✅ Review baseline changes in PRs (should only improve or stay same)

**DON'T**:
- ❌ Manually edit baseline.json
- ❌ Reset baseline to hide regressions
- ❌ Commit baseline changes with functional changes (separate commits)

### 2. Regression Triage

When regression detected:

1. **Check profiling artifacts**:
   - Download `.ncu-rep` from workflow run
   - Look for memory coalescing, occupancy, stall reasons
   
2. **Reproduce locally**:
   ```bash
   cd cudadent42
   python3 bench/integrated_test.py --config training_512
   ncu --set full python3 bench/integrated_test.py --config training_512
   ```

3. **Root cause analysis**:
   - Memory: Check coalescing efficiency, L2 hit rate
   - Compute: Check SM utilization, tensor core usage
   - Occupancy: Check register pressure, shared memory usage

4. **Fix or accept**:
   - If fixable: Apply optimization, verify with benchmark
   - If unavoidable (e.g., correctness fix): Document in PR, update baseline after merge

### 3. Autotuning Workflow

**Frequency**: Weekly or after major changes

**Process**:
1. Dispatch autotune workflow (20 min budget)
2. Download suggestions report
3. Review speedup vs build complexity trade-off
4. Test suggested params on multiple configs
5. If beneficial: Apply via CMake or nvcc flags
6. Verify with full benchmark suite
7. Open PR with params change

**Diminishing returns**: If autotune finds <5% improvement, probably not worth applying (variance noise).

---

## Troubleshooting

### Ratchet script fails with "No results found"

**Cause**: Benchmark output format mismatch

**Fix**: Check `results/current.json` exists and has expected format:
```json
{
  "config": {...},
  "performance": {
    "mean_time_ms": 0.053,
    ...
  }
}
```

### Baseline keeps getting worse instead of better

**Cause**: Regression threshold too loose or baseline corruption

**Fix**:
1. Check recent commits for performance regressions
2. Re-run baseline: `rm results/baseline.json && run workflow`
3. Tighten regression threshold to `-2.0` or `-1.0`

### Auto-profiling artifacts empty

**Cause**: Nsight Compute not installed or insufficient permissions

**Fix**:
```bash
# Check ncu available
which ncu

# Install Nsight Compute
sudo apt-get install nvidia-nsight-compute

# Run with sudo if needed (CI)
sudo ncu --set full ...
```

### Autotune finds no improvements

**Cause**: Parameter space exhausted or kernel already optimal

**Actions**:
- Expand parameter search space (more values, new params)
- Profile manually to find new optimization angles
- Consider algorithmic changes (e.g., block-sparse, kernel fusion)

---

## Future Enhancements

### Phase 1: Bootstrap CIs (Q4 2025)

Add statistical rigor:
```python
from scipy.stats import bootstrap

def compute_bootstrap_ci(samples, confidence_level=0.95):
    res = bootstrap((samples,), np.median, n_resamples=10000, 
                   confidence_level=confidence_level)
    return res.confidence_interval
```

### Phase 2: Cliff's Delta Effect Sizes (Q4 2025)

Add effect size reporting:
```python
def compute_cliffs_delta(baseline_samples, treatment_samples):
    n1, n2 = len(baseline_samples), len(treatment_samples)
    dominance = sum(t < b for t in treatment_samples for b in baseline_samples)
    delta = (dominance - (n1 * n2 / 2)) / (n1 * n2)
    return delta  # -1 to +1
```

### Phase 3: Multi-Config Benchmarking (Q1 2026)

Test matrix of configs:
```yaml
strategy:
  matrix:
    config: [training_512, training_1k, inference_2k, inference_4k]
```

### Phase 4: Bayesian Optimization (Q1 2026)

Replace exhaustive search with Bayesian optimizer:
```python
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f=benchmark_with_params,
    pbounds={
        'BLOCK_M': (64, 256),
        'BLOCK_N': (64, 256),
        'NUM_STAGES': (1, 4)
    }
)
```

---

## Summary

**What you have now**:
- ✅ Automatic regression detection
- ✅ Continuous baseline ratcheting
- ✅ Auto-profiling on performance changes
- ✅ Manual parameter optimization workflow

**What this enables**:
- Never ship performance regressions
- Baseline steadily improves over time
- Evidence-based optimization (profiling + tuning)
- Portfolio signal: systematic performance engineering

**Cost**: ~$0.007 per PR (~2 min GPU time)  
**Value**: Prevents regressions, accelerates optimization, demonstrates rigor

---

## Contact

**Brandon Dent**  
**Email**: b@thegoatnote.com  
**LinkedIn**: [linkedin.com/in/brandon-dent-84aba2130](https://linkedin.com/in/brandon-dent-84aba2130)  
**GitHub**: [github.com/GOATnote-Inc/periodicdent42](https://github.com/GOATnote-Inc/periodicdent42)

**For Periodic Labs**: This feedback loop system demonstrates systematic performance engineering - the foundation for iterative kernel optimization at scale.

