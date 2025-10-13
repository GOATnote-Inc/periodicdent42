# CI Integration

## Overview

Automated benchmark on pull requests with regression detection.

## Requirements

- Self-hosted runner with CUDA GPU
- Runner labels: `self-hosted`
- Python 3.10+, torch, numpy

## Setup

```bash
# 1. Create baseline (first time only)
cd cudadent42/bench
python integrated_test.py --output results.json
mkdir -p .baseline
cp results.json .baseline.json

# 2. Commit baseline
git add .baseline.json
git commit -m "Add benchmark baseline"
git push
```

## Usage

### Trigger benchmark

Add label `benchmark` to any PR, or run manually via Actions tab.

### Baseline update

When PR merges to main, baseline automatically updates.

### Regression threshold

Default: -3.0% (configurable in workflow file)

## Files

- `.github/workflows/cuda_benchmark.yml` - Workflow definition
- `cudadent42/bench/integrated_test.py` - Benchmark runner
- `cudadent42/bench/compare_baseline.py` - Regression detector
- `cudadent42/bench/.baseline.json` - Reference results

## Output Structure

```json
{
  "correctness": {
    "passed": true,
    "max_abs_error": 0.000483,
    "mean_abs_error": 0.000062,
    "correlation": 1.0
  },
  "performance": {
    "mean_time_ms": 0.053,
    "std_dev_ms": 0.002,
    "throughput_gflops": 20272,
    "bandwidth_gb_s": 317.0
  },
  "roofline": {
    "arithmetic_intensity": 64.0,
    "bottleneck": "Memory",
    "efficiency_pct": 105.6,
    "recommendation": "..."
  },
  "config": {
    "batch_size": 32,
    "num_heads": 8,
    "seq_len": 128,
    "head_dim": 64,
    "dtype": "float16"
  }
}
```

## Comparison Output

```json
{
  "speedup": 1.0,
  "improvement_pct": 0.2,
  "is_regression": false,
  "baseline_time_ms": 0.053,
  "current_time_ms": 0.053,
  "threshold": -3.0
}
```

## Testing Locally

```bash
# Run benchmark
cd cudadent42/bench
python integrated_test.py --output test_results.json

# Compare to baseline (creates baseline if missing)
python compare_baseline.py test_results.json --threshold -3.0 --output comparison.json

# View results
cat test_results.json | python -m json.tool
cat comparison.json | python -m json.tool
```

## Troubleshooting

### Build fails
Check `setup.py` and CUDA environment.

### Correctness fails
Max error threshold is 1e-3 for FP16. Check kernel implementation.

### Regression detected
Profile with `ncu` to identify bottleneck. Revert if unintentional.

### Baseline missing
Create with `python integrated_test.py --output .baseline.json`

