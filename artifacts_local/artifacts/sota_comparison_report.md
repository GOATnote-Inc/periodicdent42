# SOTA Attention Benchmark Comparison

**GPU**: NVIDIA L4 (23.7 GB)
**Date**: 2025-10-13 18:23:47
**Iterations**: 100 (warmup: 20)
**Statistical Method**: Bootstrap 95% CI (N=1000 resamples)

---

## Executive Summary

Tested **4 configurations** across **1 implementations**:

- PyTorch SDPA

## Results by Configuration

### Baseline (B=32, H=8, S=512, D=64)

| Implementation | Latency (ms) | 95% CI | Bandwidth (GB/s) | Memory (MB) |
|----------------|--------------|--------|------------------|-------------|
| PyTorch SDPA | **0.3251** | [0.3226, 0.3400] | 201.7 | 84 |

**Winner**: PyTorch SDPA (0.3251 ms)

### Large (B=16, H=16, S=1024, D=64)

| Implementation | Latency (ms) | 95% CI | Bandwidth (GB/s) | Memory (MB) |
|----------------|--------------|--------|------------------|-------------|
| PyTorch SDPA | **1.3235** | [1.3199, 1.3365] | 102.0 | 169 |

**Winner**: PyTorch SDPA (1.3235 ms)

### Optimized (B=32, H=8, S=128, D=64)

| Implementation | Latency (ms) | 95% CI | Bandwidth (GB/s) | Memory (MB) |
|----------------|--------------|--------|------------------|-------------|
| PyTorch SDPA | **0.0512** | [nan, nan] | 317.1 | 21 |

**Winner**: PyTorch SDPA (0.0512 ms)

### Small (B=4, H=8, S=256, D=64)

| Implementation | Latency (ms) | 95% CI | Bandwidth (GB/s) | Memory (MB) |
|----------------|--------------|--------|------------------|-------------|
| PyTorch SDPA | **0.0492** | [nan, nan] | 83.8 | 5 |

**Winner**: PyTorch SDPA (0.0492 ms)

## Statistical Analysis

All results use **bootstrap 95% confidence intervals** (N=1000 resamples).
Implementations with **non-overlapping CIs** are statistically significantly different.

## Reproducibility

### Environment
- GPU: NVIDIA L4
- PyTorch: 2.2.1+cu121
- CUDA: 12.1

### Run Command
```bash
python sota_comparison.py --iterations 100 --warmup 20
```

## Raw Data

```json
[
  {
    "implementation": "PyTorch SDPA",
    "config_name": "Optimized (B=32, H=8, S=128, D=64)",
    "median_ms": 0.05119999870657921,
    "mean_ms": 0.052909119091928004,
    "std_ms": 0.004750771439652374,
    "ci_95_low": NaN,
    "ci_95_high": NaN,
    "throughput_gflops": 20294.07864709306,
    "bandwidth_gb_s": 317.0949788608291,
    "memory_mb": 21.102592,
    "iterations": 100,
    "error": null
  },
  {
    "implementation": "PyTorch SDPA",
    "config_name": "Baseline (B=32, H=8, S=512, D=64)",
    "median_ms": 0.3251200020313263,
    "mean_ms": 0.33265631943941115,
    "std_ms": 0.019424419400598754,
    "ci_95_low": 0.32256001234054565,
    "ci_95_high": 0.3399679958820343,
    "throughput_gflops": 51644.49968349115,
    "bandwidth_gb_s": 201.7363268886373,
    "memory_mb": 84.410368,
    "iterations": 100,
    "error": null
  },
  {
    "implementation": "PyTorch SDPA",
    "config_name": "Small (B=4, H=8, S=256, D=64)",
    "median_ms": 0.04915200173854828,
    "mean_ms": 0.05006336089223623,
    "std_ms": 0.0035291468007477822,
    "ci_95_low": NaN,
    "ci_95_high": NaN,
    "throughput_gflops": 10723.828812764692,
    "bandwidth_gb_s": 83.77991259972416,
    "memory_mb": 5.275648,
    "iterations": 100,
    "error": null
  },
  {
    "implementation": "PyTorch SDPA",
    "config_name": "Large (B=16, H=16, S=1024, D=64)",
    "median_ms": 1.3235199451446533,
    "mean_ms": 1.3155328190326692,
    "std_ms": 0.05449787746009363,
    "ci_95_low": 1.3199360370635986,
    "ci_95_high": 1.3365080042528663,
    "throughput_gflops": 52236.99153817421,
    "bandwidth_gb_s": 102.0253740979965,
    "memory_mb": 168.820736,
    "iterations": 100,
    "error": null
  }
]
```
