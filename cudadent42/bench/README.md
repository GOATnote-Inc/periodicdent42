# CUDA Kernel Benchmark Suite

Objective, reproducible benchmarking infrastructure for FlashAttention kernels on L4 GPU.

## Quick Start

```bash
# Run PyTorch SDPA baseline
python3 run_sdpa.py

# Run our kernel (once build succeeds)
python3 run_ours.py

# Compare results
python3 compare.py

# View report
cat ../benchmarks/ATTN_L4_report.md
```

## Files

### Core Benchmark Scripts

- **`run_sdpa.py`** - Benchmark PyTorch SDPA (cuDNN Flash backend)
- **`run_fa.py`** - Benchmark FlashAttention-2 or xFormers (if installed)
- **`run_ours.py`** - Benchmark our FA-1 kernel
- **`compare.py`** - Aggregate results to CSV and markdown tables

### Production Harness

- **`benchmark_harness.py`** - Production-grade benchmarking framework with:
  - CUDA event timing
  - Statistical analysis (mean, median, std, percentiles)
  - L2 cache flushing support
  - Clock locking awareness
  - GFLOPS and bandwidth calculation
  - JSON output for reproducibility
  - Baseline comparison

- **`example_harness_usage.py`** - Demonstrates harness integration

### Output

- **`out/`** - Benchmark results directory
  - `sdpa_results.jsonl` - PyTorch SDPA raw results
  - `ours_results.jsonl` - Our kernel raw results
  - `comparison.csv` - Aggregated comparison
  - `comparison.md` - Human-readable table
  - `harness_results/` - Production harness JSON outputs

## Benchmark Protocol

### Standard Configuration

```python
warmup_iterations = 200
benchmark_iterations = 500
timing_method = "CUDA events"
exclude_memory_transfers = True
seed = 42
```

### Workload Matrix

| B | H | S | d | Precision | Total Queries |
|---|---|---|---|-----------|---------------|
| 1 | 1 | 128 | 64 | FP16/BF16 | 128 |
| 8 | 4 | 128 | 64 | FP16/BF16 | 4,096 |
| 32 | 8 | 128 | 64 | FP16/BF16 | 32,768 |
| 32 | 8 | 256 | 64 | FP16/BF16 | 65,536 |

### Metrics Collected

- Latency: mean, median, std dev, min, max, 95th/99th percentile
- Throughput: GFLOPS (4*B*H*S²*d operations)
- Bandwidth: GB/s (4*B*H*S*d elements × sizeof)
- Queries per second
- GPU utilization (from profiler)

## Using the Production Harness

### Basic Usage

```python
from benchmark_harness import CUDABenchmarkHarness, BenchmarkConfig

# Configure
config = BenchmarkConfig(
    warmup_iterations=200,
    benchmark_iterations=500,
    flush_l2_cache=False,
    lock_clocks=False
)

# Create harness
harness = CUDABenchmarkHarness(config)

# Define kernel wrapper
def my_kernel():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    # ... launch kernel ...
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end)  # ms

# Benchmark
result = harness.benchmark_kernel(
    kernel_func=my_kernel,
    kernel_name="my_attention_kernel",
    flop_count=4 * B * H * S * S * d,
    memory_bytes=4 * B * H * S * d * 2
)

# Save
harness.save_results(result, Path("out/my_kernel.json"))
```

### Configuration Sweep

```bash
# Run sweep across all configurations
python3 example_harness_usage.py --sweep
```

### Baseline Comparison

```python
# Compare against baseline
comparison = harness.compare_results(
    baseline_path=Path("out/harness_results/baseline.json"),
    current_result=result
)

print(f"Speedup: {comparison['speedup']:.3f}x")
print(f"Improvement: {comparison['improvement_pct']:.2f}%")
```

## Environment

### Hardware
- GPU: NVIDIA L4 (SM_89, 58 SMs, 48KB shared mem, 300 GB/s)
- Driver: 570.172.08
- CUDA: 12.1
- cuDNN: 8902

### Software
- PyTorch: 2.2.1+cu121
- Python: 3.10
- OS: Ubuntu 22.04 LTS

## Results

Current status: PyTorch SDPA baseline complete (8 configs).

See `../benchmarks/ATTN_L4_report.md` for full report.

### PyTorch SDPA Performance (FP16)

| Config | Latency (ms) | GFLOPS | Bandwidth (GB/s) |
|--------|--------------|--------|------------------|
| B=1, H=1, S=128 | 0.0421 | - | - |
| B=8, H=4, S=128 | 0.0406 | - | - |
| B=32, H=8, S=128 | 0.0426 | - | - |
| B=32, H=8, S=256 | 0.0980 | - | - |

## Troubleshooting

### Build Issues

```bash
# Clean rebuild
cd ../
rm -rf build/ python/flashmoe_science/_C*
python3 setup.py build_ext --inplace

# Check extension
python3 -c "import flashmoe_science._C; print('OK')"
```

### Import Errors

```bash
# Set paths
export LD_LIBRARY_PATH=/path/to/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/periodicdent42/cudadent42/python:$PYTHONPATH
```

### Missing Dependencies

```bash
pip install numpy torch
```

## Best Practices

1. **Always warmup**: 200+ iterations for GPU to reach steady state
2. **Use CUDA events**: More accurate than Python `time.time()`
3. **Exclude transfers**: Benchmark kernel only, not H2D/D2H
4. **Fix seed**: Reproducible inputs across runs
5. **Run 500+ iterations**: Statistical significance
6. **Lock clocks** (optional): `sudo nvidia-smi -lgc 0,-1` for minimal variance
7. **Save raw data**: JSON files for auditing
8. **Document environment**: GPU, driver, CUDA, PyTorch versions

## References

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- PyTorch SDPA docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Nsight Compute: https://developer.nvidia.com/nsight-compute

