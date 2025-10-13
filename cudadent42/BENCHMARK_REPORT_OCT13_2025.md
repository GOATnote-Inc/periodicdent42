# FlashAttention Kernel Benchmark Report

**Date**: October 13, 2025  
**Author**: Autonomous CUDA Optimization System  
**Purpose**: Quantify kernel performance against industry-standard baseline

## Hardware and Software Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA L4 (Ada Lovelace, SM_89) |
| SMs | 58 |
| Peak FP16 | ~30 TFLOP/s (theoretical) |
| Memory BW | 300 GB/s |
| CUDA Toolkit | 12.1 |
| Driver | 535.129.03 |
| PyTorch | 2.2.1+cu121 |
| Instance | cudadent42-l4-dev (GCP us-central1-a) |
| Date | 2025-10-13 |

## Benchmark Protocol

### Methodology
- Warmup: 20-50 iterations per kernel
- Measurement: 100-500 iterations per configuration
- Synchronization: `torch.cuda.synchronize()` before/after each run
- Timing: Python `time.perf_counter()` with CUDA events
- Statistics: Mean ± std, 95% confidence intervals (n≥100)

### Configuration Space
Fixed parameters:
- Data type: FP16
- Head dimension: D=64
- Attention type: Causal (lower triangular mask)
- Scale: 1/√D

Varied parameters:
- Batch size (B): 1, 4, 8, 8, 16
- Number of heads (H): 1, 8, 8, 8, 8
- Sequence length (S): 128, 128, 128, 256, 512

## Results

### Comparative Performance

| Configuration | Our Kernel (ms) | PyTorch SDPA (ms) | Ratio | GFLOP/s (Ours) | GFLOP/s (PyTorch) |
|---------------|-----------------|-------------------|-------|----------------|-------------------|
| B=1, H=1, S=128  | 0.579 ± 0.004 | 0.085 ± 0.011 | 6.8×  | 7.2   | 49.4   |
| B=4, H=8, S=128  | 0.752 ± 0.002 | 0.085 ± 0.008 | 8.9×  | 178.6 | 1588.2 |
| B=8, H=8, S=128  | 1.281 ± 0.004 | 0.086 ± 0.008 | 14.9× | 209.6 | 3130.5 |
| B=8, H=8, S=256  | 4.067 ± 0.009 | 0.116 ± 0.012 | 35.0× | 264.0 | 9242.6 |
| B=16, H=8, S=512 | 23.900 ± 0.021 | 0.326 ± 0.019 | 73.2× | 359.4 | 26312.6 |

**Interpretation**: Ratio >1.0 indicates our kernel is slower. Our implementation ranges from 6.8× to 73.2× slower than PyTorch SDPA across tested configurations.

### Correctness Verification

Tested configuration: B=8, H=8, S=128, D=64, FP16, causal
- Maximum absolute difference: 0.001953
- Mean absolute difference: 0.000002
- Status: Within FP16 numerical tolerance

### Throughput Analysis

**Peak theoretical (L4 FP16)**: ~30,000 GFLOP/s

| Implementation | Min GFLOP/s | Max GFLOP/s | Peak % |
|----------------|-------------|-------------|--------|
| Our kernel     | 7.2         | 359.4       | 0.02-1.2% |
| PyTorch SDPA   | 49.4        | 26,312.6    | 0.2-88% |

PyTorch SDPA achieves up to 88% of theoretical peak at large batch sizes. Our kernel achieves 0.02-1.2% of peak, indicating severe underutilization.

## Bottleneck Analysis

### Observed Patterns

1. **Scaling degradation**: Performance ratio worsens with workload size (6.8× → 73.2×)
2. **Low GFLOP/s**: Peak 359 GFLOP/s vs PyTorch's 26,313 GFLOP/s
3. **Fixed overhead dominant**: Small batch (B=1) shows best relative performance

### Root Cause Hypothesis

From Session 1 profiling:
- **GPU utilization**: 3.4% at B=1, H=1
- **CTA count**: 2 CTAs launched (L4 has 58 SMs)
- **Parallelism bottleneck**: Kernel grid insufficient for GPU capacity

Expected behavior:
- Small batches: Low CTA count → underutilization → slow
- Large batches: Should improve with more CTAs, but ratio worsens

**Contradiction**: Large batches get relatively worse, suggesting additional bottleneck beyond parallelism.

### Secondary Bottlenecks (To Profile)

1. **Memory access pattern**: Non-coalesced loads/stores
2. **Shared memory conflicts**: Bank conflicts in 64×64 tiles
3. **Register pressure**: Spillage reducing occupancy
4. **Algorithmic overhead**: Online softmax recomputation cost

## Comparison to Literature

**FlashAttention-2 (Dao et al., 2023)**:
- Reported: 1.3-2.0× faster than PyTorch SDPA on A100
- Our result: 6.8-73.2× slower than PyTorch SDPA on L4
- Gap: 8.8-146× performance difference vs published baseline

**Note**: Direct comparison difficult due to:
- Different GPU architecture (A100 vs L4)
- Potential implementation differences
- Version/optimization changes in PyTorch 2.2.1

## Discussion

### Current State
The kernel produces numerically correct results but exhibits poor performance across all tested configurations. The degradation at larger workloads suggests the implementation has both parallelism limitations and computational inefficiencies.

### Identified Issues

**Critical**:
1. Insufficient CTAs launched (2 CTAs on 58-SM GPU = 3.4% utilization)
2. Unknown memory bottleneck (performance worsens at scale)

**High Priority**:
1. Memory access patterns not profiled
2. Shared memory bank conflicts not measured
3. Occupancy metrics unavailable

### Recommended Next Steps

**Iteration 3: Nsight Compute Profiling** (30 min)
- Profile with `ncu --set full` at B=8, H=8, S=128
- Extract memory bandwidth utilization
- Measure warp occupancy and stall reasons
- Identify top 3 bottlenecks

**Iteration 4: Memory Coalescing Fix** (45-60 min)
- Implement vectorized loads (float4)
- Fix strided access patterns
- Expected: 1.5-2× improvement

**Iteration 5: Persistent Kernel** (60-90 min)
- Implement work queue for small batches
- Target: Improve 6.8× ratio to 2-3×

## Conclusions

Benchmark data demonstrates significant performance gap between the current implementation and PyTorch SDPA. The kernel achieves 0.02-1.2% of theoretical peak performance compared to PyTorch's 0.2-88%.

Performance improves marginally with increased parallelism (7.2 → 359.4 GFLOP/s) but degrades relative to baseline (6.8× → 73.2×slower), indicating both parallelism and computational bottlenecks.

Next iteration should focus on profiler-guided optimization to identify and address memory or compute inefficiencies.

## References

- PyTorch 2.2.1 SDPA: `torch.nn.functional.scaled_dot_product_attention`
- FlashAttention-2: Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
- NVIDIA L4 Datasheet: https://resources.nvidia.com/en-us-l4-datasheet

## Appendix: Raw Data

```
Configuration: B=8, H=8, S=128, D=64 (n=500)
Our kernel:    1.2829 ± 0.0055 ms (95% CI: ±0.0005 ms)
PyTorch SDPA:  0.0860 ± 0.0187 ms (95% CI: ±0.0016 ms)
Correctness:   max_diff=0.001953, mean_diff=0.000002
```

