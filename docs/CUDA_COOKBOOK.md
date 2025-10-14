# CUDA Kernel Engineering Cookbook

**Purpose**: Hermetic, reproducible CUDA kernel development for NVIDIA L4 (SM_89)

**Target**: 0.5-1.0× performance with profiler receipts > 10× claims with no evidence

**Philosophy**: Profile → Diagnose → Fix → Measure → Document → Repeat

---

## Quick Start

```bash
# 1. Setup environment
source scripts/verify_env.sh

# 2. Build kernel
cd ext && python setup_fa_s512.py build_ext --inplace && cd ..

# 3. Test correctness
python cudadent42/bench/correctness_fuzz.py

# 4. Benchmark performance
python cudadent42/bench/baseline_comprehensive.py --only s512

# 5. Profile (optional)
S=512 bash scripts/profile_sdpa.sh
```

---

## Architecture Primer: NVIDIA L4 (SM_89 / Ada Lovelace)

| Resource | Specification |
|----------|---------------|
| **SMs** | 58 streaming multiprocessors |
| **Shared Memory** | 48 KB per SM (soft limit), 100 KB max |
| **Registers** | 65,536 × 32-bit per SM |
| **Warp Size** | 32 threads |
| **Max Threads/Block** | 1024 |
| **Max Blocks/SM** | 32 (dynamic) |
| **HBM2e Bandwidth** | 300 GB/s peak (242 GB/s sustained) |
| **FP16 Tensor Cores** | 242 TFLOPS peak (30 TFLOPS sustained) |
| **L2 Cache** | 48 MB |
| **TDP** | 72W |

### Memory Hierarchy (Latency)

1. **Registers**: 1 cycle (~0.5 ns)
2. **Shared Memory**: 28-32 cycles (~15 ns) - 100× faster than DRAM
3. **L2 Cache**: ~200 cycles (~100 ns)
4. **HBM2e**: ~300-400 cycles (~150-200 ns)

**Key Insight**: Optimize for shared memory reuse → minimize DRAM accesses.

---

## Profiling & Metrics

### Essential Nsight Compute Metrics

```bash
S=512 bash scripts/profile_sdpa.sh
```

**What to Look At** (Priority Order):

1. **Tensor Core Utilization** (`smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`)
   - Target: >80% (indicates compute saturation)
   - If low: Increase block size, warps, or reduce stalls

2. **DRAM Throughput** (`dram__throughput.avg.pct_of_peak_sustained_elapsed`)
   - Target: <30% for compute-bound workloads (S≥512)
   - If high: Improve data reuse, increase tile sizes

3. **L2 Cache Hit Rate** (`lts__t_sector_hit_rate.pct`)
   - Target: >70% (indicates good tiling)
   - If low: Adjust tile sizes to fit in L2

4. **Warp Stall Reasons** (`smsp__warp_issue_stalled_*`)
   - Identify top stall: `long_scoreboard`, `barrier`, `math_pipe_throttle`, `not_selected`
   - Fix priority: long_scoreboard > barrier > math_pipe

5. **Shared Memory Bank Conflicts** (`l1tex__data_bank_conflicts_pipe_lsu.sum`)
   - Target: 0 (no conflicts)
   - Fix: Pad arrays by +1 element, or use swizzle patterns

6. **Occupancy** (`sm__warps_active.avg.pct_of_peak_sustained_active`)
   - Target: >50% for compute-bound, but can be lower if high reuse
   - If low: Reduce register usage, shared memory, or increase threads/block

---

## Benchmarking Protocol

### Baseline Characterization

```python
python cudadent42/bench/baseline_comprehensive.py --only s512
```

**Output**:
- Median latency + 95% bootstrap CI
- P50/P95/P99 tail latencies
- Throughput (GFLOPS)
- Memory bandwidth (GB/s)
- GPU state (power, clocks, temperature)

**Statistics**:
- N=100 samples (warm-up: 20 iterations)
- Bootstrap CIs (10,000 resamples, seed=42)
- Coefficient of variation (CV) - warn if >12%
- Temperature monitoring - warn if >80°C

### Correctness & Validation

```python
python cudadent42/bench/correctness_fuzz.py
```

**Test Matrix**: 27 configurations
- S ∈ {448, 512, 640}
- B ∈ {16, 32, 48}
- H ∈ {4, 8, 16}
- D = 64 (fixed)

**Tolerances** (FP16):
- Absolute: atol = 2e-3
- Relative: rtol = 1e-3

**Pass Requirement**: 100% of tests (27/27)

---

## Build Strategy

### Pre-Compiled Extension (Recommended)

**Avoids PyTorch JIT timeout** (>5 minutes)

```bash
cd ext
python setup_fa_s512.py build_ext --inplace
python -c "import fa_s512; print('OK')"
```

**Build Time**:
- First build: 5-15 minutes (cold cache, full compile)
- Rebuild: 10-30 seconds (with ccache)

**Environment Variables**:
```bash
export TORCH_CUDA_ARCH_LIST="8.9"  # L4 only (5× faster than multi-arch)
export MAX_JOBS=$(nproc)  # Parallel builds
export CUDAFLAGS="-O3 --use_fast_math -lineinfo"
export CCACHE_DIR="$HOME/.ccache"  # Compilation cache
```

### Tools

- **Ninja**: Parallel builds (5-10× faster than make)
- **ccache**: Compilation caching (10-30s rebuild)
- **Nsight Compute**: Profiling (metric collection)

---

## Optimization Catalog

**Order to Try** (Based on Roofline Model):

### 1. Increase Occupancy (If <50%)

**Goal**: More concurrent work per SM

```cuda
// Before
__launch_bounds__(128)  // 4 warps, low occupancy

// After
__launch_bounds__(256, 2)  // 8 warps, 2 blocks/SM
```

**Expected**: +10-30% if memory-bound

### 2. Maximize Tensor Core Utilization

**Goal**: Keep tensor cores busy (>80% active)

**Check**: `smsp__pipe_tensor_cycles_active`

**Fixes**:
- Use `mma.sync` or `wmma` for FP16 matmul
- Increase tile sizes (BLOCK_M, BLOCK_N)
- Reduce warp stalls (see Nsight)

**Expected**: +30-50% if <60% utilization

### 3. Eliminate Shared Memory Bank Conflicts

**Goal**: 0 bank conflicts

**Check**: `l1tex__data_bank_conflicts_pipe_lsu.sum`

```cuda
// Padding to avoid conflicts
__shared__ half Q_smem[BLOCK_M][BLOCK_K + 1];  // +1 for padding
```

**Expected**: +5-15% if conflicts present

### 4. Loop Unrolling

**Goal**: Reduce loop overhead

```cuda
#pragma unroll 4
for (int k = 0; k < K; k += 4) {
    // Load 4 elements at once
}
```

**Expected**: +5-10% if memory-bound

### 5. Asynchronous Copy (`cp.async`)

**Goal**: Overlap memory and compute

```cuda
// SM80+ (Ampere/Ada/Hopper)
__pipeline_memcpy_async(&smem[0], &gmem[0], sizeof(float) * N);
__pipeline_commit();
__pipeline_wait_prior(0);
```

**Expected**: +10-20% if memory-bound

---

## Ablation & CI

### Performance Pass Bars

**CI Gate** (`.github/workflows/perf.yml`):

1. **Correctness**: 100% pass rate (27/27 configs)
2. **Regression**: < 3% slower (with non-overlapping 95% CIs)
3. **Improvement**: ≥ 10% faster + non-overlapping CIs + |Cliff's δ| ≥ 0.3

### Statistical Requirements

```python
python cudadent42/bench/ci_compare.py new_results.json .ci/baseline_s512.json
```

**Exit Codes**:
- 0: Maintained or improved (statistically significant)
- 1: Regression detected (>3% slower, non-overlapping CIs)
- 2: No significant difference (CIs overlap or small effect size)

**Metrics**:
- **Bootstrap 95% CI**: Non-parametric confidence intervals
- **Cliff's Delta**: Effect size (require |δ| ≥ 0.3 for "medium" effect)
- **Mann-Whitney U**: Non-parametric significance test (p < 0.05)

---

## Deployment

### Import & Use

```python
import fa_s512

# Q, K, V: [B, H, S=512, D=64] FP16 tensors
output = fa_s512.fa_s512(Q, K, V)
```

### Fallback Strategy

```python
try:
    import fa_s512
    use_custom = True
except ImportError:
    use_custom = False

if use_custom and S == 512 and D == 64:
    output = fa_s512.fa_s512(Q, K, V)
else:
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
```

---

## References

### NVIDIA Documentation

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Ada (SM_89) Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/)

### Papers

- FlashAttention: [Dao et al., 2022](https://arxiv.org/abs/2205.14135)
- FlashAttention-2: [Dao, 2023](https://arxiv.org/abs/2307.08691)
- Roofline Model: [Williams et al., 2009](https://doi.org/10.1145/1498765.1498785)

### Tools

- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [NVIDIA Profiler](https://developer.nvidia.com/nvidia-visual-profiler)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA Templates for Linear Algebra

---

## Troubleshooting

### Build Issues

**Problem**: "ncu not found"  
**Solution**: Install Nsight Compute 2024.1.1 (matches CUDA 12.1)

**Problem**: Build timeout (>5 min)  
**Solution**: Use pre-compiled extension (`ext/setup_fa_s512.py`), not JIT

**Problem**: "Ninja not found"  
**Solution**: `pip install ninja` and add `~/.local/bin` to PATH

### Performance Issues

**Problem**: "Slower than PyTorch SDPA"  
**Solution**: Profile with Nsight, identify top bottleneck, fix priority 1-5

**Problem**: "High variance (CV >12%)"  
**Solution**: Check GPU temperature (<80°C), increase sample size (N=100 → 200)

**Problem**: "CIs overlap, can't claim improvement"  
**Solution**: Increase sample size or accept that difference is not significant

---

## Quick Reference

### Commands

```bash
# Environment
bash scripts/verify_env.sh

# Build
cd ext && python setup_fa_s512.py build_ext --inplace && cd ..

# Test
python cudadent42/bench/correctness_fuzz.py

# Benchmark
python cudadent42/bench/baseline_comprehensive.py --only s512

# Profile
S=512 bash scripts/profile_sdpa.sh

# Compare
python cudadent42/bench/ci_compare.py new.json .ci/baseline_s512.json
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success / Maintained / Improved |
| 1 | Failure / Regression |
| 2 | Skipped / No significant difference |

---

**Status**: Cookbook complete and operational  
**Last Updated**: 2025-10-14  
**Maintainer**: GOATnote Autonomous Research Lab Initiative

