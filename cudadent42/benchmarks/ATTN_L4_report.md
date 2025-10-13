# FlashAttention Kernel Benchmark Report - NVIDIA L4

## 1. Hardware and Software Environment

| Component | Version/Model |
|-----------|---------------|
| GPU | NVIDIA L4 (SM_89, Ada Lovelace) |
| CUDA Cores | 7,424 |
| Tensor Cores | 232 (4th gen) |
| Memory | 24GB GDDR6 |
| Memory Bandwidth | 300 GB/s |
| Peak FP16 | 242 TFLOPS (with sparsity) |
| Driver | 570.172.08 |
| CUDA | 12.1 |
| cuDNN | 8902 |
| PyTorch | 2.2.1+cu121 |
| OS | Ubuntu 22.04 LTS |

## 2. Benchmark Protocol

- **Timing method**: CUDA events (`torch.cuda.Event`)
- **Warmup iterations**: 200
- **Measurement iterations**: 500
- **Data transfer**: Excluded (H2D/D2H not measured)
- **Seed**: Fixed (42) for reproducibility
- **Precision**: FP16 and BF16 (native support on SM_89)
- **Causal masking**: False (full attention)
- **Repository commit**: 8933e95

### Configuration Matrix

| B | H | S | d | Total Queries |
|---|---|---|---|---------------|
| 1 | 1 | 128 | 64 | 128 |
| 8 | 4 | 128 | 64 | 4,096 |
| 32 | 8 | 128 | 64 | 32,768 |
| 32 | 8 | 256 | 64 | 65,536 |

## 3. Baseline Implementations

1. **PyTorch SDPA**: `torch.nn.functional.scaled_dot_product_attention`
   - Backend: Flash attention (cuDNN-based)
   - Version: PyTorch 2.2.1
   
2. **FlashAttention-2**: Official Dao-AILab implementation
   - Status: Not installed on test system
   
3. **Our Kernel**: FlashAttention-Science (FA-1 architecture)
   - Status: Compilation blocker (ATen type template instantiation)
   - Build time: >90 minutes of debugging
   - Issue: `undefined symbol: _ZN8flashmoe23flash_attention_forwardIN3c108BFloat16EEEvPKT_S5_S5_PS3_Pfiiiifb`

## 4. Results

### 4.1 PyTorch SDPA (Baseline)

| Config | dtype | Latency (ms) | Std (ms) | Queries/sec |
|--------|-------|--------------|----------|-------------|
| B=1, H=1, S=128 | fp16 | 0.0421 | 0.0062 | 3.04M |
| B=8, H=4, S=128 | fp16 | 0.0406 | 0.0042 | 100.9M |
| B=32, H=8, S=128 | fp16 | 0.0426 | 0.0055 | 769.0M |
| B=32, H=8, S=256 | fp16 | 0.0980 | 0.0093 | 668.8M |
| B=1, H=1, S=128 | bf16 | 0.0403 | 0.0037 | 3.18M |
| B=8, H=4, S=128 | bf16 | 0.0410 | 0.0044 | 99.9M |
| B=32, H=8, S=128 | bf16 | 0.0427 | 0.0044 | 767.8M |
| B=32, H=8, S=256 | bf16 | 0.0965 | 0.0048 | 679.1M |

### 4.2 Our Kernel (FA-1)

**Status**: Build failure prevents benchmarking.

**Attempted fixes**:
- Template instantiation for `at::Half` and `at::BFloat16`
- Added `to_float()` and `from_float()` device function overloads
- Included `<ATen/ATen.h>` and `<c10/core/ScalarType.h>`
- Result: Unresolved extern function errors

**Previous measurements** (from Session N+6, commit prior to template refactor):
- B=8, H=8, S=128, d=64: 0.579 ms (14.9× slower than PyTorch SDPA)
- Note: These measurements use different configuration (H=8 vs H=4/8) and are not directly comparable to current baseline.

### 4.3 Correctness

**PyTorch SDPA**: Reference implementation (assumed correct).

**Our Kernel**: Unable to verify due to build failure.

## 5. Profiling Summary

### PyTorch SDPA (B=8, H=8, S=128)

Profiling deferred due to time constraints. Previous profiling data from Session 1 (Oct 13):

**Kernel Launch Configuration**:
- Blocks: 2 (on 58-SM GPU = 3.4% utilization)
- Threads per block: 256
- Registers per thread: Not measured
- Shared memory per block: Not measured

**Performance Counters** (Session 1, our kernel):
- SM Throughput: <5% of peak (estimated from block count)
- DRAM Throughput: Not measured
- Achieved FLOP/s: Not measured

**Roofline Analysis**: Deferred pending successful build.

## 6. Interpretation

1. **PyTorch SDPA performance**: Consistent ~0.04-0.10 ms latency across configurations, achieving 668-769M queries/sec for larger batches. No significant difference between FP16 and BF16.

2. **Our kernel status**: Template instantiation issues prevent benchmarking. The FA-1 architecture compiles successfully with native CUDA types (`__half`, `__nv_bfloat16`) but fails when using ATen types (`at::Half`, `at::BFloat16`) required for PyTorch binding.

3. **Build complexity**: 90+ minutes of debugging template issues highlights the challenge of maintaining cross-library type compatibility in CUDA extensions. A simpler approach (single compilation unit or native-only types) is recommended for reproducibility.

## 7. Recommendations

1. **Short-term**: Revert to native CUDA types (`__half`, `__nv_bfloat16`) with type conversion at the Python binding layer.

2. **Medium-term**: Implement simple wrapper structs to avoid ATen type dependencies in device code.

3. **Long-term**: Adopt single-header implementation pattern (e.g., `bindings_native.cu` including kernel source directly) to avoid separate compilation and linking issues.

## 8. Reproducibility

### Build Commands

```bash
cd cudadent42
rm -rf build/ python/flashmoe_science/_C*
python3 setup.py build_ext --inplace
```

### Benchmark Commands

```bash
# PyTorch SDPA
python3 bench/run_sdpa.py

# Our kernel (once build succeeds)
export LD_LIBRARY_PATH=/path/to/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/periodicdent42/cudadent42/python:$PYTHONPATH
python3 bench/run_ours.py

# Comparison
python3 bench/compare.py
```

### Artifacts

- Raw results: `bench/out/sdpa_results.jsonl`
- Comparison CSV: `bench/out/comparison.csv`
- Comparison MD: `bench/out/comparison.md`

## 9. Conclusion

PyTorch SDPA demonstrates consistent performance across FP16 and BF16 precisions on L4 GPU. Our FA-1 kernel requires build system refactoring to enable benchmarking. Template instantiation complexity prevented comparative measurements in this session.

**Measurement Status**: 1/3 implementations benchmarked (PyTorch SDPA complete, FlashAttention-2 unavailable, our kernel blocked).

---

**Report generated**: October 13, 2025  
**Session duration**: 150 minutes  
**Blocker**: Template compilation (90 minutes debugging)  
**GPU cost**: $1.20 (L4 @ $0.48/hour × 2.5 hours)

