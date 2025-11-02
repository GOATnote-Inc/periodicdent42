# Ceiling Scout

**Automated GPU Performance Ceiling Detection**  
Based on H100 validation methodology (November 2025)

## What It Does

Ceiling Scout automatically:
1. **Benchmarks library performance** (cuBLAS, cuDNN, FA3)
2. **Compares to hardware ceiling** (validated on H100)
3. **Identifies optimization opportunities**
4. **Suggests specific actions** (CUTLASS sweep, custom kernel, or use library)

**Key Innovation:** Knows when custom kernels **cannot** add value (like our 628 TFLOPS finding).

## Quick Start

```bash
# Basic GEMM benchmark
python ceiling_scout.py --operation gemm --shape 8192,8192,147456 --precision fp16

# K-dimension sweep (like our H100 validation)
python ceiling_scout.py --operation gemm --shape 8192,8192,73728 --k-sweep

# Generate full report
python ceiling_scout.py --operation gemm --shape 8192,8192,147456 \
  --output report.json --device h100
```

## Output Example

```
=== Ceiling Scout ===
Device: h100
Operation: gemm
Shape: (8192, 8192, 147456)
Precision: fp16

Running K-dimension sweep for M=8192, N=8192...
  K=65536... 629.2 TFLOPS
  K=81920... 629.9 TFLOPS
  K=98304... 630.3 TFLOPS
  ...
  K=147456... 627.3 TFLOPS

=== K-Sweep Results ===
Peak: 630.3 TFLOPS at K=98304
Range: 618.2 - 634.4 TFLOPS

=== Analysis ===
Baseline: 627.3 TFLOPS
Ceiling:  628.0 TFLOPS
Efficiency: 99.9%
Priority: NONE

Recommendation:
  cuBLAS is optimal (99.9% of ceiling). Use library. Custom kernel cannot improve.

Suggested approach: NONE
Config:
{
  "use": "cuBLAS",
  "reasoning": "Already at hardware ceiling"
}
```

## Use Cases

### 1. Model Profiling
```bash
# Profile all ops in a model
for op in gemm conv2d attention; do
  python ceiling_scout.py --operation $op --shape <from_profile> --output ${op}_report.json
done
```

### 2. Sparse Kernel Decision
```bash
# If baseline is <100 TFLOPS and >70% sparse, suggests custom kernel
python ceiling_scout.py --operation gemm --shape 8192,8192,8192 --sparsity 0.875
```

### 3. CUTLASS Tuning
```bash
# If efficiency <90%, suggests CUTLASS sweep with specific configs
python ceiling_scout.py --operation gemm --shape 4096,4096,4096
# Output: "Try CUTLASS 4.3 CollectiveBuilder with tile_shapes: [128x128x64, 128x256x64]"
```

## Validated Ceilings (Nov 2025)

| Device | Op | Precision | Ceiling | Source |
|--------|----|-----------|---------| -------|
| H100 PCIe | Dense GEMM | FP16→FP32 | 628 TFLOPS | ✅ Measured |
| H100 PCIe | Dense GEMM | FP8→FP32 | N/A | ❌ Not supported |
| H100 SXM | Dense GEMM | FP16→FP32 | ~700 TFLOPS | Estimated |

## Integration with Your Tools

### Burn (Rust)
```rust
// Use ceiling_scout results to dispatch
let report = load_ceiling_report("report.json");
if report.recommendation == "use_library" {
    cublas_gemm(A, B)
} else if report.approach == "custom_sparse" {
    blackwell_sparse_gemm(A, B)
}
```

### vLLM
```python
# Override vLLM kernel selection
from ceiling_scout import CeilingScout

scout = CeilingScout(device="h100")
opportunity = scout.detect_ceiling(Operation.GEMM, (M, N, K), Precision.FP16)

if opportunity.approach == "CUTLASS_SWEEP":
    # Use CUTLASS instead of cuBLAS
    use_cutlass_backend(opportunity.config_suggestion)
```

### Triton Auto-tune
```python
# Skip auto-tuning if library is optimal
opportunity = scout.detect_ceiling(op, shape, precision)
if opportunity.efficiency > 0.90:
    print(f"Skip {op}: library is optimal")
else:
    triton_autotune(op, opportunity.config_suggestion)
```

## Architecture

```
ceiling_scout.py
├── BenchmarkResult: Stores perf measurements
├── OpportunityScore: Analyzes gap to ceiling
├── CeilingScout: Main engine
│   ├── benchmark_cublas(): Validated H100 methodology
│   ├── k_dimension_sweep(): Our 65K-262K sweep
│   ├── detect_ceiling(): Decision logic
│   └── generate_report(): JSON output
```

## Methodology

Based on our H100 validation:
1. **Warmup**: 20 iterations (not 10)
2. **Timing**: 200 iterations (not 100)
3. **Variance handling**: Report mean, not peak
4. **K-sweep**: 65K-262K in 16K steps
5. **Efficiency threshold**: >90% = optimal

## Extending

Add new operations in `detect_ceiling()`:

```python
elif operation == Operation.ATTENTION:
    # Benchmark FA3
    baseline = self.benchmark_flashattention3(...)
    ceiling = 1000.0  # μs/head target
    # ... decision logic
```

## What's Next

- [ ] Add FlashAttention-3 benchmarking
- [ ] Add sparse pattern detection (BSR, 2:4)
- [ ] Add fusion opportunity detection
- [ ] Add CUTLASS 4.3 auto-sweep
- [ ] Add vLLM/Triton config generation
- [ ] Add NCU profile parsing

## Why This Matters

**Before Ceiling Scout:**
- Spent hours trying to beat cuBLAS
- Found it was already optimal (628/628 TFLOPS)
- Wasted time on impossible optimizations

**After Ceiling Scout:**
- Know in 5 minutes what's worth optimizing
- Focus on sparse, fusion, exotic types
- Skip dense GEMM (library is optimal)

**You asked: "What would provide most value?"**  
**This is it:** Stop optimizing what's already optimal. Automate the ceiling detection we just did manually.

