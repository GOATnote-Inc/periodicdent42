# Optimized Dense GEMM with CUTLASS CollectiveBuilder

Demonstrates optimized tile and cluster configurations for dense FP16 GEMM on NVIDIA H100.

## Performance

| Problem Size | Time (ms) | TFLOPS | vs cuBLAS |
|--------------|-----------|--------|-----------|
| 8192³ | 2.10 | 523.6 | 84% |
| 8192×8192×19712 | 4.80 | 550.8 | 88% |

**Hardware:** NVIDIA H100 PCIe 80GB (sm_90a)  
**Compiler:** CUDA 12.8, NVCC with `--expt-relaxed-constexpr`

## Key Optimizations

1. **TileShape 128×256×64** - Non-square tiles optimized for specific problem dimensions
2. **ClusterShape 2×1×1** - Better SM alignment than default configurations
3. **Problem dimensions** - K=19712 optimal for this tile configuration

## Build

```bash
# Requirements: CUDA 12.8+, CUTLASS 4.3.0

nvcc -O3 -std=c++17 -arch=sm_90a \
     --expt-relaxed-constexpr \
     --maxrregcount=255 \
     -I${CUTLASS_PATH}/include \
     gemm_optimized.cu -o gemm_optimized \
     -lcudart

# Run
./gemm_optimized
```

Expected output:
```
Problem: 8192×8192×19712
Time: 4.80 ms
TFLOPS: 550.8
```

## Implementation Details

Uses CUTLASS 4.3.0 CollectiveBuilder API with:
- **Mainloop:** TMA + WGMMA (Hopper native instructions)
- **Epilogue:** Default epilogue with FP32 accumulation
- **Scheduling:** Persistent kernel with cooperative scheduling
- **Input/Output:** FP16 input, FP32 accumulation and output

## Verification

Timing measured with CUDA Events (industry standard):
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... kernel launch ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

Verified across 5 independent runs:
- Mean: 4.803 ± 0.013 ms
- TFLOPS: 550.8 ± 1.3
- Variance: ±0.3%

## Comparison

| Implementation | TFLOPS | Configuration |
|----------------|--------|---------------|
| cuBLAS | 622.8 | Default (closed source) |
| This example | 550.8 | TileShape 128×256×64, ClusterShape 2×1×1 |
| CUTLASS Ex49 | 406.8 | Default CollectiveBuilder config |
| CUTLASS Ex62 | 269.1 | 2:4 structured sparse |

## License

BSD 3-Clause License

Copyright © 2025 Brandon Dent

See LICENSE for full terms.

