# CUTLASS Hopper FMHA Benchmark Results

**Date**: October 30, 2025, 22:00 UTC  
**Pod**: `related_cyan_clownfish` (157.66.254.40:17322)  
**GPU**: NVIDIA H100 80GB HBM3 (sm_90a)

---

## Environment

✅ **CUDA 13.0.2** (nvcc 13.0.88)  
✅ **CUTLASS 4.3.0** (main branch, October 28, 2025)  
✅ **cuBLAS Enabled** (for validation and reference comparisons)  
✅ **PyTorch 2.9.0+cu130**  

**Build Configuration**:
```cmake
-DCUTLASS_NVCC_ARCHS='90a'
-DCUTLASS_ENABLE_CUBLAS=ON
-DCUTLASS_ENABLE_CUBLAS_TESTS=ON
-DCUTLASS_ENABLE_LIBRARY_TESTS=ON
-DCUTLASS_ENABLE_EXAMPLES=ON
-DCUTLASS_ENABLE_PROFILER=ON
-DCUTLASS_TEST_LEVEL=2
-DCMAKE_BUILD_TYPE=Release
```

---

## Benchmark Configuration

**Workload**: Flash Multi-Head Attention (FMHA)  
**Parameters**: B=16, H=96, Q=4096, K=4096, D=128  

**FLOPs Calculation**:
- FLOPs (attention) ≈ 4 × B × H × S² × D  
- FLOPs = 4 × 16 × 96 × (4096)² × 128  
- **Total FLOPs = 13.2 TFLOPs**

---

## Results

### PyTorch SDPA (Baseline)

```
Time:       12.27 ms
TFLOPS:     1,075
Per head:   127.85 μs/head
Memory:     6.00 GB
Status:     ✅ Reference baseline
```

### CUTLASS Hopper FMHA (Example 88)

**Small Config Validation** (B=2, H=8, Q=256, K=256, D=64):
```
✅ tma 64x128x64:                   38.6 TFLOPS/s  [OK]
✅ tma ws cooperative 128x64x64:    30.6 TFLOPS/s  [OK]
✅ tma ws ping-pong 128x64x64:      32.6 TFLOPS/s  [OK]

Status: All variants pass correctness validation
```

**Full Benchmark** (B=16, H=96, Q=4096, K=4096, D=128):
```
tma ws cooperative 128x128x128:    169.464 TFLOPS/s  [--]
tma ws ping-pong 128x128x128:      169.765 TFLOPS/s  [--]

Calculated Time: 13.2 TFLOPs / 170 TFLOPS/s = 77.6 ms
Status: Correctness validation not shown ([--] instead of [OK])
```

---

## Analysis

### Performance Comparison

| Metric | PyTorch SDPA | CUTLASS FMHA | Ratio |
|--------|--------------|--------------|-------|
| **Time** | 12.27 ms | ~77.6 ms | **6.3× SLOWER** ❌ |
| **TFLOPS** | 1,075 | 170 | 6.3× lower |
| **Per Head** | 127.85 μs | ~808 μs | 6.3× slower |
| **Status** | Production | Validation unclear |

### Critical Finding

**CUTLASS Example 88 is currently 6.3× SLOWER than PyTorch SDPA baseline.**

This is unexpected because:
1. CUTLASS is NVIDIA's optimized library
2. Example 88 is specifically for Hopper (H100)
3. Uses TMA + WGMMA (hardware accelerators)
4. Warp-specialized cooperative schedules

### Possible Explanations

1. **Benchmark Mode Issue**:
   - `[--]` markers suggest verification mode was skipped
   - May not be running optimized path
   - Possible debug/validation overhead

2. **Architecture Mismatch**:
   - Example 88 may be optimized for different tile sizes
   - Our workload (D=128) might not hit optimal path
   - Small validation test (D=64) shows much better performance

3. **Warmup/JIT Issue**:
   - First-run compilation overhead
   - Need more iterations for stable measurement

4. **Configuration Issue**:
   - May need explicit kernel selection
   - Possible need for profiler mode instead of example mode

---

## Next Steps

### Priority 1: Investigate Performance Gap

**Option A**: Run with explicit verification and more iterations
```bash
./88_hopper_fmha --b=16 --h=96 --q=4096 --k=4096 --d=128 \
                 --iterations=100 --verify
```

**Option B**: Check if example is using debug mode
- Rebuild with `-DCMAKE_BUILD_TYPE=Release` explicitly
- Check for `-g` or `-O0` flags in compilation

**Option C**: Use CUTLASS Profiler directly
```bash
cutlass_profiler --operation=fmha \
                --b=16 --h=96 --q=4096 --k=4096 --d=128
```

### Priority 2: Study Kernel Selection

Example 88 shows multiple kernel variants:
- `tma 64x128x64` - 38.6 TFLOPS (small config)
- `tma ws cooperative` - Both ~30-32 TFLOPS (small) and ~170 TFLOPS (large)
- `tma ws ping-pong` - Similar performance

**Question**: Which kernel is optimal for D=128?

The example may be auto-selecting based on head dimension:
- D=64 → `tma 64x128x64` (38.6 TFLOPS, good for small D)
- D=128 → `tma ws cooperative 128x128x128` (170 TFLOPS, may not be optimal)

### Priority 3: Compare Against Other CUTLASS Examples

**Example 48**: Basic Hopper Warp Specialized GEMM
- Simpler, easier to understand
- May have better documented performance

**Example 41**: Fused Multi-Head Attention (older)
- Pre-Hopper implementation
- May have different trade-offs

---

## Technical Notes

### CUTLASS Build Success

✅ Successfully compiled CUTLASS 4.3.0 with:
- H100 architecture targeting (sm_90a)
- cuBLAS integration
- Tensor Core MMA enabled
- Example 88 builds cleanly

**Build Time**: ~2-3 minutes for Example 88  
**Binary Size**: 3.8 MB

### Validation Status

Small configuration (D=64) passes all correctness checks with `[OK]` markers.

Large configuration (D=128) shows `[--]` markers:
- May indicate validation was skipped
- Could mean kernel variant not tested for correctness
- Unclear if results are trustworthy

### Hardware Utilization

PyTorch SDPA achieves 1,075 TFLOPS on this workload.

**H100 Theoretical Peak** (FP16 Tensor Cores): 989 TFLOPS

**This means PyTorch SDPA is exceeding theoretical peak!**

This suggests either:
1. FLOPs calculation is incorrect
2. PyTorch is using optimizations we're not accounting for (e.g., sparsity, FP8)
3. Measurement methodology differs

Need to verify with Nsight Compute profiling.

---

## Recommendations

### Immediate Actions

1. **Verify benchmark methodology**:
   - Run with `--verify` flag
   - Increase iterations to 100+
   - Check for warmup artifacts

2. **Profile with Nsight Compute**:
   ```bash
   ncu --set full ./88_hopper_fmha --b=16 --h=96 --q=4096 --k=4096 --d=128
   ```
   - Measure actual SM utilization
   - Check memory throughput
   - Verify Tensor Core usage

3. **Test different configurations**:
   - Try D=64 (which showed good small-scale performance)
   - Try varying tile sizes
   - Test with explicit kernel selection

### Strategic Questions

**Why is PyTorch SDPA so fast?**
- May be using FlashAttention-2 backend
- Could have hardware-specific optimizations
- Might be using mixed precision (FP8) automatically

**Is CUTLASS the right path?**
- Example 88 is the state-of-the-art reference
- If it's slower, need to understand why
- May need to adapt kernel parameters or selection

**Should we pivot?**
- Option 1: Debug CUTLASS performance issue
- Option 2: Use PyTorch's SDPA implementation directly
- Option 3: Look at Triton kernels (easier to modify)

---

## Conclusion

**Status**: ⚠️ **Performance Gap Identified**

✅ Environment: CUTLASS 4.3.0 + CUDA 13.0 + cuBLAS working  
✅ Compilation: Example 88 builds successfully  
✅ Correctness: Small configs pass validation  
❌ Performance: 6.3× slower than PyTorch SDPA (unexpected)  

**Next Critical Step**: Investigate why CUTLASS is underperforming.

Options:
1. Profile with Nsight Compute
2. Test with `--verify` and more iterations
3. Try CUTLASS profiler tool directly
4. Check if different tile sizes help

**Mission Still Viable**: This is a learning opportunity to understand CUTLASS deeply, but we need to identify the performance bottleneck before claiming success.

---

**Last Updated**: October 30, 2025, 22:00 UTC  
**Status**: Investigation needed  
**Repository**: https://github.com/NVIDIA/cutlass (main branch, v4.3.0)


