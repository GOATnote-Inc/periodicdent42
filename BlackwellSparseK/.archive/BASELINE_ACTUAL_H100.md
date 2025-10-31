# H100 Baseline - ACTUAL MEASURED (Oct 30, 2025)

**Method:** CUDA Events (cudaEventElapsedTime)  
**Hardware:** H100 80GB HBM3 (sm_90a)  
**Kernel:** BSR sparse GEMM, cooperative loads, WMMA Tensor Cores  
**Config:** M=8192, N=8192, K=8192, BM=128, BN=128, BK=32, topk=16 blocks/row

---

## 📊 Performance (10 trials)

| Metric        | Value      | Notes                    |
| :------------ | :--------- | :----------------------- |
| **Latency**   | **0.619 ms** | Mean (617-621 ms range)  |
| **TFLOPS**    | **111.0**    | Sustained compute        |
| **CV**        | **0.3%**     | Excellent repeatability  |

### Trial Data

```
Trial 1:  0.618 ms,  111.3 TFLOPS
Trial 2:  0.621 ms,  110.7 TFLOPS
Trial 3:  0.619 ms,  110.9 TFLOPS
Trial 4:  0.618 ms,  111.2 TFLOPS
Trial 5:  0.621 ms,  110.6 TFLOPS
Trial 6:  0.617 ms,  111.4 TFLOPS  ← Best
Trial 7:  0.620 ms,  110.9 TFLOPS
Trial 8:  0.620 ms,  110.8 TFLOPS
Trial 9:  0.621 ms,  110.7 TFLOPS
Trial 10: 0.619 ms,  111.0 TFLOPS
```

**Statistics:**
- p50: 0.619 ms
- p99: 0.621 ms
- Std Dev: 0.0013 ms
- CV: 0.21% (sub-1% jitter ✅)

---

## 🎯 Kernel Architecture

**Memory Access:**
- Cooperative thread loads (all threads participate)
- A: [128×32] row-major → shared memory row-major
- B: [32×128] row-major → shared memory column-major (transpose)
- Synchronization: `__syncthreads()` per K-block

**Compute:**
- WMMA Tensor Cores: FP16 inputs → FP32 accumulators
- Warp tiles: 64×64
- CTA tiles: 128×128
- K-slice: 32 (multiple of 16 for WMMA)

**Compilation:**
- Registers: 168
- Barriers: 1
- Spills: 0 ✅
- Binary: 1.1M

---

## 🚀 Next Steps

### Immediate: Use CUTLASS Tools for TMA

**CUTLASS provides the TMA infrastructure - we USE it, not reimplement:**

1. **Study CUTLASS Example 48** (Hopper Warp-Specialized GEMM)
   - File: `/opt/cutlass/examples/48_hopper_warp_specialized_gemm/`
   - Uses: `CollectiveBuilder` with `KernelScheduleAuto`
   - Automatic TMA + pipeline generation

2. **Use CUTLASS CollectiveBuilder**
   ```cpp
   using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
       ArchTag,  // Sm90
       OperatorClass,  // OpClassTensorOp
       ElementA, LayoutA, AlignmentA,
       ElementB, LayoutB, AlignmentB,
       ElementAccumulator,
       TileShape, ClusterShape,
       StageCountAutoCarveout<...>,
       KernelScheduleAuto  // ← Chooses TMA automatically
   >::CollectiveOp;
   ```

3. **Let CUTLASS handle:**
   - TMA descriptor creation
   - Pipeline state management
   - Memory barriers
   - Producer/consumer coordination

### Target Performance

| Optimization | Target Latency | Target TFLOPS | Speedup |
| :----------- | :------------- | :------------ | :------ |
| **Baseline** | 0.619 ms       | 111           | 1.0×    |
| TMA          | 0.400 ms       | 172           | 1.5×    |
| Tuned        | 0.310 ms       | 222           | 2.0×    |

**Target:** 2× speedup (0.31 ms) using CUTLASS's built-in optimizations

---

## 📦 Deliverables

✅ **Working baseline kernel**  
✅ **CUDA Events timing (no Nsight required)**  
✅ **Actual performance numbers**  
✅ **Repeatability verified (CV < 1%)**  
⬜ Use CUTLASS CollectiveBuilder for TMA  
⬜ Profile with Nsight Compute (when permissions available)  
⬜ xFormers integration  

---

**Status:** Baseline established with real H100 measurements  
**Next:** Integrate CUTLASS Collective API for automatic TMA optimization

