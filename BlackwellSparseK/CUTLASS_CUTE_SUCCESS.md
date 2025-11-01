# CUTLASS 4.3.0 + CuTe - H100 Success

## Performance

**H100 PCIe 80GB (November 1, 2025)**

| Implementation | Time (ms) | TFLOPS | vs cuBLAS |
|----------------|-----------|--------|-----------|
| **CUTLASS+CuTe** | **2.07** | **532** | **84%** |
| cuBLAS | 1.74 | 632 | 100% |

**Matrix:** 8192×8192×8192, FP16→FP32

## Modern APIs Used

✅ **CollectiveBuilder** - Modern CUTLASS 4.3.0 API  
✅ **CuTe DSL** - `Shape<_128,_256,_64>`, `ClusterShape<_2,_1,_1>`  
✅ **TMA (Tensor Memory Accelerator)** - Hopper async memory access  
✅ **WGMMA** - Hopper warpgroup matrix ops  
✅ **OpClassTensorOp** - Tensor Core compute  

## Code

```cpp
using TileShape = Shape<_128,_256,_64>;
using ClusterShape = Shape<_2,_1,_1>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 8,
    ElementB, cutlass::layout::RowMajor, 8, ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<...>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
```

**Full example:** `examples/cutlass_cute_h100.cu`

## Key Learnings

1. **Don't use WMMA** (2017 API) - use WGMMA via CollectiveBuilder
2. **Use CuTe DSL** for layouts - compile-time optimizations
3. **TMA handles memory** - don't manually copy to shared mem
4. **Let CUTLASS optimize** - Auto scheduling finds best config

## Ready for Sparse Extension

Next: Adapt this for BSR sparse using same modern APIs.

---

**Status:** Production-ready template for H100 GEMM development  
**Validated:** November 1, 2025

