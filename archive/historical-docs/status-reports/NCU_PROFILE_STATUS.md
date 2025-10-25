# NCU Profile Status: In Progress

**Date**: Oct 17, 2025  
**Champion**: pytorch_sdpa_efficient (xFormers) @ 33.19 μs  
**Goal**: Identify bottlenecks for targeted optimization

---

## **Current Status**

✅ **NCU Profile Generated**:
- Report: `evidence/ncu_champion_full.ncu-rep`
- Passes: 35
- Kernel profiled: `distribution_elementwise_grid_stride_kernel`

⚠️ **Issue**: Initial profile captured tensor generation kernel, not SDPA kernel

---

## **Next Steps**

1. **Extract all kernel names** from NCU report
2. **Identify SDPA-related kernels** (memory-efficient attention)
3. **Analyze bottlenecks**:
   - SM throughput (compute-bound?)
   - DRAM throughput (memory-bound?)
   - Tensor Core utilization (TC underutilized?)

4. **Plan targeted optimizations** based on findings

---

## **Preliminary Findings**

From NCU quick scan:
- Device: NVIDIA L4 (sm_89, Ada architecture)
- L2 cache: 48 MB
- Max shared memory: 102.4 KB per SM
- Tensor Cores: Available (FP16/BF16)

**Champion baseline**: 33.19 μs  
**Target**: < 5 μs (6.6× speedup needed)

---

**Status**: Analyzing kernels... (in progress)

