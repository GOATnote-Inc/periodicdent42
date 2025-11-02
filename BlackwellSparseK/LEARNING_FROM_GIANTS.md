# Learning from Giants: cuBLAS & CUTLASS 4.3

## Session: November 2, 2025

### Objective
Understand NVIDIA's performance ceiling before optimizing our sparse kernel.

---

## Hardware Ceiling: cuBLAS Performance

### Measured on H100 (sm_90a)

| Size | Time (ms) | TFLOPS | Notes |
|------|-----------|--------|-------|
| 8192³ | 1.791 | **613.8** | Primary target |
| 4096³ | 0.251 | 547.0 | Smaller sizes less efficient |
| 2048³ | 0.037 | 464.9 | Memory bound region |

**Key Insight:** 613.8 TFLOPS is our ceiling for 8192³ FP16→FP32 GEMM.

---

## CUTLASS 4.3 CollectiveBuilder Pattern

### From Example 49 (`/opt/cutlass/examples/49`)

**Type System:**
```cpp
using ElementA = cutlass::half_t;          // Not raw half!
using ElementB = cutlass::half_t;
using ElementAccumulator = float;          // FP32 accumulation
```

**Tile Configuration:**
```cpp
TileShape:    Shape<_128, _128, _64>       // M, N, K
ClusterShape: Shape<_2, _1, _1>            // Thread block clusters
```

**Auto Schedules (The Magic):**
```cpp
using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<...>;
```

**CollectiveBuilder Pattern:**
1. Build epilogue first (needed for shared memory carveout)
2. Build mainloop with carveout calculation
3. Assemble GemmUniversal kernel
4. Wrap in GemmUniversalAdapter

---

## Current Status: Our Sparse Kernel

### Iteration 13 Result (Validated)
- **55.2 TFLOPS** (sparse, correct)
- **WMMA-based** (not using CollectiveBuilder)
- **~9% efficiency** vs cuBLAS (55.2 / 613.8)

### Why Low?
1. Not using TMA (Tensor Memory Accelerator)
2. Not using warp specialization
3. Not using CUTLASS's pipeline optimizations
4. Manual WMMA instead of auto-optimized collectives

---

## Path Forward

### Option A: Pure Custom Kernel
- Continue optimizing WMMA manually
- Add TMA manually
- Target: Maybe 100-150 TFLOPS (16-24% efficiency)
- **Risk:** Reinventing CUTLASS internals

### Option B: Adapt CollectiveBuilder (Recommended)
- Understand Example 49's pattern completely
- Create sparse variant using same infrastructure
- Leverage Auto schedules, TMA, warp specialization
- Target: 300-400 TFLOPS sparse (50-65% efficiency accounting for sparsity overhead)
- **Benefit:** Standing on shoulders of giants

---

## Next Steps

1. **Get Example 49 compiling standalone** - understand stride construction
2. **Profile Example 49 vs cuBLAS** - confirm they match (~600 TFLOPS)
3. **Study CollectiveBuilder source** - understand how Auto works
4. **Design sparse adaptation** - where to inject BSR logic
5. **Implement & validate** - beat current 55.2 TFLOPS

---

## Questions to Answer

1. How does `KernelScheduleAuto` select TMA vs cp.async?
2. Where in CollectiveBuilder can we inject sparse indexing?
3. Can we use CollectiveBuilder per-tile and orchestrate sparse logic outside?
4. What's the theoretical ceiling for 87.5% sparse BSR on H100?

