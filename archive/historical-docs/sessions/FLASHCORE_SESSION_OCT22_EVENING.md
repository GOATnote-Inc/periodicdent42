# FlashCore WMMA Session Summary - October 22, 2025 (Evening)

**Duration**: ~4 hours  
**Status**: 🎉 **MAJOR BREAKTHROUGH** - QK^T kernel validated!  
**Next**: Fix P·V accumulation

---

## 🏆 Key Achievement

**QK^T kernel achieves CORRECTNESS**: Max error **0.00195** (target: <0.05) ✅

This is a **massive** improvement from the 11 billion error we started with!

---

## The Journey

### Starting Point
- Unified WMMA architecture (v6 QK^T + v7.1 P·V)
- 8 warps (256 threads)
- Custom cp_async implementation
- **Result**: QK^T 11B error ❌, P·V inf error ❌

### Debugging Iterations

**Attempt 1**: Changed WMMA layout  
- Tried `row_major` matrix_b → Made it worse
- **Learning**: Original `col_major` was correct!

**Attempt 2**: Fixed cp_async  
- Changed to proper `constexpr` template approach
- Added `detail/cp_async.hpp` with correct PTX inline asm
- **Result**: Compilation fixed, but still high error

**Attempt 3**: 🎯 **BREAKTHROUGH** - 16 warps!  
- Increased from 8 → 16 warps (256 → 512 threads)
- Better occupancy and parallelism
- **Result**: QK^T **0.00195 error** ✅ PASS!

---

## What Fixed QK^T

### 1. **16 Warps (512 Threads)**
```cuda
constexpr int kWarpsPerBlock = 16;  // Was 8
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 512
```

**Impact**: 2× parallelism = better hardware utilization on L4

### 2. **Proper cp_async Implementation**
```cpp
// detail/cp_async.hpp
template <int BYTES>
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n" ::
            "r"(smem_addr), "l"(src), "n"(BYTES));
}
```

**Impact**: Correct async memory transfers with compile-time size check

### 3. **Constexpr Checks**
```cpp
namespace flashcore {
constexpr bool kCpAsyncSupported = FLASHCORE_CP_ASYNC_SUPPORTED != 0;

__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    if constexpr (kCpAsyncSupported) {
        detail::cp_async_cg<16>(dst, src);
    } else {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    }
}
}
```

**Impact**: Zero-cost abstraction, optimal code generation

### 4. **Dimension Validation**
```cpp
TORCH_CHECK(D == flashcore::v6::kTileK,
            "Head dimension must be 64 for FlashCore WMMA QK^T");
```

**Impact**: Clear error messages, prevents silent failures

---

## P·V Status

### Current Issue
**Accumulation across kv_tiles** - each tile contributes to the same output

**Attempted**: AtomicAdd on WMMA fragments
```cuda
// Current (doesn't work well)
for (int i = 0; i < c_frag.num_elements; ++i) {
    int row = i / kWmmaN;
    int col = i % kWmmaN;
    atomicAdd(&dst[row * kTileD + col], c_frag.x[i]);
}
```

**Result**: 3.20 error (worse than before)

**Root cause**: WMMA fragment layout is complex, simple row/col indexing doesn't match actual element positions

### Next Approaches

**Option A**: Sequential accumulation
```cuda
// Pseudo-code
for each tile:
    compute P_tile @ V_tile → temp_frag
    load existing accum → accum_frag
    add temp_frag to accum_frag
    store accum_frag back
```

**Option B**: Separate buffers
```cuda
// Store each tile separately
float tile_results[kv_tiles][M][D];
for each tile:
    compute → tile_results[tile]
// Sum at end
for each position:
    output[m][d] = sum(tile_results[:][m][d])
```

**Option C**: Rethink tiling
- Process all K/V in one WMMA pass (larger tiles)
- Eliminates need for accumulation

---

## Files Created/Modified

### New Files
```
flashcore/flashcore_unified.cu          (501 lines) - Single-file implementation
flashcore/detail/cp_async.hpp           (32 lines)  - Proper async copy
FLASHCORE_UNIFIED_STATUS.md             - Progress tracking
FLASHCORE_SESSION_OCT22_EVENING.md      - This document
```

### Modified Files
```
flashcore/flashcore_wmma_common.cuh     - Added constexpr checks
flashcore/build_wmma.py                 - Updated to use unified.cu
```

---

## Metrics

### Correctness (on L4 GPU)
```
QK^T: Max error = 0.001951 ✅ PASS
P·V:  Max error = 3.203125 ❌ FAIL
```

### Build
```
Compilation: ✅ Success (sm_89, 16 warps, cp.async)
Warnings: 0
Errors: 0
```

### Git
```
Commits: 2 (89278f0, 70035ed)
Files changed: 12
Lines added: 1,686
Lines removed: 34
```

---

## Performance Estimates (Once P·V Fixed)

### Individual Kernels
```
QK^T: ~80-120 μs (WMMA + cp.async + vectorized + 16 warps)
P·V:  ~80-120 μs (same optimizations)
Total: ~160-240 μs (unfused)
```

### Path to <40 μs
1. **Fix P·V** (current blocker)
2. **Benchmark** on L4
3. **Profile** with NCU (identify bottlenecks)
4. **Fuse softmax** (eliminate P matrix, ~50% speedup)
5. **Warp specialization** (producer/consumer overlap)
6. **Tile tuning** (128×128 vs 64×64)
7. **Final polish** (instruction-level opts)

**Confidence**: High - QK^T proves architecture works!

---

## Lessons Learned

### 1. **Occupancy Matters**
Doubling warps from 8 → 16 fixed QK^T completely. L4 has enough resources for high parallelism.

### 2. **Constexpr is Powerful**
Using `if constexpr` eliminates runtime branches and generates optimal code for both Ampere/Ada and older architectures.

### 3. **WMMA is Tricky**
Fragment layouts are non-trivial. Can't assume simple row-major indexing. Need to use WMMA store operations properly.

### 4. **Incremental Progress**
- v1: Massive errors (11B)
- v2: Still massive
- v3: Breakthrough! (0.00195)

Persistence pays off!

### 5. **Hardware Validation is Key**
Testing on real L4 GPU revealed issues that local testing might miss.

---

## Next Session Plan

### Immediate (30-60 min)
1. Implement sequential accumulation for P·V
2. Test on L4
3. Validate correctness (both kernels < 0.05)

### Short-term (2-4 hours)
1. Benchmark unfused performance
2. Profile with NCU
3. Document baseline metrics

### Medium-term (4-8 hours)
1. Implement fused softmax
2. Optimize towards <100 μs
3. Final push to <40 μs

---

## Code Snapshot

### Unified Architecture
```cuda
namespace flashcore {
namespace v6 {  // QK^T
    constexpr int kTileM = 64;
    constexpr int kTileN = 64;
    constexpr int kTileK = 64;
    constexpr int kWarpsPerBlock = 16;  // KEY: 16 warps!
    // ...
}

namespace v7_1 {  // P·V
    constexpr int kTileM = 64;
    constexpr int kTileN = 64;
    constexpr int kTileD = 64;
    constexpr int kWarpsPerBlock = 16;  // Match QK^T
    // ...
}
}
```

### cp_async Pattern
```cpp
// Compile-time dispatch
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    if constexpr (kCpAsyncSupported) {
        detail::cp_async_cg<16>(dst, src);  // Ada/Ampere
    } else {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);  // Fallback
    }
}
```

---

## Resources Used

### Compute
- **GPU**: NVIDIA L4 (Ada, sm_89) via GCP
- **Time**: ~20 minutes GPU time (compilation + testing)
- **Cost**: ~$0.30

### Development
- **Iterations**: ~15 build/test cycles
- **Files deployed**: 7 (via gcloud scp)
- **Commits**: 2

---

## Acknowledgments

- **Reference**: `flashcore_phase1_proven_wmma.cu` - proved WMMA pattern was correct
- **User guidance**: Provided unified implementation template with correct patterns
- **PyTorch**: Excellent C++ extension framework for rapid prototyping

---

## Status Summary

✅ **QK^T**: VALIDATED (0.00195 error)  
🔧 **P·V**: Debugging accumulation (3.20 error)  
📊 **Performance**: TBD (pending P·V fix)  
🎯 **Target**: <40 μs (beat PyTorch SDPA 43 μs)

**Next step**: Fix P·V accumulation pattern  
**ETA**: 30-60 minutes for correctness, then benchmark

---

**Session end**: October 22, 2025, 19:00 PST  
**Next session**: Continue with P·V fix

🎉 **Major milestone achieved: QK^T kernel working!** 🎉

