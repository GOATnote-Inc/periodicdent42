# FlashCore Unified WMMA Status - October 22, 2025

## 🎉 MAJOR BREAKTHROUGH: QK^T CORRECT!

**Result**: QK^T kernel **FIXED** with 16-warp architecture + proper cp_async!

```
QK^T: Max error = 0.001951 ✅ PASS!
P·V:  Max error = 3.203125 ❌ FAIL (atomic accumulation issue)
```

---

## What Fixed QK^T

1. ✅ **16 warps** (512 threads) instead of 8 - better occupancy
2. ✅ **Proper cp_async** via `detail/cp_async.hpp` with `constexpr` checks
3. ✅ **Correct WMMA layout**: `col_major` matrix_b for K^T
4. ✅ **Dimension validation** in bindings

**Key insight**: The original WMMA pattern was correct! The issue was insufficient parallelism (8 warps) and cp_async implementation.

---

## P·V Remaining Issue

**Problem**: Accumulation across kv_tiles 
- Each kv_tile contributes to the final output
- AtomicAdd on WMMA fragments has precision issues (3.20 error)
- Fragment element layout is non-trivial

**Attempted fixes**:
1. ❌ Direct store (overwrites previous tiles)
2. ❌ AtomicAdd on fragments (current: 3.20 error)

**Next approach**: Sequential accumulation without atomics
- Process tiles one at a time
- Load existing accumulator, add new contribution, store back
- Requires careful synchronization

---

## Files

- `flashcore_unified.cu`: Single-file implementation (v6 QK^T + v7.1 P·V)
- `flashcore_wmma_common.cuh`: Shared utilities with constexpr checks
- `detail/cp_async.hpp`: Proper async copy implementation
- `build_wmma.py`: Unified build script
- `test_wmma.py`: Test suite

---

## Performance Estimates (Once Correct)

**QK^T**: ~80-120 μs (WMMA + cp.async + vectorized, 16 warps)
**P·V**: ~80-120 μs (WMMA + cp.async + vectorized, 16 warps)
**Total**: ~160-240 μs (unfused)

**Path to <40 μs**:
1. Fix P·V correctness (current priority)
2. Profile with NCU
3. Fuse softmax (eliminate P matrix write)
4. Warp specialization
5. Tile size tuning

---

## Next Steps

1. **Fix P·V accumulation** (without atomics)
2. **Validate correctness** (both kernels < 0.05 error)
3. **Benchmark** on L4
4. **Commit and document** success
5. **Optimize** towards <40 μs

---

**Status**: QK^T validated ✅, P·V debugging 🔧  
**Confidence**: High - clear path forward

