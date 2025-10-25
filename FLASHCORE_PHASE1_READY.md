# FlashCore Phase 1 Ready - Optimization Roadmap

**Date**: October 22, 2025, 20:00 PST  
**Status**: 🚀 **READY FOR PHASE 1** - Both kernels validated, baseline established  
**Next**: Fuse softmax (Phase 1.1)

---

## ✅ Validation Complete

### Correctness (L4 GPU)
```
QK^T: Max error 0.001948 ✅ PASS
P·V:  Max error 0.000000 ✅ PASS (PERFECT!)
```

**Both kernels are CORRECT and ready for optimization!**

### Performance Baseline (L4 GPU)
```
QK^T:  141.54 μs
P·V:    57.01 μs
Total: 198.54 μs (unfused)

PyTorch SDPA: 22.84 μs (reference)
Gap: 8.69× slower (expected for unfused)
```

**Baseline established - ready to optimize!**

---

## Phase 1: CUDA Optimization Roadmap (2-4 hours)

### 1.1 Fuse Softmax (1-2 hours) ← **START HERE**

**Current architecture:**
```
QK^T → write S (F32, [B,H,S,S]) → GPU memory
     ↓
Read S → Softmax → write P (F16, [B,H,S,S]) → GPU memory
     ↓
Read P → P·V → write O (F16, [B,H,S,D])
```

**Target architecture:**
```
QK^T → Softmax (in-register/shared) → P·V → write O (F16, [B,H,S,D])
        ↑ No intermediate GPU memory writes!
```

**Expected speedup**: 198 μs → 80-100 μs (2× improvement)

**Implementation approach:**
```cuda
// Fused kernel combining v6 QK^T + softmax + v7.1 P·V
__global__ void flashcore_fused_attention(
    const half* Q, const half* K, const half* V,
    half* O,
    int B, int H, int S, int D,
    float scale) {
    
    // Phase 1: Compute QK^T into shared memory (reuse from v6)
    __shared__ float S_tile[kTileM][kTileN];  // Scores
    compute_qkt_wmma(..., S_tile, scale);
    __syncthreads();
    
    // Phase 2: Softmax in-place on S_tile
    // - Compute max per row (warp reduce)
    // - Compute exp and sum per row
    // - Normalize and convert to FP16
    __shared__ half P_tile[kTileM][kTileN];  // Probabilities
    fused_softmax(S_tile, P_tile);
    __syncthreads();
    
    // Phase 3: Compute P·V (reuse from v7.1)
    compute_pv_wmma(P_tile, V, O);
}
```

**Files to modify:**
- Create: `flashcore/flashcore_fused.cu`
- Update: `flashcore/build_wmma.py` (add fused kernel)
- Update: `flashcore/test_wmma.py` (add fused test)

**Validation criteria:**
- ✅ Correctness: Max error < 0.05
- ✅ Performance: 80-100 μs (2× speedup vs 198 μs)
- ✅ PTXAS: No register spills, < 64KB SMEM

---

### 1.2 Profile with NCU (30 min)

**Command:**
```bash
ncu --set full \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
             dram__throughput.avg.pct_of_peak_sustained_elapsed,\
             smsp__average_warps_active.avg.pct_of_peak_sustained_active \
    python3 test_wmma.py > ncu_fused.txt
```

**Look for:**
- **TC Utilization**: Should be >60% (WMMA active)
- **DRAM Throughput**: Should be <50% (memory-bound indicator)
- **Warp Occupancy**: Should be >70% (parallelism)
- **Stall Reasons**: Memory vs compute bound

**Output:** `FLASHCORE_NCU_ANALYSIS.md` with bottleneck identification

---

### 1.3 Tile Size Tuning (1 hour)

**Test configurations:**
```cuda
// Option A: Current (baseline)
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileD = 64;

// Option B: Larger tiles (more compute per thread block)
constexpr int kTileM = 96;
constexpr int kTileN = 96;
constexpr int kTileD = 96;

// Option C: Maximum (L4 has 99 KB SMEM per SM)
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileD = 128;

// Option D: Asymmetric (optimize for S >> D)
constexpr int kTileM = 64;
constexpr int kTileN = 128;
constexpr int kTileD = 64;
```

**Metrics to track:**
| Config | Latency (μs) | Occupancy | SMEM (KB) | Registers | Notes |
|--------|--------------|-----------|-----------|-----------|-------|
| 64×64  | TBD          | TBD       | ~48       | ~64       | Baseline |
| 96×96  | TBD          | TBD       | ~72       | ~96       | More compute |
| 128×128| TBD          | TBD       | ~99       | ~128      | Max tiles |
| 64×128 | TBD          | TBD       | ~56       | ~72       | Asymmetric |

**Selection criteria:**
- Minimize latency (primary)
- Maximize occupancy (>50%)
- Avoid spills (registers < 128)
- Stay within SMEM limit (< 99 KB)

---

### 1.4 Warp Specialization (1 hour)

**Concept:** Overlap data movement with computation

**Current:** All warps do the same work sequentially
```
All warps: cp.async K/V → wait → compute → cp.async next tile → wait → compute
```

**Target:** Producer/consumer specialization
```
Producer warps (0-7):   cp.async K/V tile N+1 (async, non-blocking)
Consumer warps (8-15):  Compute on K/V tile N (WMMA)
                        ↑ Overlap!
```

**Implementation:**
```cuda
__global__ void attention_warp_specialized(...) {
    const int warp_id = threadIdx.x / 32;
    
    if (warp_id < 8) {
        // Producer: Stage K/V tiles ahead of computation
        for (int tile = warp_id; tile < num_tiles; tile += 8) {
            prefetch_kv_async(tile);
            cp_async_commit();
        }
    } else {
        // Consumer: Compute QK^T + softmax + PV
        const int compute_warp = warp_id - 8;
        for (int tile = 0; tile < num_tiles; ++tile) {
            cp_async_wait<1>();  // Wait for tile N (N+1 staging in parallel)
            compute_attention_tile(tile, compute_warp);
        }
    }
}
```

**Expected speedup:** 80-100 μs → 40-60 μs (hide memory latency)

---

### 1.5 Expected Progress

**After each step:**
```
Baseline (unfused):    198.54 μs  ← Current ✅
After 1.1 (fuse):       80-100 μs  ← 2× speedup
After 1.2 (profile):    Identify bottlenecks
After 1.3 (tiles):      60-80 μs   ← 1.3× speedup
After 1.4 (warp spec):  40-60 μs   ← 1.5× speedup
────────────────────────────────────────────
Target:                 <40 μs     ← Beat PyTorch!
```

**Confidence:** High - each optimization is proven in literature
- Fusion: FlashAttention paper (2× typical)
- Tile tuning: Standard practice (10-30%)
- Warp specialization: Ampere+ feature (20-50%)

---

## Phase 2-5: Rust Integration (Ready to Start After Phase 1)

### Prerequisites
- ✅ CUDA kernels correct
- ✅ Performance baseline measured
- ⏳ <40 μs target achieved (after Phase 1)

### Roadmap
See `FLASHCORE_RUST_INTEGRATION_ROADMAP.md` for complete details:
- Phase 2: FFI bindings (2-3 hours)
- Phase 3: Security & testing (2-3 hours)
- Phase 4: Performance validation (1 hour)
- Phase 5: CI/CD (1-2 hours)

**Total time:** 10-15 hours to production-ready Rust integration

---

## Current Architecture

### Files
```
flashcore/
├── flashcore_unified.cu          ← QK^T + P·V (validated ✅)
├── flashcore_wmma_common.cuh     ← Shared utilities
├── detail/
│   └── cp_async.hpp              ← Async copy primitives
├── build_wmma.py                 ← Build script
├── test_wmma.py                  ← Test suite
└── TODO: flashcore_fused.cu      ← Phase 1.1 (next)
```

### Kernel Specs
```cuda
// QK^T kernel (v6)
- 16 warps (512 threads)
- 64×64×64 tiles
- WMMA 16×16×16 (FP16→FP32)
- cp.async 2-stage
- Vectorized loads (128-bit)
- Output: F32 scores [B,H,S,S]

// P·V kernel (v7.1)  
- 16 warps (512 threads)
- 64×64×64 tiles
- WMMA 16×16×16 (FP16→FP32)
- cp.async 2-stage
- Sequential accumulation
- Output: F16 [B,H,S,D]
```

---

## Next Action Items

### Immediate (Tonight/Tomorrow)
1. **START**: Implement fused softmax (Phase 1.1)
2. Validate correctness (error < 0.05)
3. Measure performance (target: 80-100 μs)

### This Week
1. Complete Phase 1 (all 4 steps)
2. Achieve <40 μs target
3. Document optimizations

### Next Week
1. Begin Rust integration (Phase 2)
2. Security audit (Phase 3)
3. Production prep (Phase 4-5)

---

## Success Criteria

### Technical
- ✅ Correctness: QK^T 0.0019, P·V 0.0000 error
- ✅ Baseline: 198.54 μs measured
- ⏳ Phase 1 complete: <40 μs
- ⏳ Rust integrated: Memory safe host
- ⏳ Production ready: CI/CD + tests

### Business
- ⏳ 10-30% throughput improvement
- ⏳ Cost savings (less GPU time)
- ⏳ Security hardening (Rust + fuzzing)

---

## Resources

### Hardware
- **GPU**: NVIDIA L4 (Ada, sm_89)
- **SMEM**: 99 KB per SM
- **TC**: 242 TFLOPS FP16
- **Bandwidth**: 300 GB/s

### Software
- **CUDA**: 12.2
- **PyTorch**: 2.5.1+cu121
- **Rust**: 1.70+ (for Phase 2)

### References
- FlashAttention-2 paper (fusion techniques)
- NVIDIA Ampere architecture guide (warp specialization)
- CUTLASS library (WMMA patterns)

---

## Conclusion

**Status**: ✅ **READY TO PROCEED**

Both kernels are validated and working correctly. Performance baseline is established. Clear optimization path with proven techniques.

**Next step**: Implement fused softmax (Phase 1.1) to achieve 2× speedup (198 μs → 80-100 μs).

**Timeline**: 2-4 hours to <40 μs, then Rust integration follows.

**Confidence**: High - foundation is solid, optimizations are standard practice.

---

**Let's build it! 🚀**

