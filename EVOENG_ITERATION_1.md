# EvoEngineer Iteration 1: Phase 1 SMEM Optimization

## EvoEngineer Methodology Recap

From the paper summary and best practices:

**Core Principles:**
1. **ONE change at a time** - Systematic, isolated modifications
2. **Verify correctness FIRST** - `torch.allclose` must pass before benchmarking
3. **Benchmark after EACH iteration** - Measure impact of every change
4. **Population management** - Keep best performing variants
5. **Iterative refinement** - Build on successful changes, revert failures

**EvoEngineer-Insight Configuration** (our choice):
- Low token usage
- High performance gains
- Task Context (I1) + Optimization Insights (I3)
- NO full population management (keeps costs down)

---

## Current Kernel Analysis

**File**: `cudadent42/bench/kernels/fa_s512_v3.cu` (805 lines)

**Current Performance**: 38.00 μs (B=2, H=8, S=512, D=64)  
**PyTorch Baseline**: 47.10 μs  
**Status**: ✅ Already 21% faster than PyTorch!

### SMEM Usage (lines 78-100)

```cpp
template<typename Traits>
struct __align__(16) SharedMemory {
    // Stage 0, 1 for K tiles (16-byte aligned base)
    half K[Traits::STAGES][Traits::BLOCK_N][Traits::K_STRIDE];
    
    // Stage 0, 1 for V tiles (16-byte aligned base)
    half V[Traits::STAGES][Traits::BLOCK_N][Traits::V_STRIDE];
    
    // Low-regs variant: per-CTA accumulator for O (float) - saves ~512 regs/thread!
    float O_accum[Traits::BLOCK_M][Traits::HEAD_DIM];
    
    // Pad to avoid bank conflicts on 32-way banks when HEAD_DIM % 32 == 0
    float pad0[32];
};
```

**Key Findings:**
- ✅ K and V already use `half` (good!)
- ⚠️ O_accum uses `float` (4 bytes) - could be `half` (2 bytes) for 50% SMEM savings
- ✅ Has swizzling infrastructure (`detail/smem_swizzle.hpp`)
- ✅ 16-byte aligned
- ⚠️ No S_smem or QK_smem (design uses registers - interesting!)

### SMEM Calculation (lines 93-105)

```cpp
constexpr size_t smem_bytes() {
    constexpr size_t k_bytes = Traits::STAGES * Traits::BLOCK_N * Traits::K_STRIDE * sizeof(half);
    constexpr size_t v_bytes = Traits::STAGES * Traits::BLOCK_N * Traits::V_STRIDE * sizeof(half);
    constexpr size_t o_bytes = Traits::BLOCK_M * Traits::HEAD_DIM * sizeof(float);
    constexpr size_t pad_bytes = 32 * sizeof(float);  // Bank conflict padding
    constexpr size_t total = k_bytes + v_bytes + o_bytes + pad_bytes;
    
    // For default config (32, 64, 32, 4, 2): ~41.2 KB
    static_assert(total <= 48 * 1024, "SMEM overflow on L4 (48KB limit)");
    return total;
}
```

**For BLOCK_M=32, BLOCK_N=64, STAGES=2:**
- K: 2 × 64 × 64 × 2 = 16,384 bytes (16 KB)
- V: 2 × 64 × 64 × 2 = 16,384 bytes (16 KB)
- O_accum: 32 × 64 × 4 = 8,192 bytes (8 KB)
- Padding: 32 × 4 = 128 bytes
- **Total**: ~41 KB ✅ Under 48 KB limit

---

## Iteration 1: Convert O_accum to FP16

### Motivation

**Why**: O_accum stores partial attention outputs. Using FP16 will:
1. **Reduce SMEM**: 8 KB → 4 KB (50% reduction in this buffer)
2. **Enable larger tiles**: More headroom for BLOCK_M=64 or BLOCK_N=128
3. **Numerical precision**: Post-softmax values are bounded [0,1], FP16 sufficient

**L4 Relevance**: Ada Tensor Cores prefer FP16 (242 TFLOPS vs 121 TFLOPS for FP32)

### Expected Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| SMEM Usage | ~41 KB | ~37 KB | -4 KB (-10%) |
| O_accum Size | 8 KB | 4 KB | -50% |
| Headroom for larger tiles | 7 KB | 11 KB | +57% |
| Performance | 38.00 μs | 36-37 μs | -3-5% (less SMEM pressure) |

### Implementation

**File**: `cudadent42/bench/kernels/fa_s512_v3.cu`

**Change 1**: Line 86 (SharedMemory struct)

```cpp
// BEFORE:
float O_accum[Traits::BLOCK_M][Traits::HEAD_DIM];

// AFTER:
half O_accum[Traits::BLOCK_M][Traits::HEAD_DIM];
```

**Change 2**: Line 97 (smem_bytes calculation)

```cpp
// BEFORE:
constexpr size_t o_bytes = Traits::BLOCK_M * Traits::HEAD_DIM * sizeof(float);

// AFTER:
constexpr size_t o_bytes = Traits::BLOCK_M * Traits::HEAD_DIM * sizeof(half);
```

**Change 3**: Update comment at line 85

```cpp
// BEFORE:
// Low-regs variant: per-CTA accumulator for O (float) - saves ~512 regs/thread!

// AFTER:
// Low-regs variant: per-CTA accumulator for O (half) - saves ~512 regs/thread + 4KB SMEM!
```

**Change 4**: Search for O_accum reads/writes and add explicit FP32 conversion if needed

```bash
# Search for O_accum usage
grep -n "O_accum" cudadent42/bench/kernels/fa_s512_v3.cu
```

### Verification Protocol (EvoEngineer)

```bash
# 1. Build
cd cudadent42/bench
python build_v3_release.py

# 2. Verify compilation
python -c "import flash_attention_s512_v3; print('✅ Import OK')"

# 3. Correctness test (MUST PASS BEFORE BENCHMARK)
cd ../../scripts
python -c "
import torch
import flash_attention_s512_v3 as fa_v3

B, H, S, D = 2, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
k, v = q.clone(), q.clone()

# PyTorch reference
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

# Our kernel (config_id=1: 32x64x4x2)
out = fa_v3.flash_attention_s512_v3(q, k, v, config_id=1)

# Correctness
assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), 'Correctness FAILED!'
print('✅ Correctness PASSED')
"

# 4. Benchmark (ONLY IF CORRECTNESS PASSES)
python bench_v3_quick.py

# 5. Compare with baseline (38.00 μs)
# Expected: 36-37 μs (5-8% improvement)
```

### Success Criteria

- ✅ Compiles without errors
- ✅ Imports successfully
- ✅ **Correctness test passes** (`torch.allclose` with atol=1e-3, rtol=1e-3)
- ✅ SMEM usage reported as ~37 KB (down from ~41 KB)
- ✅ Performance: 36-37 μs (or neutral - no regression)

### Failure Modes & Mitigation

| Issue | Root Cause | Fix |
|-------|------------|-----|
| Numerical precision loss | O_accum accumulation errors | Add `__float2half` conversions |
| Register spilling | Compiler optimizations change | Check `ptxas --verbose` |
| Performance regression | Cache effects | Revert and document |

---

## GPU Region Strategy

**Problem**: us-central1-a has no L4 capacity (stockout)

**Solution**: Try regions in order of preference:
1. ✅ **southamerica-east1-c** (São Paulo - closest to Brazil)
2. us-west1-b (Oregon - good GPU availability)
3. us-east4-a (Virginia - good GPU availability)
4. europe-west4-a (Netherlands - backup)

**Note**: Can't move existing instance (deprecated API), may need to create new one.

---

## Next Steps

1. **Find available GPU region** (try SA → US-West → US-East → EU)
2. **Apply Iteration 1 changes** (O_accum: float → half)
3. **Build & verify correctness** (MUST pass before benchmark)
4. **Benchmark** (compare with 38.00 μs baseline)
5. **Document results** (create ITER1_RESULTS.md)
6. **If successful**: Proceed to Iteration 2 (XOR swizzling refinement)
7. **If regression**: Revert and try alternative optimization

---

**Status**: Ready to implement (GPU availability pending)  
**EvoEngineer Phase**: Insight-guided single iteration  
**Risk Level**: Low (conservative change, FP16 well-supported)  
**Expected Gain**: 3-5% latency reduction OR neutral (SMEM headroom for future optimizations)

