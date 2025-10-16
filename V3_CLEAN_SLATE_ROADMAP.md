# V3 Clean Slate Roadmap - October 16, 2025

## Executive Summary

**Objective**: Build a correct, performant FlashAttention S=512 kernel for L4 (sm_89) from scratch  
**Approach**: Scalar-only implementation → Correctness first → Incremental optimization  
**Timeline**: 1-2 weeks to working baseline, then systematic optimization  
**Philosophy**: **Measure twice, cut once. Correctness is non-negotiable.**

---

## Why Clean Slate?

### Lessons from Failed V3
1. **No working baseline**: All commits in V3 history were fundamentally broken
2. **Premature optimization**: WMMA/Tensor Cores added before scalar correctness
3. **Big-bang changes**: 64×64 tile integration changed 10+ things simultaneously
4. **Missing correctness gates**: "CORRECTNESS ACHIEVED" commits never validated on hardware
5. **Compiler warnings ignored**: WMMA local memory warnings were deployment blockers

### Clean Slate Principles
1. ✅ **Correctness first, performance second**
2. ✅ **One change at a time, test after each**
3. ✅ **Establish baseline, then optimize incrementally**
4. ✅ **Treat compiler warnings as hard errors**
5. ✅ **Validate on hardware at every step**

---

## Phase 0: Baseline Verification (Day 0, 30 minutes) 🚨 CRITICAL

### Why This Matters
The "48μs PyTorch baseline" from earlier may be incorrect (cold start, wrong backend, etc.).  
**True baseline could be 5-10μs** (properly optimized) which changes all targets by 10×.

### Step 0.1: Verify PyTorch SDPA Baseline
**File**: `scripts/verify_sdpa_baseline_l4.py`

**Implementation**:
```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, S, D = 2, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda').contiguous()
k, v = q.clone(), q.clone()

# Warm up (10 iters) - exclude compilation
for _ in range(10):
    _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
torch.cuda.synchronize()

# Force Flash Attention backend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(100):
        start.record()
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # Convert to μs
    
    p50 = sorted(times)[50]
    print(f"PyTorch SDPA p50: {p50:.2f} μs")
    print(f"Expected range: 5-10 μs (optimal) or 20-50 μs (suboptimal)")
    
    if p50 < 15:
        print("🚨 TRUE BASELINE: 5-10μs → Phase 1 target: 20-40μs")
    else:
        print("✓ BASELINE CONFIRMED: 20-50μs → Phase 1 target: 100-200μs")
```

**Gate**: Record true baseline. All subsequent targets scale from this.

---

## Phase 1: Scalar FlashAttention Baseline (Week 1)

### Goal
Working scalar-only FlashAttention kernel that:
- ✅ Passes correctness tests (oracle, parity with PyTorch SDPA)
- ✅ Compiles without warnings
- ✅ Achieves **2-4× slower than verified PyTorch baseline**
- ✅ Uses only scalar operations (no WMMA, no vectorization yet)

### Step 1.1: Minimal Kernel Skeleton (Day 1)
**File**: `cudadent42/bench/kernels/fa_s512_v3_scalar.cu`

**Implementation**:
```cpp
// Simplest possible FlashAttention for S=512, D=64
// ONE tile at a time, NO optimizations, NO fancy memory tricks
// Goal: CORRECTNESS ONLY

__global__ void flash_attention_s512_scalar(
    const half* Q,  // [B, H, S, D]
    const half* K,  // [B, H, S, D]
    const half* V,  // [B, H, S, D]
    half* O,        // [B, H, S, D]
    float scale,
    int B, int H, int S,
    bool is_causal
) {
    // Launch: B*H blocks, each block processes one (batch, head)
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh % H;
    
    // Each thread processes one query row
    const int qid = threadIdx.x;  // row in [0, S)
    if (qid >= S) return;
    
    // Load Q row into registers (64 floats)
    float q_row[64];
    for (int d = 0; d < 64; d++) {
        int idx = ((b * H + h) * S + qid) * 64 + d;
        q_row[d] = __half2float(Q[idx]);
    }
    
    // Online softmax accumulators
    float m_i = -INFINITY;  // max
    float l_i = 0.0f;       // sum of exp
    float O_acc[64] = {0};  // output accumulator
    
    // Loop over K,V tiles (one row at a time for simplicity)
    for (int kid = 0; kid < S; kid++) {
        // Causal masking
        if (is_causal && kid > qid) break;
        
        // Compute dot product Q · K^T
        float score = 0.0f;
        for (int d = 0; d < 64; d++) {
            int k_idx = ((b * H + h) * S + kid) * 64 + d;
            float k_val = __half2float(K[k_idx]);
            score += q_row[d] * k_val;
        }
        score *= scale;
        
        // Online softmax update
        float m_new = fmaxf(m_i, score);
        float correction = expf(m_i - m_new);
        l_i = l_i * correction + expf(score - m_new);
        
        // Rescale O_acc
        for (int d = 0; d < 64; d++) {
            O_acc[d] *= correction;
        }
        
        // Accumulate V contribution
        float weight = expf(score - m_new);
        for (int d = 0; d < 64; d++) {
            int v_idx = ((b * H + h) * S + kid) * 64 + d;
            float v_val = __half2float(V[v_idx]);
            O_acc[d] += weight * v_val;
        }
        
        m_i = m_new;
    }
    
    // Normalize and write output
    for (int d = 0; d < 64; d++) {
        int o_idx = ((b * H + h) * S + qid) * 64 + d;
        O[o_idx] = __float2half(O_acc[d] / l_i);
    }
}
```

**Success Criteria**:
- ✅ Compiles without warnings
- ✅ Runs without CUDA errors
- ✅ Produces non-NaN outputs
- ⏳ Correctness TBD (next step)

**Time**: 2-3 hours (write + compile + smoke test)

---

### Step 1.2: Correctness Tests (Day 1-2)
**File**: `tests/test_v3_scalar_correctness.py`

**Tests**:
1. **Parity with PyTorch SDPA** (non-causal):
   - Shape: (2, 8, 512, 64)
   - Tolerance: atol=1e-2, rtol=1e-2
   - Success: `torch.allclose` returns True

2. **Parity with PyTorch SDPA** (causal):
   - Same shape, is_causal=True
   - Same tolerance
   - Success: `torch.allclose` returns True

3. **Oracle test** (single tile):
   - Manually computed expected output for tiny input
   - Validates algorithm correctness independent of PyTorch

**Implementation**:
```python
import torch
import torch.nn.functional as F

def test_scalar_parity_noncausal():
    from build_v3_scalar import build_v3_scalar
    m = build_v3_scalar()
    
    B, H, S, D = 2, 8, 512, 64
    q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    k, v = q.clone(), q.clone()
    scale = 1.0 / (D ** 0.5)
    
    ref = F.scaled_dot_product_attention(q, k, v)
    out = m.forward(q, k, v, scale, False)
    
    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), \
        f"max_diff={((out - ref).abs().max()):.6f}"
    print("✓ Non-causal parity test PASSED")

def test_scalar_parity_causal():
    # Same as above, but with is_causal=True
    pass

def test_scalar_oracle():
    # Tiny 2×2 input with hand-computed expected output
    pass
```

**Success Criteria**:
- ✅ All 3 tests pass
- ✅ No NaN outputs
- ✅ max_diff < 0.01 (1% error acceptable for FP16)

**Time**: 3-4 hours (write tests + debug failures)

**Gate**: **DO NOT PROCEED** until all correctness tests pass.

---

### Step 1.3: Performance Baseline (Day 2-3)
**File**: `scripts/bench_v3_scalar_baseline.py`

**Benchmark**:
```python
# Warm-up: 10 iterations
# Benchmark: 100 iterations with CUDA events
# Report: p50, p90, p99, mean, std
```

**Expected Performance**:
- B=2, H=8, S=512, D=64: **100-200μs**
- Comparison: PyTorch SDPA ≈ 48μs (from baseline verification)
- **Gap**: 2-4× slower (expected for unoptimized scalar)

**Success Criteria**:
- ✅ Latency 100-200μs (not 9ms!)
- ✅ No NaN outputs
- ✅ Reproducible (std < 10% of mean)

**Time**: 2 hours (write script + run benchmarks)

**Gate**: If latency > 500μs, investigate before proceeding.

---

### Step 1.4: Clean Build System (Day 3)
**File**: `cudadent42/bench/build_v3_scalar.py`

**Requirements**:
- ✅ Single source of truth for compile flags
- ✅ Release mode: `-O3 -use_fast_math -DNDEBUG`
- ✅ Debug mode: `-g -G -DDEBUG` (for cuda-gdb)
- ✅ No `-DUSE_WMMA` (not ready yet)
- ✅ Clean rebuild on flag changes

**Time**: 1-2 hours

---

### Phase 1 Summary
**Duration**: 3-4 days  
**Deliverables**:
1. ✅ Scalar kernel (`fa_s512_v3_scalar.cu`)
2. ✅ Correctness tests (3 tests, all passing)
3. ✅ Performance baseline (100-200μs)
4. ✅ Build system (`build_v3_scalar.py`)

**Gate**: All tests pass + performance < 500μs → Proceed to Phase 2

---

## Phase 2: Memory Optimizations (Week 2)

### Goal
Reduce latency from 100-200μs → 50-80μs using scalar optimizations only.

### Step 2.1: Shared Memory for K,V + Bank Conflict Mitigation (Day 4-5) 🚨 CRITICAL

**Change**: Load K,V tiles into SMEM to reduce GMEM traffic

**⚠️ L4-SPECIFIC TRAP**: HEAD_DIM=64 × 2 bytes/half = 128 bytes = **exactly 32 banks**  
Without swizzling/padding: **32-way bank conflicts** → 32× slower SMEM → **NEGATIVE speedup**

**Implementation** (swizzling and SMEM together):
```cpp
// Block size: 32×32 tile (BLOCK_M=32, BLOCK_N=32)
#define BLOCK_M 32
#define BLOCK_N 32
#define HEAD_DIM 64

// BAD (will cause 32-way conflicts):
// __shared__ half K_smem[BLOCK_N][HEAD_DIM];

// GOOD Option 1: Padding (simple, wastes 12.5% SMEM)
__shared__ half K_smem[BLOCK_N][HEAD_DIM + 8];  // +8 padding breaks conflicts
__shared__ half V_smem[BLOCK_N][HEAD_DIM + 8];

// BETTER Option 2: XOR swizzling (optimal, no waste)
#define SWIZZLE_BITS 3
__device__ __forceinline__ int swizzle_offset(int row, int col) {
    // XOR bits [6:4] of row with bits [6:4] of column offset
    return ((row >> 2) ^ (col >> 4)) & ((1 << SWIZZLE_BITS) - 1);
}

__shared__ half K_smem_raw[BLOCK_N * (HEAD_DIM + 8)];
__shared__ half V_smem_raw[BLOCK_N * (HEAD_DIM + 8)];

__device__ __forceinline__ half* K_swizzled(int row, int col) {
    int swizzle = swizzle_offset(row, col);
    int offset = row * (HEAD_DIM + 8) + col + (swizzle << 2);
    return &K_smem_raw[offset];
}

// Usage:
*K_swizzled(row, col) = K_gmem[...];  // Store
half k_val = *K_swizzled(row, col);   // Load
```

**Expected Speedup**: 1.5-2× (from 150μs → 75-100μs)

**Gate**: 
- ✅ Correctness tests still pass
- ✅ Speedup ≥ 1.3× (NOT degradation!)
- ✅ **NEW (L4)**: `shared_load_transactions_per_request < 2.0` in Nsight Compute
- ⚠️ **If you see 32.0**: You have 32-way conflicts, fix immediately

---

### Step 2.2: Two-Stage Pipelining (Day 6)
**Change**: Use `cp.async` to prefetch next K,V tile while computing current

**Implementation**:
- STAGES=2 (double buffer K,V in SMEM)
- SMEM: 2×8KB = 16KB for K, 2×8KB = 16KB for V (32KB total)
- Async copy Group 0 and 1

**Expected Speedup**: 1.2-1.5× (from 85μs → 60-70μs)

**Gate**: Correctness + speedup ≥ 1.1×

---

### Step 2.3: Vectorized Loads (Day 7)
**Change**: Use `uint4` (128-bit / 16-byte) loads for Q,K,V

**L4 Requirement**: 16-byte alignment mandatory for optimal coalescing

**Implementation**:
```cpp
// Ensure alignment at kernel launch
assert((uintptr_t)Q % 16 == 0);
assert((uintptr_t)K % 16 == 0);
assert((uintptr_t)V % 16 == 0);

// Use uint4 for 128-bit (16-byte) loads of 8×fp16
const uint4* Q_vec = reinterpret_cast<const uint4*>(Q_gmem);
uint4 q_data = Q_vec[tid];  // Loads 8 halfs at once

// For HEAD_DIM=64: 64 elements ÷ 8 per load = 8 loads per row
#pragma unroll
for (int i = 0; i < 8; i++) {
    uint4 data = Q_vec[row * 8 + i];
    // Unpack uint4 → 8 halfs and use...
    half* halfs = reinterpret_cast<half*>(&data);
    for (int j = 0; j < 8; j++) {
        q_row[i * 8 + j] = __half2float(halfs[j]);
    }
}
```

**Expected Speedup**: 1.1-1.3× (from 65μs → 50-60μs)

**Gate**: 
- ✅ Correctness + speedup ≥ 1.05×
- ✅ **NEW (L4)**: Nsight `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` shows ~128-bit transactions
- ✅ **NEW (L4)**: DRAM bandwidth > 200 GB/s (67% of L4's 300 GB/s peak)

---

### Phase 2 Summary
**Duration**: 3-4 days  
**Target**: 50-80μs (competitive with PyTorch SDPA @ 48μs)  
**Gate**: All tests pass + performance meets target → Proceed to Phase 3

---

## Phase 3: Tensor Core Integration (Week 3)

### Goal
Achieve 15-25μs using WMMA (Tensor Cores) for Q·K^T and P@V.

### Step 3.1: WMMA for Q·K^T (Day 8-10) 🚨 ADA-SPECIFIC

**Change**: Replace scalar dot product with `wmma::mma_sync`

**⚠️ CRITICAL FOR L4 (Ada sm_89)**: Use **FP16 accumulation** (not FP32) for 2× throughput  
**Why**: Ada Tensor Cores: 242 TFLOPS @ FP16 accumulation vs 121 TFLOPS @ FP32  
**Safe for attention**: Post-softmax values bounded [0,1] → FP16 precision sufficient

**Requirements (Ada sm_89 specific)**:
```cpp
#include <mma.h>
using namespace nvcuda;

// Fragment types for Ada (Fourth-generation Tensor Cores)
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

// KEY FOR L4: Use FP16 accumulation (not FP32) for 2× throughput
wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;  // half, not float!

// Load from SMEM (NOT registers - this caused previous failures)
wmma::load_matrix_sync(a_frag, &Q_smem[warp_m * 16][0], HEAD_DIM);
wmma::load_matrix_sync(b_frag, &K_smem[warp_n * 16][0], HEAD_DIM);

// Initialize accumulator
wmma::fill_fragment(c_frag, __float2half(0.0f));

// Compute (this uses Tensor Cores)
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result
wmma::store_matrix_sync(&QK_smem[warp_m * 16][warp_n * 16], 
                        c_frag, BLOCK_N, wmma::mem_row_major);
```

**Expected Speedup**: 2-3× (from 60μs → 20-30μs, possibly better due to FP16)

**Gate**: 
- ✅ Compiles **without warnings** (local memory = instant rejection)
- ✅ Correctness tests pass with **relaxed tolerance**: `atol=2e-2, rtol=2e-2` (FP16 accumulation)
- ✅ Speedup ≥ 2× (not just 1.5×, due to 2× TC throughput)
- ✅ **NEW (L4)**: Nsight shows `sm__inst_executed_pipe_tensor > 0` (Tensor Cores active)
- ✅ **NEW (L4)**: Nsight shows `sm__sass_thread_inst_executed_op_hadd_pred_on` (FP16 ops, not FP32)

---

### Step 3.1.5: Warp Tiling for WMMA (Day 11) 🚨 MANDATORY FOR L4

**Why**: Single-warp WMMA (one 16×16 tile per warp) leaves 75% of Tensor Cores idle on L4

**Change**: Each warp computes multiple 16×16 tiles

**Implementation**:
```cpp
// Each warp computes 2×2 tiles = 32×32 output
#define WARP_M 2  
#define WARP_N 2

// Per-warp accumulator array
wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag[WARP_M][WARP_N];

// Initialize all accumulators
for (int wm = 0; wm < WARP_M; wm++) {
    for (int wn = 0; wn < WARP_N; wn++) {
        wmma::fill_fragment(c_frag[wm][wn], __float2half(0.0f));
    }
}

// Warp-level loop over tiles
for (int wm = 0; wm < WARP_M; wm++) {
    for (int wn = 0; wn < WARP_N; wn++) {
        // Load 16×16 Q tile
        wmma::load_matrix_sync(a_frag, 
            &Q_smem[warp_id * 32 + wm * 16][0], HEAD_DIM);
        
        // Load 16×16 K tile
        wmma::load_matrix_sync(b_frag,
            &K_smem[wn * 16][0], HEAD_DIM);
        
        // Compute
        wmma::mma_sync(c_frag[wm][wn], a_frag, b_frag, c_frag[wm][wn]);
    }
}
```

**Expected Speedup**: 1.5× (from 25μs → 17μs) from better Tensor Core utilization

**Gate**: 
- ✅ Correctness + speedup ≥ 1.3×
- ✅ **NEW (L4)**: Nsight shows effective TFLOPS > 150 (up from ~80 in Step 3.1)

---

### Step 3.2: WMMA for P@V (Day 12)
**Change**: Use WMMA for attention_weights @ V

**Expected Speedup**: 1.2-1.5× (from 17μs → 12-15μs)

**Gate**: Correctness + speedup ≥ 1.1×

---

### Phase 3 Summary
**Duration**: 6-7 days (revised with warp tiling)  
**Target**: 12-20μs (with FP16 accumulation + warp tiling)  
**Gate**: All tests pass + performance meets target → Proceed to Phase 3.5

---

## Phase 3.5: L2 Cache Persistence (Day 13, 2-4 hours) 🚨 HIGH ROI FOR L4

### Why L4 Is Special
- **48MB L2 cache** (4× larger than A100's 12MB)
- For B=2, H=8, S=512, D=64: K,V = 2×8×512×64×2 bytes = **1MB per batch**
- Entire K,V fits in L2 → massive speedup if pinned

### Step 3.5.1: Enable L2 Persistence for K,V
**Change**: Pin K,V in L2 cache across multiple kernel launches

**Implementation**:
```cpp
// Before kernel launch (in host code)
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = K_gmem;
attr.accessPolicyWindow.num_bytes = B * H * S * D * sizeof(half);
attr.accessPolicyWindow.hitRatio = 1.0f;  // Keep in L2
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

cudaStreamSetAttribute(stream, 
    cudaStreamAttributeAccessPolicyWindow, &attr);

// Repeat for V_gmem (K and V are reused across Q tiles)
attr.accessPolicyWindow.base_ptr = V_gmem;
cudaStreamSetAttribute(stream, 
    cudaStreamAttributeAccessPolicyWindow, &attr);

// Launch kernel
kernel<<<grid, block, 0, stream>>>(...);
```

**Expected Speedup**: 1.15-1.25× (from 15μs → 12-13μs) for multi-head/batch scenarios

**Gate**: 
- ✅ Correctness (no functional change)
- ✅ Speedup ≥ 1.1× for B≥2
- ✅ **NEW (L4)**: Nsight shows `lts__t_sectors_op_read_hit_rate` > 85%

### Phase 3.5 Summary
**Duration**: 2-4 hours  
**Target**: 12-13μs (competitive with optimized PyTorch)  
**Gate**: L2 hit rate > 85% → Proceed to Phase 4

---

## Phase 4: Advanced Optimizations (Week 4+)

### Potential Optimizations (Priority Order)
1. **Increase tile size** (32×64, 48×64, 64×64): Higher arithmetic intensity
2. **Register tiling** for P@V: Reduce SMEM pressure
3. **Warp-level softmax**: Reduce synchronization overhead
4. **Kernel fusion** (RoPE, LayerNorm): Reduce kernel launches
5. **Multi-query optimization** (GQA): If H_kv < H_q
6. **Split-K optimization** (disabled by default on L4): Test if beneficial for S>2048
7. **FP8 Tensor Cores** (Ada supports FP8): Experimental, 2× throughput vs FP16

### Optimization Loop
For each optimization:
1. Implement on separate branch
2. Run correctness tests (must pass)
3. Run benchmarks (must improve ≥ 3%)
4. Run Nsight Compute (validate metrics)
5. If all gates pass: merge to main
6. If any gate fails: revert and try different approach

---

## Success Criteria (Final)

### Correctness (Non-Negotiable)
- ✅ All parity tests pass (atol=1e-2, rtol=1e-2)
- ✅ Oracle test passes
- ✅ No CUDA errors or warnings
- ✅ No NaN outputs for any input shape

### Performance (Target)
- ✅ B=2, H=8, S=512, D=64: **< 25μs** (2× faster than PyTorch SDPA @ 48μs)
- ✅ B=8: < 80μs
- ✅ p90 not worse than p50 + 20%
- ✅ Nsight Compute: SM busy ≥ 60%, DRAM throughput ≥ 70%

### Code Quality
- ✅ No compiler warnings
- ✅ Clean build system
- ✅ Comprehensive tests (unit, parity, oracle)
- ✅ Documentation for each optimization

---

## Testing Strategy

### Correctness Gates (Run After Every Change)
```bash
pytest tests/test_v3_scalar_correctness.py -v
```
**Pass criteria**: All tests pass, max_diff < 0.01

### Performance Gates (Run After Optimization)
```bash
python scripts/bench_v3_scalar_baseline.py --shapes canonical
```
**Pass criteria**: p50 improves ≥ 3% vs previous best

### Regression Detection
Keep `leaderboard.json` with best p50 for each config:
```json
{
  "scalar_baseline": {"p50_us": 150.2, "commit": "abc123"},
  "smem_32x32": {"p50_us": 95.1, "commit": "def456"},
  "wmma_qk": {"p50_us": 28.3, "commit": "ghi789"}
}
```

---

## Risk Mitigation

### Risk 1: WMMA Still Broken
**Mitigation**: Establish scalar baseline **first** (Phase 1-2). If WMMA fails again, scalar performance (50-80μs) is still production-ready.

### Risk 2: Performance Target Missed
**Mitigation**: Incremental optimization with clear gates. If optimization X doesn't work, revert and try Y. Always have working baseline.

### Risk 3: Correctness Regression
**Mitigation**: Run tests after **every single change**. Git bisect if regression detected. Never merge without green tests.

---

## Timeline Summary (Revised with L4 Priorities)

| Phase | Duration | Goal | Target Latency | L4-Specific Changes |
|-------|----------|------|----------------|---------------------|
| **Phase 0** | 0.5 days | Verify PyTorch baseline | N/A | **NEW**: Baseline could be 5-10μs! |
| **Phase 1** | 3-4 days | Scalar baseline | 2-4× slower than PyTorch | No change |
| **Phase 2** | 4-5 days | Memory + bank conflicts | 50-80μs | **+1 day**: Swizzling mandatory |
| **Phase 3** | 6-7 days | TC + warp tiling | 12-20μs | **+1 day**: FP16 accum + warp tiling |
| **Phase 3.5** | 0.25 days | L2 persistence | 12-13μs | **NEW**: Moved up from Phase 4 |
| **Phase 4** | Ongoing | Polish | < 12μs | Removed L2 tuning |

**Total to production**: 14-17 days (2-2.5 weeks) for Tensor Core + L2 tuning

---

## Next Immediate Steps

1. ✅ Clean slate branch created: `feature/v3_clean_slate`
2. ⏳ **Phase 0**: Verify PyTorch SDPA baseline (30 minutes) - **DO THIS FIRST!**
3. ⏳ **Implement Step 1.1**: Minimal scalar kernel (2-3 hours)
4. ⏳ **Implement Step 1.2**: Correctness tests (3-4 hours)
5. ⏳ **Gate check**: All tests pass → Continue
6. ⏳ **Implement Step 1.3**: Performance baseline (2 hours)
7. ⏳ **Gate check**: Latency meets target → Phase 1 complete

**CRITICAL**: Phase 0 (baseline verification) determines all subsequent targets.  
**First milestone**: Working scalar kernel with correctness validation (Day 1-2)

---

## Philosophy Reminder

> **"Perfect is the enemy of good, but correct is the enemy of nothing."**
>
> We're building a production kernel, not a research prototype.  
> Correctness is non-negotiable. Performance is incremental.  
> Test after every change. Revert if tests fail.  
> No exceptions.

---

## L4-Specific Testing Checklist 🚨 RUN AFTER EACH OPTIMIZATION

Use Nsight Compute to validate L4-specific metrics at each phase:

### Phase 2+ (Memory): Bank Conflict Detection
```bash
ncu --metrics shared_load_transactions_per_request,\
              shared_store_transactions_per_request \
    ./your_kernel_phase2

# Target: < 2.0 (anything > 10.0 means serious conflicts)
# If you see 32.0: Catastrophic! Implement swizzling immediately.
```

### Phase 3+ (Tensor Cores): Utilization Check
```bash
ncu --metrics sm__inst_executed_pipe_tensor.avg \
    ./your_kernel_phase3

# Target: > 0 (Phase 3), ideally > 1000 per SM
# Also check FP16 ops:
ncu --metrics sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
              sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
    ./your_kernel_phase3

# FP16 ops should be >> FP32 ops (hadd >> fadd)
```

### Phase 2+ (Memory): Bandwidth Check
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./your_kernel_phase2

# Target: > 70% (Phase 2), > 80% (Phase 3)
# L4 peak: 300 GB/s, so 70% = 210 GB/s
```

### Phase 3.5+ (L2 Cache): Hit Rate Check
```bash
ncu --metrics lts__t_sectors_op_read_hit_rate.pct \
    ./your_kernel_phase3_5

# Target: > 85% after L2 persistence tuning
# L4's 48MB L2 should easily hold K,V for B=2-8
```

### All Phases: Performance Summary
```bash
# Get overall performance metrics
ncu --metrics sm__cycles_elapsed.avg,\
              sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              dram__bytes.sum.per_second \
    ./your_kernel

# Compare to theoretical peak:
# L4: 242 TFLOPS @ FP16 TC, 300 GB/s DRAM bandwidth
```

---

## Critical L4 Warnings ⚠️ DON'T SKIP!

### Warning 1: Bank Conflicts Will Kill Phase 2 Performance
If you implement Step 2.1 without swizzling/padding, you'll see **negative speedup**.  
**Validation**: After Step 2.1, `shared_load_transactions_per_request` MUST be < 2.0.  
**If it's 32.0**: You've serialized everything. Go back and add swizzling.

### Warning 2: FP32 Accumulation Wastes 50% Throughput
Your Step 3.1 mentions "FP32 for numerical stability" - **incorrect for L4**.  
Ada gets 2× throughput with FP16 accumulation, and attention's bounded range makes it safe.  
**Validation**: Nsight should show `hadd` (FP16 ops), not `fadd` (FP32 ops).

### Warning 3: Single-Warp WMMA Underutilizes L4
If each warp computes only one 16×16 tile, you're using 25% of Tensor Core capacity.  
Step 3.1.5 (warp tiling) is **mandatory**, not optional.  
**Validation**: Effective TFLOPS should be > 80 (Phase 3) → 150+ (Phase 3.1.5).

---

## Final Success Criteria (Reconciled with L4)

### Phase 1: Scalar Baseline
- ✅ 2-4× slower than verified PyTorch baseline
- ✅ All correctness tests pass (atol=1e-2, rtol=1e-2)

### Phase 2: Memory Optimizations  
- ✅ 50-80μs (or appropriate target based on Phase 0 baseline)
- ✅ **Nsight**: `shared_load_transactions_per_request < 2.0`
- ✅ **Nsight**: `dram__throughput > 200 GB/s`

### Phase 3: Tensor Cores + Warp Tiling
- ✅ 12-20μs (faster than PyTorch SDPA's likely 5-10μs when properly verified)
- ✅ **Nsight**: Tensor Core instructions > 0
- ✅ **Nsight**: FP16 ops (hadd) >> FP32 ops (fadd)
- ✅ **Nsight**: Effective TFLOPS > 150

### Phase 3.5: L2 Cache Persistence
- ✅ 12-13μs on canonical shapes
- ✅ **Nsight**: L2 hit rate > 85%

### Phase 4: Polish
- ✅ < 12μs on canon_3 (B=2, H=8, S=512, D=64)
- ✅ < 30μs on canon_1 (B=4, H=16, S=2048, D=128)
- ✅ Beat PyTorch SDPA by 20-40% on all canonical shapes

---

**Status**: ✅ L4-optimized roadmap complete. Ready to begin Phase 0 (baseline verification).  
**Branch**: `feature/v3_clean_slate`  
**Next**: Run `scripts/verify_sdpa_baseline_l4.py` to determine true baseline

