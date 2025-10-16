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

## Phase 1: Scalar FlashAttention Baseline (Week 1)

### Goal
Working scalar-only FlashAttention kernel that:
- ✅ Passes correctness tests (oracle, parity with PyTorch SDPA)
- ✅ Compiles without warnings
- ✅ Achieves ~100-200μs on L4 (B=2, H=8, S=512, D=64)
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

### Step 2.1: Shared Memory for K,V (Day 4-5)
**Change**: Load K,V tiles into SMEM to reduce GMEM traffic

**Implementation**:
- Block size: 32×32 tile (BLOCK_M=32, BLOCK_N=32)
- SMEM: 32×64 half for K, 32×64 half for V (8KB each, 16KB total)
- Each block processes BLOCK_M query rows

**Expected Speedup**: 1.5-2× (from 150μs → 75-100μs)

**Gate**: Correctness tests still pass + speedup ≥ 1.3×

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
**Change**: Use `float4` (128-bit) loads for Q,K,V

**Implementation**:
- Ensure 16-byte alignment
- Load 4 `half` at a time (8 bytes)
- Coalesce global memory accesses

**Expected Speedup**: 1.1-1.3× (from 65μs → 50-60μs)

**Gate**: Correctness + speedup ≥ 1.05×

---

### Phase 2 Summary
**Duration**: 3-4 days  
**Target**: 50-80μs (competitive with PyTorch SDPA @ 48μs)  
**Gate**: All tests pass + performance meets target → Proceed to Phase 3

---

## Phase 3: Tensor Core Integration (Week 3)

### Goal
Achieve 15-25μs using WMMA (Tensor Cores) for Q·K^T and P@V.

### Step 3.1: WMMA for Q·K^T (Day 8-10)
**Change**: Replace scalar dot product with `wmma::mma_sync`

**Requirements (Critical for Ada sm_89)**:
- ✅ Load Q,K from **SHARED MEMORY** (NOT local/registers)
- ✅ Use `wmma::load_matrix_sync(..., mem_row_major)`
- ✅ Fragment: `wmma::fragment<matrix_a, 16, 16, 16, half, row_major>`
- ✅ Accumulate in FP32 for numerical stability
- ⚠️ **Validate on hardware**: No "local memory" warnings

**Expected Speedup**: 2-3× (from 60μs → 20-30μs)

**Gate**: 
- ✅ Compiles **without warnings** (local memory = instant rejection)
- ✅ Correctness tests pass (atol=1e-2, rtol=1e-2)
- ✅ Speedup ≥ 1.5×

---

### Step 3.2: WMMA for P@V (Day 11-12)
**Change**: Use WMMA for attention_weights @ V

**Expected Speedup**: 1.2-1.5× (from 25μs → 15-20μs)

**Gate**: Correctness + speedup ≥ 1.1×

---

### Phase 3 Summary
**Duration**: 5-6 days  
**Target**: 15-25μs (3-5× faster than PyTorch SDPA)  
**Gate**: All tests pass + performance meets target → Proceed to Phase 4

---

## Phase 4: Advanced Optimizations (Week 4+)

### Potential Optimizations (Priority Order)
1. **Increase tile size** (32×64, 48×64, 64×64): Higher arithmetic intensity
2. **Register tiling** for P@V: Reduce SMEM pressure
3. **Warp-level softmax**: Reduce synchronization overhead
4. **FP16 accumulation** for Ada: 2× TC throughput
5. **Kernel fusion** (RoPE, LayerNorm): Reduce kernel launches
6. **Multi-query optimization**: If H_kv < H_q
7. **L2 cache persistence**: `cudaStreamSetAttribute` for hot data

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

## Timeline Summary

| Phase | Duration | Goal | Target Latency |
|-------|----------|------|----------------|
| **Phase 1: Scalar Baseline** | 3-4 days | Correctness + basic perf | 100-200μs |
| **Phase 2: Memory Opts** | 3-4 days | SMEM, pipelining, vectorization | 50-80μs |
| **Phase 3: Tensor Cores** | 5-6 days | WMMA for QK and PV | 15-25μs |
| **Phase 4: Advanced** | Ongoing | Tile size, register tiling, etc. | < 15μs |

**Total to production**: 2-3 weeks for Tensor Core version

---

## Next Immediate Steps

1. ✅ Clean slate branch created: `feature/v3_clean_slate`
2. ⏳ **Implement Step 1.1**: Minimal scalar kernel (2-3 hours)
3. ⏳ **Implement Step 1.2**: Correctness tests (3-4 hours)
4. ⏳ **Gate check**: All tests pass → Continue
5. ⏳ **Implement Step 1.3**: Performance baseline (2 hours)
6. ⏳ **Gate check**: Latency < 500μs → Phase 1 complete

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

**Status**: ✅ Roadmap complete. Ready to begin Phase 1 Step 1.1.  
**Branch**: `feature/v3_clean_slate`  
**Next**: Implement minimal scalar kernel

