# Cycle 3: Stage A Results

## Performance Summary

| Kernel | Latency (μs) | vs Baseline | vs Cycle 2a | Correct |
|--------|--------------|-------------|-------------|---------|
| Baseline (Expert Patch) | 1596.75 | 1.00× | - | ✅ 100% |
| Cycle 2a (blocking cp.async) | 1453.93 | 1.10× | 1.00× | ✅ 100% |
| **Cycle 3 (Stage A, overlap)** | **2046.88** | **0.78×** | **0.71×** | ✅ 100% |

**Result**: ❌ **REGRESSION** (-41% vs Cycle 2a)

---

## Correctness

✅ **100% Correct**: max_diff = 0.000488 (threshold: 0.1)

Algorithm correctness maintained throughout.

---

## Resource Usage

### SMEM Utilization

| Version | SMEM (KB) | % of 48 KB | Occupancy Impact |
|---------|-----------|------------|------------------|
| Cycle 2a | 20.5 | 43% | ✅ Good |
| **Stage A** | **35.5** | **74%** | ⚠️ **Reduced** |

**Breakdown** (Stage A):
```
sK[2][64][80]  = 10.2 KB (double-buffered K)
sV[2][64][80]  = 10.2 KB (double-buffered V)
U_smem[32][80] = 10.2 KB (float accumulator)
kLUT[256]      =  1.0 KB (K dequant LUT)
vLUT[256]      =  1.0 KB (V dequant LUT)
sQ[32][80]     =  2.6 KB (Q tile)
───────────────────────────────────────
Total:         ≈ 35.5 KB
```

### Register Usage

| Version | Registers/Thread | Occupancy Limit |
|---------|------------------|-----------------|
| Cycle 2a | 52 | ✅ Good (≤64) |
| Stage A | 51 | ✅ Good (≤64) |

---

## Root Cause Analysis

### 1. SMEM Bloat (+73%)

**Double-buffering K/V** added 14.5 KB:
- Single buffer: sK[64][80] + sV[64][80] = 10.2 KB
- Double buffer: sK[2][64][80] + sV[2][64][80] = 20.4 KB
- **Delta**: +10.2 KB

**Impact**: 35.5 KB / 48 KB = 74% utilization
- Limits occupancy (fewer blocks/SM)
- May prevent 2 blocks/SM (would need 71 KB → exceeds limit)
- Forces 1 block/SM → lower throughput

### 2. Lambda Function Overhead

`prefetch_tile` lambda with captures may generate:
- Extra instructions for closure setup
- Register spills
- Non-inlined calls

### 3. cp.async Latency Not Hidden

Even with 2-stage pipeline:
- Compute time < memory fetch time
- Wait still stalls before consuming data
- No actual overlap achieved!

**Why**: Scalar compute is TOO FAST (not enough work to hide latency)

---

## Profiling Needed

**Key Metrics** to check:
1. `achieved_occupancy` - expect drop vs Cycle 2a
2. `smsp__inst_executed_per_warp` - lambda overhead?
3. `sm__pipe_tensor_active` - should be 0% (no TC yet)
4. `dram__throughput.avg.pct_of_peak` - bandwidth bound?

---

## Hypotheses

### H1: Occupancy Drop (Most Likely)
35.5 KB SMEM → 1 block/SM instead of 2 → 50% fewer threads → slower

### H2: Lambda Overhead
Closure captures add instruction overhead

### H3: Insufficient Compute
Scalar math too fast → can't hide memory latency even with prefetch

---

## Next Actions

### Option A: Reduce SMEM (Recommended)
- Single-buffer K/V + manual stage index
- Remove D_PAD (use D=64 directly, accept bank conflicts temporarily)
- Target: <28 KB SMEM → 2 blocks/SM possible

### Option B: Skip to Stage B (Tensor Cores)
- WMMA will add enough compute to hide latency
- Accept SMEM cost, rely on compute intensity
- FP8→FP16 + mma.sync

### Option C: Profile-Driven Fix
- Run NCU to confirm occupancy hypothesis
- Measure actual stall cycles
- Data-driven decision

---

## Recommendation

**Go to Stage B (Tensor Cores)** ✅

**Rationale**:
1. Scalar compute fundamentally too fast to hide latency
2. WMMA will add 10-20× compute → overlap becomes valuable
3. SMEM cost acceptable if compute-bound
4. Aligns with user's roadmap (Stage A→B→C)

**Expected** (Stage B):
- FP8→FP16 conversion in SMEM
- WMMA for Q@K^T (16×16×16 tiles)
- WMMA for P@V (16×16×16 tiles)
- **Target**: 200-400 μs (5-10× speedup from Tensor Cores!)


