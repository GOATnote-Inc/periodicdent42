# Cycle 4: Next Actions - 3 Prioritized Bets

## 🥇 BET 1: Move to Tensor Cores (Stage B) ⭐ RECOMMENDED

**Goal**: Replace scalar dot products with WMMA (Ada Tensor Cores)

**Changes**:
1. Convert FP8→FP16 in SMEM using LUTs (already have kLUT/vLUT!)
2. Implement Q@K^T with `wmma::mma_sync` (16×16×16 tiles)
3. Implement P@V with `wmma::mma_sync` (16×16×16 tiles)
4. Keep online softmax in FP32 (between WMMA passes)

**Expected**: **5-10× speedup** → 200-400 μs
- Tensor Core throughput >> scalar ALU
- Compute will finally be intensive enough to hide memory latency
- SMEM cost justified by compute intensity

**Risk**: Low (well-understood Ada WMMA API)

**Time**: 2-3 hours (implement + test + validate)

**Success Criteria**:
- ≥ 5× faster than Cycle 2a → ≤ 300 μs
- 100% correctness maintained
- NCU shows `sm__pipe_tensor_active > 30%`

---

## 🥈 BET 2: Reduce SMEM + Keep Scalar (Conservative)

**Goal**: Fix Stage A occupancy issue without Tensor Cores

**Changes**:
1. Single-buffer K/V (remove double-buffering)
2. Manual prefetch with explicit index (remove lambda)
3. Reduce D_PAD to D (accept bank conflicts temporarily)

**Expected**: 1.2-1.5× speedup → 1000-1200 μs
- 2 blocks/SM instead of 1 (SMEM: ~25 KB vs 35.5 KB)
- Better occupancy
- Still scalar-bound

**Risk**: Medium (may still be too slow)

**Time**: 1 hour

**Success Criteria**:
- Faster than Cycle 2a (< 1453 μs)
- Occupancy restored (NCU confirms 2 blocks/SM)

---

## 🥉 BET 3: Profile-First (Data-Driven)

**Goal**: Understand exact bottleneck before next move

**Changes**:
1. Run NCU on Stage A vs Cycle 2a
2. Compare:
   - `achieved_occupancy`
   - `smsp__inst_executed_per_warp`
   - `dram__throughput`
   - `sm__pipe_tensor_active`
3. Data-driven decision

**Expected**: Clear diagnosis of regression

**Risk**: Low (just profiling)

**Time**: 30 minutes

**Success Criteria**:
- Definitive answer: occupancy vs compute vs memory
- Informed decision for next cycle

---

## Recommendation: BET 1 (Tensor Cores)

**Why**:
1. **Fundamental limitation**: Scalar compute can't hide latency
2. **Aligns with roadmap**: Stage B is next in user's plan
3. **Biggest potential**: 5-10× speedup vs 1.2-1.5× for BET 2
4. **Production path**: Real kernels use Tensor Cores on Ada/Hopper

**User's guidance**:
> "For QKᵀ and PV, use mma.sync.m16n16k16 per warp"
> "Expected: 3-5× additional speedup → 200-300 μs"

**Skip BET 2** because:
- Even if we fix SMEM, still scalar-bound
- Diminishing returns without changing compute model

**After Stage B**:
- If ≥ 5× faster: Continue to Stage C (persistent CTAs)
- If < 5× faster: Profile and diagnose Tensor Core utilization

---

## Stage B Implementation Plan

### Step 1: FP8→FP16 Conversion (30 min)

```cuda
__shared__ alignas(16) half sQ16[TILE_M][D_PAD];
__shared__ alignas(16) half sK16[TILE_N][D_PAD];
__shared__ alignas(16) half sV16[TILE_N][D_PAD];

// Convert Q once
for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
    for (int d = lane; d < D; d += 32) {
        uint8_t u = sQ[r][d];
        // Use existing Q dequant
        float f = dequant_sim_fp8(u, q_s);
        sQ16[r][d] = __float2half(f);
    }
}

// Convert K/V per tile (use LUT)
for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
    int n = idx / D, d = idx % D;
    sK16[n][d] = __float2half(kLUT[sK[read][n][d]]);
    sV16[n][d] = __float2half(vLUT[sV[read][n][d]]);
}
```

### Step 2: WMMA Q@K^T (1 hour)

```cuda
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major>  A;
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major>  B;
wmma::fragment<wmma::accumulator, 16,16,16, float>               C;

// Each warp computes 16×16 output tile
const int warp_m = (warp_id / 4) * 16;  // 0 or 16
const int warp_n = (warp_id % 4) * 16;  // 0, 16, 32, or 48

wmma::fill_fragment(C, 0.0f);
for (int k = 0; k < D; k += 16) {
    wmma::load_matrix_sync(A, &sQ16[warp_m][k], D_PAD);
    wmma::load_matrix_sync(B, &sK16[warp_n][k], D_PAD);
    wmma::mma_sync(C, A, B, C);
}
// C now contains 16×16 scores in FP32
```

### Step 3: Online Softmax (30 min)

Keep existing algorithm, operate on WMMA output tiles

### Step 4: WMMA P@V (30 min)

Similar to Q@K^T but multiply P (probabilities) × V

---

## Success Metrics (Stage B)

| Metric | Target | Stretch |
|--------|--------|---------|
| Latency | ≤ 300 μs | ≤ 200 μs |
| vs Cycle 2a | ≥ 5× | ≥ 7× |
| vs xFormers | ≤ 15× gap | ≤ 10× gap |
| Correctness | 100% | 100% |
| TC Active | > 30% | > 50% |

---

## Ready to Start?

**Next Command**: Implement Stage B (WMMA Tensor Cores)

Type **GO** to begin Cycle 4, or **STOP** to pause here.

