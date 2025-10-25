# FlashCore Fused Attention Kernel - Design Document

**Target**: <40 μs latency for B=1, H=8, S=512, D=64 on NVIDIA L4 (sm_89)  
**Baseline**: 634 μs (multi-query scalar kernel)  
**Required Speedup**: ~16× (aggressive but achievable)

---

## Architecture Overview

This kernel implements **fully fused attention** using the FlashAttention-2 online softmax algorithm combined with WMMA (Tensor Cores) for both Q@K^T and P@V operations. The key innovation is computing attention in a single kernel pass without materializing the intermediate score matrix S or probability matrix P in global memory.

### Key Features

1. **Fused Online Softmax**: Running softmax statistics (m, l) updated per tile
2. **WMMA Q@K^T**: 16×16×16 Tensor Core operations for score computation
3. **WMMA P@V**: 16×16×16 Tensor Core operations for output accumulation
4. **2-Stage cp.async**: Overlap K/V loads with computation
5. **Warp-Level Reductions**: Parallel max/sum for softmax normalization

---

## Tiling Strategy

### CTA (Thread Block) Configuration

```
BLOCK_M = 64        // Query rows per CTA
BLOCK_N = 64        // Key/Value rows per CTA  
BLOCK_K = 64        // Head dimension (D)
HEAD_DIM_SMEM = 72  // Padded to 72 for alignment (64 + 8)

NUM_WARPS = 4       // 2×2 warp grid
WARP_M = 2          // Warps along M dimension
WARP_N = 2          // Warps along N dimension

WMMA_M = 16         // WMMA tile height
WMMA_N = 16         // WMMA tile width
WMMA_K = 16         // WMMA tile depth
```

### Warp Layout (2×2 Grid)

```
       N dimension (64 columns)
    ┌─────────┬─────────┐
    │ Warp 0  │ Warp 1  │  Rows 0-15
    │ (0-15,  │ (0-15,  │
M   │  0-15)  │ 16-31)  │
    ├─────────┼─────────┤
dim │ Warp 2  │ Warp 3  │  Rows 16-31
    │ (16-31, │ (16-31, │
    │  0-15)  │ 16-31)  │
    └─────────┴─────────┘
    
Each warp computes 16×16 sub-tile
4 warps cover 32×32 CTA tile (for 32×32 variant)
8 warps cover 64×64 CTA tile (for 64×64 variant)
```

### Work Distribution

- **Per CTA**: Processes BLOCK_M query rows, iterates over all BLOCK_N key/value tiles
- **Per Warp**: Computes WMMA_M × WMMA_N = 16×16 sub-tile
- **Per Thread**: Holds 8 elements of WMMA accumulator fragment

---

## Memory Layout

### Shared Memory Organization

```cuda
// Query tile (staged once per CTA, reused across all K/V tiles)
__shared__ alignas(16) half sQ[BLOCK_M][HEAD_DIM_SMEM];  // 64×72 = 4.5 KB

// Key/Value tiles (double-buffered for cp.async)
__shared__ alignas(16) uint8_t sK_u8[NUM_STAGES][BLOCK_N][HEAD_DIM_SMEM];  // 2×64×72 = 9 KB
__shared__ alignas(16) uint8_t sV_u8[NUM_STAGES][BLOCK_N][HEAD_DIM_SMEM];  // 2×64×72 = 9 KB

// Transposed K for WMMA (col-major)
__shared__ alignas(16) half sKT[BLOCK_N][HEAD_DIM_SMEM];  // 64×72 = 4.5 KB

// Probability matrix (fused softmax result, per tile)
__shared__ alignas(16) half sP[BLOCK_M][BLOCK_N];  // 64×64 = 8 KB

// Softmax statistics (per query row)
__shared__ alignas(16) float m_smem[BLOCK_M];  // Max per row: 64×4 = 256 B
__shared__ alignas(16) float l_smem[BLOCK_M];  // Sum per row: 64×4 = 256 B

// Output accumulator (unnormalized)
__shared__ alignas(16) float U_smem[BLOCK_M][HEAD_DIM_SMEM];  // 64×72×4 = 18 KB

// Total SMEM: ~54 KB (fits in 48 KB with 32×32 tiles, or request 100 KB for 64×64)
```

### Layout Conventions

- **Q**: Row-major `[BLOCK_M][HEAD_DIM_SMEM]` for WMMA matrix A
- **K^T**: Col-major `[BLOCK_N][HEAD_DIM_SMEM]` for WMMA matrix B (transposed staging)
- **V**: Row-major `[BLOCK_N][HEAD_DIM_SMEM]`
- **P**: Row-major `[BLOCK_M][BLOCK_N]` for WMMA matrix A (softmax output)
- **Padding**: HEAD_DIM_SMEM = 72 (instead of 64) to avoid bank conflicts

---

## Algorithm: Fused Online Softmax

### High-Level Flow

```
For each CTA (processing BLOCK_M query rows):
    1. Load Q tile into sQ (staged once, reused)
    2. Initialize: m_smem[r] = -inf, l_smem[r] = 0, U_smem[r] = 0
    
    For each K/V tile t (BLOCK_N keys/values):
        3. Async load K, V into sK_u8[stage], sV_u8[stage]
        4. Wait for cp.async completion, convert to half, transpose K -> sKT
        
        5. WMMA Q @ K^T -> c_frag (FP32, 16×16 per warp)
        
        6. Fused softmax (per warp, per row):
            a. Extract row maxima from c_frag using warp reduction
            b. Update global max: m_new = max(m_smem[r], m_tile[r])
            c. Rescale previous output: U *= exp(m_old - m_new)
            d. Compute P_tile = exp(c_frag - m_new) and sum l_add
            e. Update global sum: l_new = l_old * exp(m_old - m_new) + l_add
            f. Write P_tile to sP (as half)
        
        7. __syncthreads() to ensure sP is visible
        
        8. WMMA P @ V -> c_frag_pv (FP32, 16×16 per warp)
        9. Accumulate into U_smem += c_frag_pv
    
    10. Final normalization: O = U / l_smem
    11. Write O to global memory
```

### Detailed Softmax Algorithm (Per Warp, Per Row)

Based on FlashAttention-2 online softmax (see `research_fused_flashcore.md`):

```cuda
// Per warp: c_frag holds 16×16 Q@K^T scores in FP32
for each row r in warp's 16 rows:
    // Step 1: Find max in current tile
    float m_tile = -INFINITY;
    for each elem i in c_frag belonging to row r:
        if c_frag.x[i] is valid (col < S):
            m_tile = max(m_tile, c_frag.x[i]);
    m_tile = warp_reduce_max(m_tile);  // All lanes get same m_tile
    
    // Step 2: Update global max
    float m_old = m_smem[r];
    float m_new = max(m_old, m_tile);
    if (lane_id == 0) m_smem[r] = m_new;
    
    // Step 3: Rescale previous output U (if not first tile)
    float scale_old = exp(m_old - m_new);
    for each elem in U_smem[r]:
        U_smem[r][d] *= scale_old;
    
    // Step 4: Compute exp(score - m_new) and sum
    float l_add = 0.0f;
    for each elem i in c_frag belonging to row r:
        if valid:
            float prob = exp(c_frag.x[i] - m_new);
            c_frag.x[i] = prob;  // Overwrite with exp-normalized value
            l_add += prob;
    l_add = warp_reduce_sum(l_add);
    
    // Step 5: Update global sum
    float l_old = l_smem[r];
    float l_new = l_old * scale_old + l_add;
    if (lane_id == 0) l_smem[r] = l_new;
    
    // Step 6: Convert c_frag to half and store in sP
    for each elem i in c_frag:
        sP[global_r][global_c] = __float2half(c_frag.x[i]);

__syncthreads();  // Ensure all warps finish writing sP

// Now proceed to WMMA P@V using sP
```

---

## Pipeline: 2-Stage cp.async

### Overlapping Memory and Compute

```cuda
const int NUM_STAGES = 2;
int stage = 0;

// Prefetch first tile
if (num_tiles > 0) {
    async_copy_tile(sK_u8[0], K_ptr, tile_idx=0);
    async_copy_tile(sV_u8[0], V_ptr, tile_idx=0);
    __pipeline_commit();
}

for (int tile = 0; tile < num_tiles; tile++) {
    // Prefetch next tile (if exists)
    if (tile + 1 < num_tiles) {
        int next_stage = (tile + 1) % NUM_STAGES;
        async_copy_tile(sK_u8[next_stage], K_ptr, tile_idx=tile+1);
        async_copy_tile(sV_u8[next_stage], V_ptr, tile_idx=tile+1);
        __pipeline_commit();
    }
    
    // Wait for current tile to arrive
    __pipeline_wait_prior(NUM_STAGES - 1);  // For 2-stage: wait_prior(1)
    __syncthreads();  // Ensure all threads see the data
    
    // Convert uint8 -> half, transpose K -> sKT
    dequantize_and_transpose(sK_u8[stage], sKT);
    dequantize_and_transpose(sV_u8[stage], sV);
    __syncthreads();
    
    // Compute: WMMA Q@K^T, fused softmax, WMMA P@V
    compute_tile_fused(sQ, sKT, sV, sP, m_smem, l_smem, U_smem);
    
    stage = (stage + 1) % NUM_STAGES;
}
```

### cp.async Details

- Uses `__pipeline_memcpy_async` for 128B aligned copies
- Each warp copies 16 bytes (uint4) per instruction
- Coalesced across 4 lanes to form 128B cache line accesses
- `__pipeline_commit()` commits copy group
- `__pipeline_wait_prior(1)` waits for all but 1 outstanding group (2-stage overlap)

---

## WMMA Implementation Details

### Fragment Types

```cuda
using namespace nvcuda::wmma;

// Q @ K^T fragments
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag_qk;  // Q
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag_qk;  // K^T
fragment<accumulator, 16, 16, 16, float> c_frag_qk;         // Scores (FP32)

// P @ V fragments
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag_pv;  // P
fragment<matrix_b, 16, 16, 16, half, row_major> b_frag_pv;  // V
fragment<accumulator, 16, 16, 16, float> c_frag_pv;         // Output (FP32)
```

### Q @ K^T (Per Warp)

```cuda
const int warp_m = warp_id / WARP_N;  // Row index in 2×2 grid
const int warp_n = warp_id % WARP_N;  // Col index in 2×2 grid
const int tile_m = warp_m * WMMA_M;   // Start row in BLOCK_M
const int tile_n = warp_n * WMMA_N;   // Start col in BLOCK_N

fill_fragment(c_frag_qk, 0.0f);

// Iterate over head dimension D=64 in chunks of WMMA_K=16
for (int k = 0; k < HEAD_DIM; k += WMMA_K) {
    // Load Q[tile_m:tile_m+16, k:k+16] (row-major)
    load_matrix_sync(a_frag_qk, &sQ[tile_m][k], HEAD_DIM_SMEM);
    
    // Load K^T[tile_n:tile_n+16, k:k+16] (col-major, transposed)
    load_matrix_sync(b_frag_qk, &sKT[tile_n][k], HEAD_DIM_SMEM);
    
    // Multiply-accumulate: c_frag_qk += a_frag_qk @ b_frag_qk
    mma_sync(c_frag_qk, a_frag_qk, b_frag_qk, c_frag_qk);
}

// c_frag_qk now holds 16×16 scores in FP32
// Apply softmax scale
for (int i = 0; i < c_frag_qk.num_elements; i++) {
    c_frag_qk.x[i] *= softmax_scale;
}
```

### P @ V (Per Warp)

```cuda
fill_fragment(c_frag_pv, 0.0f);

// Iterate over BLOCK_N in chunks of WMMA_K=16
for (int k = 0; k < BLOCK_N; k += WMMA_K) {
    // Load P[tile_m:tile_m+16, k:k+16] (row-major)
    load_matrix_sync(a_frag_pv, &sP[tile_m][k], BLOCK_N);
    
    // Load V[k:k+16, 0:16] (row-major, treating D chunks)
    load_matrix_sync(b_frag_pv, &sV[k][0], HEAD_DIM_SMEM);
    
    // Multiply-accumulate: c_frag_pv += a_frag_pv @ b_frag_pv
    mma_sync(c_frag_pv, a_frag_pv, b_frag_pv, c_frag_pv);
}

// Store c_frag_pv to U_smem (accumulate into existing values)
// Use WMMA_ACCUM_LUT to map fragment indices to (row, col)
for (int i = 0; i < c_frag_pv.num_elements; i++) {
    auto [r, c] = WMMA_ACCUM_LUT[lane_id][i];
    int global_r = tile_m + r;
    int global_c = c;  // D dimension
    atomicAdd(&U_smem[global_r][global_c], c_frag_pv.x[i]);
}
```

### Fragment Layout LUT

```cuda
// Precalculated lookup table: WMMA_ACCUM_LUT[32 lanes][8 elements] = (row, col)
// Maps each thread's fragment index to matrix position
// Required for row-wise reductions and atomic accumulation
constexpr int WMMA_ACCUM_LUT[32][8] = {
    // Lane 0: {(0,0), (0,1), (1,0), (1,1), (8,0), (8,1), (9,0), (9,1)}
    // ... (full table based on WMMA FP16->FP32 layout on sm_89)
};
```

---

## Resource Constraints

### Launch Configuration

```cuda
__global__ void __launch_bounds__(128, 2)  // 128 threads/block, min 2 blocks/SM
flashcore_fused_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int B, int H, int S, int D
)
```

### Resource Budgets

| Resource | Target | Max | Notes |
|----------|--------|-----|-------|
| **Registers** | ≤96 per thread | 128 | Avoid spills (NCU metric) |
| **SMEM** | ~48 KB | 100 KB | Can request more with `cudaFuncSetAttribute` |
| **Threads** | 128 (4 warps) | 256 | 32 threads/warp × 4 warps |
| **Occupancy** | ≥30% | 100% | Aim for 2-4 blocks/SM (8-16 warps) |

### Memory Alignment

- All shared arrays: `alignas(16)` (128-bit alignment)
- All pointers: `__restrict__` (compiler optimization)
- cp.async: 16-byte chunks, coalesced to 128B cache lines

---

## NCU Profiling Targets

### Success Criteria

| Metric | Target | Stretch | Notes |
|--------|--------|---------|-------|
| **Latency (p50)** | <50 μs | <40 μs | Mission shape B=1,H=8,S=512,D=64 |
| **Speedup vs Baseline** | ≥12× | ≥16× | Baseline: 634 μs |
| **Correctness** | max_err <0.06 | <0.01 | vs PyTorch SDPA |
| **Occupancy** | ≥30% | ≥50% | Theoretical and achieved |
| **TC Utilization** | ≥60% | ≥80% | `sm__pipe_tensor_cycles_active` |
| **DRAM Util** | <10% | <5% | Should be compute-bound |
| **Register Spills** | 0 | 0 | CRITICAL |

### Key Metrics to Track

1. **Occupancy**:
   - `sm__warps_active.avg.pct_of_peak`
   - `launch__registers_per_thread`
   - `launch__shared_mem_per_block_static`

2. **Stalls**:
   - `smsp__warp_issue_stalled_*` (breakdown by reason)
   - Target: <30% Memory Dependency stalls

3. **Memory**:
   - `dram__throughput.avg.pct_of_peak` (expect <10%)
   - `l2_cache_hit_rate` (expect >90%)
   - `shared_efficiency` (expect >95%, check bank conflicts)

4. **Compute**:
   - `sm__inst_executed_pipe_tensor` (HMMA ops count)
   - `sm__throughput.avg.pct_of_peak` (overall SM utilization)
   - Target: >60% TC active

---

## Optimizations Applied

### Memory Optimizations

1. **Shared Memory Padding**: HEAD_DIM_SMEM = 72 (not 64) to avoid bank conflicts
2. **XOR Swizzle**: Optional for K/V staging: `d_swz = d ^ ((n & 1) * 8)`
3. **Vectorized Loads**: Use `uint4` (16 bytes) for cp.async coalescing
4. **Double Buffering**: 2-stage cp.async to overlap memory and compute

### Compute Optimizations

1. **WMMA for Q@K^T**: Leverage Tensor Cores for score computation
2. **WMMA for P@V**: Leverage Tensor Cores for output accumulation
3. **Warp Reductions**: Parallel max/sum using `__shfl_down_sync`
4. **Fused Softmax**: No intermediate S/P matrices in global memory

### Synchronization Minimization

1. **One sync per tile**: `__syncthreads()` after cp.async wait and after writing sP
2. **Warp shuffles**: No sync needed within warp for reductions
3. **Atomic accumulation**: For U_smem, allow parallel warp writes

---

## Variants and Fallbacks

### Tile Size Options

| Variant | BLOCK_M×N | Warps | SMEM | Expected Latency | Notes |
|---------|-----------|-------|------|------------------|-------|
| **Small** | 32×32 | 4 | ~35 KB | 70-100 μs | Safe, high occupancy |
| **Medium** | 48×48 | 8 | ~45 KB | 50-70 μs | Balanced |
| **Large** | 64×64 | 8 | ~54 KB | <50 μs | Requires SMEM request |

### Pipeline Options

| Stages | SMEM | Overlap | Best For |
|--------|------|---------|----------|
| **2-stage** | Base | 1 tile ahead | S ≤ 1024 |
| **3-stage** | +25% | 2 tiles ahead | S > 1024 |

---

## Implementation Checklist

### Phase 2 (Code)

- [ ] Define constants and LUT
- [ ] Implement warp reduction helpers
- [ ] Implement cp.async helpers
- [ ] Implement dequantize + transpose
- [ ] Implement WMMA Q@K^T
- [ ] Implement fused softmax (per warp, per row)
- [ ] Implement WMMA P@V
- [ ] Implement final normalization and writeback
- [ ] Create bindings and build script

### Phase 3 (Test)

- [ ] Correctness test (max_err < 0.06)
- [ ] Performance test (p50, p90)
- [ ] Multi-shape test (256, 512, 1024)
- [ ] NCU profiling (collect metrics)

### Phase 4 (Tune)

- [ ] Autotune BLOCK_M, BLOCK_N
- [ ] Test 2-stage vs 3-stage
- [ ] Verify register usage ≤ target
- [ ] Apply guardrails (spills, occupancy)

---

## References

- `flashcore/notes/research_fused_flashcore.md`: Detailed algorithm explanations
- `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`: Reference implementation
- FlashAttention-2 paper: Online softmax algorithm
- NVIDIA WMMA docs: Fragment layouts and usage

---

**Status**: Design complete, ready for Phase 2 (Implementation)  
**Next Step**: Create `flashcore/kernels/flashcore_fused_wmma.cu`

