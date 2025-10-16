# EvoEngineer-Insight Prompt for WMMA Kernel (Phase 3)

**Framework**: EvoEngineer-Insight (Task Context + Optimization Insights, no Historical Solutions)  
**Reference**: https://arxiv.org/html/2510.03760v1 (Table 3, Section 4.2)  
**Target GPU**: NVIDIA L4 (Ada Lovelace, sm_89)  
**Expected Performance**: 1.47-1.60× speedup, 58-63% validity (EvoEngineer Table 4)

---

## System Prompt

You are an expert CUDA kernel engineer specializing in Ada Lovelace (sm_89) architecture. Generate production-safe code with comprehensive guard macros, numerical stability, and performance optimization. Follow NVIDIA's WMMA best practices and Ada-specific optimizations.

---

## Task Context (I1)

### Problem Statement
Implement a **FlashAttention-style kernel** for `Q·K^T` → softmax → `P·V` using **WMMA (Warp Matrix Multiply-Accumulate)** on Ada L4 (sm_89). The kernel must be numerically stable, compile without warnings, and achieve ≥ 2× speedup over PyTorch SDPA baseline (47.10 μs).

### Fixed Parameters
```
Input shapes:
  - Batch (B): 2
  - Heads (H): 8
  - Sequence (S): 512 (fixed, compile-time constant)
  - Head dimension (D): 64
  - Dtype: torch.float16 (FP16)

CTA tile configuration:
  - TILE_M (query rows): 128
  - TILE_N (key columns): 64
  - TILE_K (head dim step): 32
  - NUM_WARPS: 8 (256 threads/CTA)
  - STAGES: 2 (double buffer for K, V)

WMMA configuration:
  - Fragment size: m16n16k16
  - Matrix A (Q): row_major
  - Matrix B (K^T): col_major
  - Accumulator: float (FP32) for Ada 2× throughput
```

### Hardware Constraints (L4 sm_89)
```
Hard limits (MUST NOT EXCEED):
  - SMEM/CTA: ≤ 48 KB (Ada has 100 KB/SM, 48 KB max per CTA)
  - Registers/thread: ≤ 64 (target for 4+ CTAs/SM occupancy)
  - Occupancy target: ≥ 4 CTAs/SM (measured via Nsight)

Memory budget:
  - Q tile: 128 × 32 × 2B = 8 KB
  - K tile: 32 × 64 × 2B = 4 KB
  - V tile: 32 × 64 × 2B = 4 KB
  - ×2 stages = 24 KB
  - Epilogue scratch: ~4-8 KB
  - Total: ~32 KB (safe margin under 48 KB)
```

### Functional Requirements
1. **Correctness**: `torch.allclose(output, pytorch_sdpa_output, atol=1e-2, rtol=1e-2)`
   - Relaxed tolerance due to FP16 WMMA + FP32 accumulation
2. **Numerical stability**: Safe softmax (subtract max before exp)
3. **Causal masking**: Support `is_causal=True` (optional, default False)
4. **No undefined behavior**: No uninitialized reads, race conditions, or OOB access

### Performance Targets
| Metric | Target | Baseline (PyTorch) |
|--------|--------|--------------------|
| Latency (p50) | < 25 μs | 47.10 μs |
| Speedup | ≥ 2× | 1.0× |
| Tensor Core utilization | ≥ 40% | 0% |

---

## Optimization Insights (I3)

### Prior Failure Modes (Learn from These)
1. **Register blow-up** (observed in earlier V3 attempts):
   - **Symptom**: Compilation shows regs/thread > 80, occupancy drops to 1-2 CTAs/SM
   - **Root cause**: Large WMMA fragment arrays in scope simultaneously, temporary scratch arrays
   - **Fix**: Narrow fragment live ranges, use `__restrict__` pointers, avoid per-thread scratch buffers

2. **Bank conflicts** (HEAD_DIM=64 creates 32-way conflicts):
   - **Symptom**: Nsight shows `shared_load_transactions_per_request > 10.0`
   - **Root cause**: Column-major access to 64-element rows (64×2B = 128B = 32 banks)
   - **Fix**: XOR swizzle or +8 padding on SMEM leading dimension

3. **WMMA local memory spills** (NVIDIA compiler bug):
   - **Symptom**: Warning "function uses too much shared memory" or "local memory" in ptxas
   - **Root cause**: Fragment arrays not aligned, passed by value
   - **Fix**: Load fragments from SMEM directly, use `load_matrix_sync(&smem[offset], stride)`

### Nsight Compute Target Metrics (Gates for Accept/Reject)
```
Must achieve after optimization:
  ✅ sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active ≥ 40%
     (Tensor Core utilization - proves WMMA is active)
  
  ✅ dram__throughput.avg.pct_of_peak_sustained_elapsed ≥ 60%
     (Memory bandwidth - proves we're not memory-starved)
  
  ✅ l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum / shared_loads < 5%
     (Bank conflicts - proves swizzle/padding is effective)
  
  ✅ sm__warps_active.avg.pct_of_peak_sustained_active (derived occupancy)
     Target: ≥ 25% (4 CTAs/SM × 8 warps = 32 active warps / 128 max = 25%)
```

**How to run**:
```bash
ncu --set full -o v3_wmma_profile \
    --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    python3 bench_v3_wmma.py
```

### Ada (sm_89) Architecture Specifics
1. **Tensor Cores (4th generation)**:
   - FP16 accumulation: 242 TFLOPS (2× faster than FP32 accumulation)
   - **Critical**: Use `wmma::fragment<accumulator, 16, 16, 16, float>` (not half!)
   - Rationale: FP32 accum has 2× throughput on Ada compared to Ampere

2. **Memory hierarchy**:
   - L2 cache: 48 MB (huge!) → high hit rate expected for K, V reuse
   - L1/SMEM: 128 KB/SM shared (192 KB on Hopper, 100 KB on Ada)
   - SMEM bank width: 32 banks × 4B = 128B (same as Ampere)

3. **Async copy (cp.async)**:
   - Available on sm_89 (Ada has cp.async.cg)
   - Use for STAGES=2 double buffering (K, V tiles)
   - Syntax: `__pipeline_memcpy_async`, `__pipeline_commit`, `__pipeline_wait_prior(1)`

---

## Implementation Requirements

### File Structure
Generate a **single file** that can be dropped into `cudadent42/bench/kernels/fa_s512_v3_wmma.cu`:

```cpp
// fa_s512_v3_wmma.cu
#pragma once
#include <cuda_runtime.h>
#include <mma.h>  // WMMA

#ifdef USE_WMMA  // Guard entire WMMA implementation

// === Helper functions ===

// 1. SMEM swizzle (bank conflict mitigation)
__device__ __forceinline__ int swizzle_offset(int row, int col) {
    // XOR bits [6:4] of row with bits [6:4] of column offset
    // For HEAD_DIM=64, this spreads accesses across 8 banks
    return ((row >> 2) ^ (col >> 4)) & 0x7;
}

// 2. SMEM tile loader (Q from GMEM → SMEM with swizzle)
__device__ void load_q_tile_smem(...) { ... }

// 3. SMEM tile loader (K, V from GMEM → SMEM with swizzle, optional cp.async)
__device__ void load_kv_tile_smem(...) { ... }

// 4. Warp-cooperative rowwise max (for softmax)
__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// 5. Warp-cooperative rowwise sum
__device__ float warp_reduce_sum(float val) { ... }

// === WMMA microkernel ===

// Q·K^T using WMMA m16n16k16
__device__ void compute_qk_wmma(
    const half* Q_smem,  // [TILE_M][HEAD_DIM], swizzled
    const half* K_smem,  // [TILE_K][TILE_N], swizzled
    float* QK_out,       // [TILE_M][TILE_N], register or SMEM
    int warp_id,
    int lane_id
) {
    using namespace nvcuda;
    
    // Fragment declarations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over HEAD_DIM in steps of 16 (WMMA K dimension)
    #pragma unroll
    for (int k_step = 0; k_step < HEAD_DIM; k_step += 16) {
        // Load 16×16 Q tile (with swizzle offset)
        wmma::load_matrix_sync(a_frag, &Q_smem[...], stride_with_swizzle);
        
        // Load 16×16 K^T tile
        wmma::load_matrix_sync(b_frag, &K_smem[...], stride_with_swizzle);
        
        // Compute C = A×B + C (Tensor Core operation)
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result to SMEM or registers
    wmma::store_matrix_sync(&QK_out[...], c_frag, stride, wmma::mem_row_major);
}

// === Softmax ===

__device__ void compute_softmax_inplace(
    float* scores,  // [TILE_M][TILE_N]
    int tile_m,
    int tile_n,
    int warp_id,
    int lane_id
) {
    // Per-row softmax with warp-cooperative reductions
    // 1. Row max
    // 2. Exp(scores - max)
    // 3. Row sum
    // 4. Normalize: scores[i] /= sum
}

// === Main kernel ===

__global__ void flash_attention_s512_v3_wmma_kernel(
    const half* Q,  // [B, H, S, D]
    const half* K,
    const half* V,
    half* O,
    int B, int H, int S, int D,
    bool is_causal
) {
    // Shared memory declarations with swizzle/padding
    __shared__ half Q_smem[TILE_M][HEAD_DIM + 8];  // +8 padding
    __shared__ half K_smem[STAGES][TILE_K][TILE_N + 8];
    __shared__ half V_smem[STAGES][TILE_K][TILE_N + 8];
    __shared__ float QK_smem[TILE_M][TILE_N];  // Intermediate for softmax
    
    // Thread/warp indices
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // CTA tile indices
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int m_block = blockIdx.x;  // Query tile index
    
    // === Step 1: Load Q tile from GMEM to SMEM (no double buffer) ===
    load_q_tile_smem(Q, Q_smem, batch_idx, head_idx, m_block, ...);
    __syncthreads();
    
    // === Step 2: Loop over K, V tiles (N dimension) ===
    int stage = 0;
    for (int n_block = 0; n_block < (S + TILE_N - 1) / TILE_N; n_block++) {
        // Load K, V tiles (use cp.async if STAGES > 1)
        #if STAGES > 1
            __pipeline_memcpy_async(&K_smem[stage][0][0], &K[...], sizeof(half) * TILE_K * TILE_N);
            __pipeline_memcpy_async(&V_smem[stage][0][0], &V[...], sizeof(half) * TILE_K * TILE_N);
            __pipeline_commit();
            __pipeline_wait_prior(1);  // Wait for stage^1
        #else
            load_kv_tile_smem(K, V, K_smem[0], V_smem[0], ...);
            __syncthreads();
        #endif
        
        // === Step 3: Compute Q·K^T using WMMA ===
        compute_qk_wmma(Q_smem, K_smem[stage], QK_smem, warp_id, lane_id);
        __syncthreads();
        
        // === Step 4: Apply causal mask (if enabled) ===
        if (is_causal) {
            // Zero out upper-triangular part
        }
        
        // === Step 5: Softmax (rowwise, in-place on QK_smem) ===
        compute_softmax_inplace(QK_smem, TILE_M, TILE_N, warp_id, lane_id);
        __syncthreads();
        
        // === Step 6: Compute P·V (FMA epilogue, NOT WMMA for first pass) ===
        // TODO: Optionally convert to WMMA in second iteration
        for (int m_local = threadIdx.x / TILE_N; m_local < TILE_M; m_local += blockDim.x / TILE_N) {
            for (int d = threadIdx.x % TILE_N; d < HEAD_DIM; d += TILE_N) {
                float acc = 0.0f;
                #pragma unroll
                for (int n_local = 0; n_local < TILE_N; n_local++) {
                    acc += QK_smem[m_local][n_local] * __half2float(V_smem[stage][n_local][d]);
                }
                O[...] = __float2half(acc);
            }
        }
        
        stage ^= 1;  // Flip stage for double buffering
    }
}

// === Launch wrapper ===

void launch_flash_attention_s512_v3_wmma(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, bool is_causal, cudaStream_t stream
) {
    assert(S == 512 && "Kernel specialized for S=512");
    assert(D == 64 && "Kernel specialized for D=64");
    
    dim3 grid(
        (S + TILE_M - 1) / TILE_M,  // M blocks (query tiles)
        H,                           // Heads
        B                            // Batch
    );
    dim3 block(NUM_WARPS * 32);  // 256 threads (8 warps)
    
    // SMEM calculation (for documentation)
    size_t smem_bytes = 
        sizeof(half) * TILE_M * (HEAD_DIM + 8) +          // Q_smem
        sizeof(half) * STAGES * TILE_K * (TILE_N + 8) * 2 +  // K_smem + V_smem
        sizeof(float) * TILE_M * TILE_N;                  // QK_smem
    
    assert(smem_bytes <= 49152 && "SMEM exceeds 48 KB limit!");
    
    flash_attention_s512_v3_wmma_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, B, H, S, D, is_causal
    );
}

#endif  // USE_WMMA
```

### Compile-Time Parameters (Read from -D flags)
```cpp
#ifndef TILE_M
#define TILE_M 128
#endif

#ifndef TILE_N
#define TILE_N 64
#endif

#ifndef TILE_K
#define TILE_K 32
#endif

#ifndef STAGES
#define STAGES 2
#endif

#ifndef ACCUM_F32
#define ACCUM_F32 1
#endif

#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif
```

### Critical Implementation Details

#### 1. Bank Conflict Mitigation (MUST IMPLEMENT)
```cpp
// Option A: XOR swizzle (preferred, no SMEM waste)
__device__ __forceinline__ int swizzle_offset(int row, int col) {
    return ((row >> 2) ^ (col >> 4)) & 0x7;
}

// Usage:
int smem_offset = row * (HEAD_DIM + 8) + col + (swizzle_offset(row, col) << 2);
half* ptr = &smem_raw[smem_offset];

// Option B: Simple padding (fallback, wastes 12.5% SMEM)
__shared__ half K_smem[TILE_K][TILE_N + 8];  // +8 padding
```

#### 2. WMMA Fragment Loading (MUST LOAD FROM SMEM)
```cpp
// ❌ WRONG (causes local memory spills)
wmma::fragment<...> frag = load_from_register(...);

// ✅ CORRECT (loads directly from SMEM)
wmma::load_matrix_sync(a_frag, &Q_smem[row][0], HEAD_DIM + 8);
```

#### 3. Numerical Stability (Safe Softmax)
```cpp
// ❌ WRONG (overflow risk)
float sum = 0.0f;
for (int i = 0; i < N; i++) sum += expf(scores[i]);
probs[i] = expf(scores[i]) / sum;

// ✅ CORRECT (subtract max first)
float row_max = warp_reduce_max(scores[0]);  // Warp-cooperative
for (int i = 0; i < N; i++) {
    scores[i] = expf(scores[i] - row_max);
    sum += scores[i];
}
sum = warp_reduce_sum(sum);
for (int i = 0; i < N; i++) scores[i] /= sum;
```

---

## Output Format

Generate **complete, compilable code** in a single file. Include:
1. **Header guards** and includes
2. **Helper functions** (swizzle, warp reductions)
3. **WMMA microkernel** (Q·K^T)
4. **Softmax implementation**
5. **P·V epilogue** (FMA, not WMMA for first pass)
6. **Main kernel** with SMEM declarations
7. **Launch wrapper** with grid/block dimensions
8. **Compile-time parameter defaults**

**Code style**:
- Use `__device__` for all device functions
- Use `__forceinline__` for small helpers
- Add comments explaining WMMA operations
- Include `assert()` for dimension checks
- Use `#pragma unroll` for short loops

**Testing checklist** (agent will validate):
- [ ] Compiles without warnings (`-Xptxas -v` output clean)
- [ ] No local memory usage (check ptxas output)
- [ ] Registers/thread ≤ 64 (check ptxas output)
- [ ] SMEM/CTA ≤ 48 KB (check kernel launch)
- [ ] Produces non-NaN outputs (smoke test)
- [ ] Nsight shows `sm__inst_executed_pipe_tensor > 0`

---

## Example Nsight Command

```bash
# Profile on GPU
ncu --set full --target-processes all \
    --kernel-name "flash_attention_s512_v3_wmma_kernel" \
    -o v3_wmma_profile \
    python3 bench_v3_wmma.py

# View metrics
ncu --import v3_wmma_profile.ncu-rep \
    --page details \
    --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active
```

---

## Success Criteria (Phase 3 Gate)

| Check | Target | Measured |
|-------|--------|----------|
| ✅ Compiles clean | 0 warnings | TBD |
| ✅ Correctness | `torch.allclose(atol=1e-2)` | TBD |
| ✅ Latency | < 25 μs | TBD |
| ✅ Tensor Core % | ≥ 40% | TBD |
| ✅ Bank conflicts | < 5% | TBD |
| ✅ Occupancy | ≥ 4 CTAs/SM | TBD |

**If any gate fails**: Set `USE_WMMA=0` and continue with scalar baseline.

---

**Generated**: October 16, 2025  
**Framework**: EvoEngineer-Insight (no historical solutions, uses insights only)  
**Expected performance**: 1.47-1.60× speedup (EvoEngineer Table 4, Claude-Sonnet-4 results)

