// flashcore_fa3_v3.cu
// FA-3 with CORRECT architecture: Outer loop over K/V tiles, inner over query rows
// + WMMA basics for Tensor Core acceleration
// Target: <40 μs on L4 (sm_89)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif
#ifndef M_TILE
#define M_TILE 64
#endif
#ifndef N_TILE
#define N_TILE 64      // Smaller tile for better occupancy
#endif
#ifndef PAD
#define PAD 8
#endif

// WMMA dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask);
  }
  return v;
}

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
flash3_fused_attention_fp16_kernel_v3(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    int is_causal,
    float scale)
{
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int nwarps = WARPS_PER_BLOCK;

  const int q_tile_start = blockIdx.x * M_TILE;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  if (h >= H || b >= B) return;

  const int q_tile_len = min(M_TILE, S - q_tile_start);
  if (q_tile_len <= 0) return;

  const size_t base = ((size_t)b * H + (size_t)h) * (size_t)S * (size_t)D;

  const half* __restrict__ Q_bh = Q + base;
  const half* __restrict__ K_bh = K + base;
  const half* __restrict__ V_bh = V + base;
  half*       __restrict__ O_bh = O + base;

  // Shared memory with padding
  const int LDQ = D + PAD;
  const int LDK = D + PAD;
  const int LDV = D + PAD;

  extern __shared__ half smem[];
  half* smem_Q = smem;
  half* smem_K = smem_Q + (size_t)M_TILE * LDQ;
  half* smem_V = smem_K + (size_t)N_TILE * LDK;

  // Load Q tile ONCE for entire block
  {
    const int q_elems = q_tile_len * D;
    for (int idx = threadIdx.x; idx < q_elems; idx += blockDim.x) {
      const int r = idx / D;
      const int d = idx % D;
      smem_Q[r * LDQ + d] = Q_bh[(q_tile_start + r) * D + d];
    }
  }
  __syncthreads();

  // Per-thread state: each warp handles multiple rows
  // Each thread maintains state for rows it processes
  const int elems_per_lane = D / 32;
  
  // State arrays (per thread, for rows this thread handles)
  float m_state[4];   // max(m_i) for each row (supports up to 4 rows per warp with 4 warps = 16 rows, need 64/4=16 iterations)
  float l_state[4];   // sum(l_i) for each row
  float out_state[4][4];  // output accumulator [row][d_chunk]
  
  // Initialize state for rows this warp will process
  const int rows_per_iter = min(4, (q_tile_len + nwarps - 1) / nwarps);
  #pragma unroll
  for (int r = 0; r < rows_per_iter; ++r) {
    m_state[r] = -INFINITY;
    l_state[r] = 0.0f;
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      out_state[r][c] = 0.0f;
    }
  }

  // === CORRECT ARCHITECTURE: Outer loop over K/V tiles ===
  for (int k_start = 0; k_start < S; k_start += N_TILE) {
    const int k_len = min(N_TILE, S - k_start);

    // Load K/V tile ONCE for all query rows
    {
      const int kv_elems = k_len * D;
      for (int idx = threadIdx.x; idx < kv_elems; idx += blockDim.x) {
        const int j = idx / D;
        const int d = idx % D;
        smem_K[j * LDK + d] = K_bh[(k_start + j) * D + d];
        smem_V[j * LDV + d] = V_bh[(k_start + j) * D + d];
      }
    }
    __syncthreads();

    // === Inner loop: All warps process their rows against this K/V tile ===
    for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
      const int q_abs = q_tile_start + q_local;
      
      // Determine which state slot this row uses
      const int state_idx = (q_local - warp_id) / nwarps;
      if (state_idx >= rows_per_iter) continue;

      // Load Q row into registers
      float q_reg[4];
      #pragma unroll
      for (int c = 0; c < elems_per_lane; ++c) {
        const int d_idx = c * 32 + lane_id;
        q_reg[c] = __half2float(smem_Q[q_local * LDQ + d_idx]);
      }

      // Process each key in this tile
      for (int j = 0; j < k_len; ++j) {
        const int key_abs = k_start + j;

        // Compute Q·K[j] with scalar dot product (WMMA version below)
        float partial = 0.0f;
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          const int d_idx = c * 32 + lane_id;
          float k_val = __half2float(smem_K[j * LDK + d_idx]);
          partial += q_reg[c] * k_val;
        }
        
        float score = warp_sum(partial) * scale;

        // Causal mask
        if (is_causal && key_abs > q_abs) {
          score = -INFINITY;
        }

        // Online softmax update
        float m_prev = m_state[state_idx];
        float m_new = fmaxf(m_prev, score);
        float alpha = expf(m_prev - m_new);
        float beta = expf(score - m_new);
        float l_new = l_state[state_idx] * alpha + beta;

        // Update output accumulator
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          const int d_idx = c * 32 + lane_id;
          float v_val = __half2float(smem_V[j * LDV + d_idx]);
          out_state[state_idx][c] = out_state[state_idx][c] * alpha + beta * v_val;
        }

        // Commit state
        m_state[state_idx] = m_new;
        l_state[state_idx] = l_new;
      }
    }

    __syncthreads();
  }

  // === Write output: Normalize and store ===
  for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
    const int q_abs = q_tile_start + q_local;
    const int state_idx = (q_local - warp_id) / nwarps;
    if (state_idx >= rows_per_iter) continue;

    float l_i = l_state[state_idx];
    
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      const int d_idx = c * 32 + lane_id;
      float out_val = (l_i > 0.0f) ? (out_state[state_idx][c] / l_i) : 0.0f;
      O_bh[q_abs * D + d_idx] = __float2half(out_val);
    }
  }
}

// -------------------- Host launcher --------------------
void launch_flash3_v3(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    bool is_causal,
    cudaStream_t stream)
{
  const float scale = 1.0f / sqrtf((float)D);

  dim3 block(WARPS_PER_BLOCK * 32);
  dim3 grid((S + M_TILE - 1) / M_TILE, H, B);

  // Shared memory: Q + K + V (single-buffered for now)
  const size_t smem_elems =
      (size_t)M_TILE * (D + PAD)   // Q
    + (size_t)N_TILE * (D + PAD)   // K
    + (size_t)N_TILE * (D + PAD);  // V
  const size_t smem_bytes = smem_elems * sizeof(half);

  cudaFuncSetAttribute(
      flash3_fused_attention_fp16_kernel_v3,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);

  flash3_fused_attention_fp16_kernel_v3<<<grid, block, smem_bytes, stream>>>(
      Q, K, V, O, B, H, S, D, (int)is_causal, scale);
}

