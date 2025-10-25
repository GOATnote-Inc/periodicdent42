// flashcore_fa3_simple.cu
// Simplified FlashAttention-3 kernel (no double-buffering)
// Goal: Get correctness first, then optimize

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Tunables
#define WARPS_PER_BLOCK 4
#define M_TILE 64
#define N_TILE 64   // Smaller tile for debugging

// Warp reduction
__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask);
  }
  return v;
}

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
flash3_simple_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    float scale)
{
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = WARPS_PER_BLOCK;

  const int tile_q_start = blockIdx.x * M_TILE;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  if (h >= H || b >= B) return;

  const int q_tile_len = min(M_TILE, S - tile_q_start);
  if (q_tile_len <= 0) return;

  // Base offset
  const size_t base = ((size_t)b * H + h) * (size_t)S * D;
  const half* Q_bh = Q + base;
  const half* K_bh = K + base;
  const half* V_bh = V + base;
  half*       O_bh = O + base;

  // Shared memory: Q tile + K tile + V tile
  extern __shared__ half smem[];
  half* smem_Q = smem;
  half* smem_K = smem_Q + M_TILE * D;
  half* smem_V = smem_K + N_TILE * D;

  // Load Q tile cooperatively
  for (int idx = threadIdx.x; idx < q_tile_len * D; idx += blockDim.x) {
    int q_local = idx / D;
    int d = idx % D;
    smem_Q[q_local * D + d] = Q_bh[(tile_q_start + q_local) * D + d];
  }
  __syncthreads();

  // Each warp processes different query rows
  for (int q_local = warp_id; q_local < q_tile_len; q_local += num_warps) {
    const int q_abs = tile_q_start + q_local;

    // Load Q row into registers (each lane gets D/32 elements)
    const int elems_per_lane = D / 32;
    float q_reg[4];  // Max D=128 → 128/32=4
    
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      int d_idx = c * 32 + lane_id;
      q_reg[c] = __half2float(smem_Q[q_local * D + d_idx]);
    }

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    // Output accumulator (FP32)
    float out_acc[4];
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      out_acc[c] = 0.0f;
    }

    // Stream over K/V tiles (no double-buffering for simplicity)
    for (int k_start = 0; k_start < S; k_start += N_TILE) {
      const int k_len = min(N_TILE, S - k_start);

      // Load K/V tile cooperatively
      for (int idx = threadIdx.x; idx < k_len * D; idx += blockDim.x) {
        int j = idx / D;
        int d = idx % D;
        smem_K[j * D + d] = K_bh[(k_start + j) * D + d];
        smem_V[j * D + d] = V_bh[(k_start + j) * D + d];
      }
      __syncthreads();

      // Process each key in this tile
      for (int j = 0; j < k_len; ++j) {
        // Compute Q · K[j] using warp reduction
        float partial = 0.0f;
        
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          int d_idx = c * 32 + lane_id;
          float k_val = __half2float(smem_K[j * D + d_idx]);
          partial += q_reg[c] * k_val;
        }
        
        // Reduce across warp
        float score = warp_sum(partial) * scale;

        // Online softmax update
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float beta = expf(score - m_new);

        // Update running sum
        float l_new = l_i * alpha + beta;

        // Update output accumulator: O = O * alpha + beta * V[j]
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          int d_idx = c * 32 + lane_id;
          float v_val = __half2float(smem_V[j * D + d_idx]);
          out_acc[c] = out_acc[c] * alpha + beta * v_val;
        }

        // Commit new state
        m_i = m_new;
        l_i = l_new;
      }

      __syncthreads();
    }

    // Normalize and write output
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      int d_idx = c * 32 + lane_id;
      float out_val = (l_i > 0.0f) ? (out_acc[c] / l_i) : 0.0f;
      O_bh[q_abs * D + d_idx] = __float2half(out_val);
    }
  }
}

// Host launcher
void launch_flash3_simple(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    cudaStream_t stream)
{
  float scale = 1.0f / sqrtf((float)D);

  dim3 block(WARPS_PER_BLOCK * 32);
  dim3 grid((S + M_TILE - 1) / M_TILE, H, B);

  // Shared memory: Q + K + V tiles
  size_t smem_bytes = (M_TILE + N_TILE + N_TILE) * D * sizeof(half);

  cudaFuncSetAttribute(
      flash3_simple_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);

  flash3_simple_kernel<<<grid, block, smem_bytes, stream>>>(
      Q, K, V, O, B, H, S, D, scale);
}

