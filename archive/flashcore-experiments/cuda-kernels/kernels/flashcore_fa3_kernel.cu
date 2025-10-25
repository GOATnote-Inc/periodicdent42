// flashcore_fa3_kernel.cu
// FlashAttention-3–style fused attention (FP16) optimized for L4
// Based on: Streaming K/V, double-buffered smem, online softmax + fused P·V

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// -------------------- Tunables --------------------
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4         // 4 warps = 128 threads
#endif
#ifndef M_TILE
#define M_TILE 64                 // queries per block
#endif
#ifndef N_TILE
#define N_TILE 128                // keys per streamed tile
#endif

// -------------------- Warp helpers --------------------
__device__ __forceinline__
float warp_sum(float v) {
  // full-warp reduction
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask);
  }
  return v;
}

// -------------------- Kernel --------------------
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
flash3_fused_attention_fp16_kernel(
    const half* __restrict__ Q,  // [B,H,S,D]
    const half* __restrict__ K,  // [B,H,S,D]
    const half* __restrict__ V,  // [B,H,S,D]
    half* __restrict__ O,        // [B,H,S,D]
    int B, int H, int S, int D,
    float scale)
{
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / 32;

  const int tile_q_start = blockIdx.x * M_TILE;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  if (h >= H || b >= B) return;

  const int q_tile_len = min(M_TILE, S - tile_q_start);
  if (q_tile_len <= 0) return;

  // Base offset for this (b,h)
  const size_t base = ((size_t)b * H + h) * (size_t)S * D;

  const half* Q_bh = Q + base;
  const half* K_bh = K + base;
  const half* V_bh = V + base;
  half*       O_bh = O + base;

  // Shared memory layout:
  // [ Q_tile (M_TILE*D) | K0 (N_TILE*D) | V0 (N_TILE*D) | K1 (N_TILE*D) | V1 (N_TILE*D) ]
  extern __shared__ half smem[];
  half* smem_Q  = smem;
  half* smem_K0 = smem_Q  + (size_t)M_TILE * D;
  half* smem_V0 = smem_K0 + (size_t)N_TILE * D;
  half* smem_K1 = smem_V0 + (size_t)N_TILE * D;
  half* smem_V1 = smem_K1 + (size_t)N_TILE * D;

  // Load Q tile into shared once (cooperative)
  {
    const int elems = q_tile_len * D;
    for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
      const int q_local = idx / D;
      const int d = idx % D;
      smem_Q[q_local * D + d] = Q_bh[(tile_q_start + q_local) * D + d];
    }
  }
  __syncthreads();

  int buf = 0;
  int k_start = 0;

  // Preload first K/V tile
  {
    const int k_len0 = min(N_TILE, S - k_start);
    const int elems = k_len0 * D;
    half* dstK = buf ? smem_K1 : smem_K0;
    half* dstV = buf ? smem_V1 : smem_V0;
    for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
      const int j = idx / D;
      const int d = idx % D;
      dstK[j * D + d] = K_bh[(k_start + j) * D + d];
      dstV[j * D + d] = V_bh[(k_start + j) * D + d];
    }
  }
  __syncthreads();

  // Each warp iterates over rows in the Q tile
  for (int q_local = warp_id; q_local < q_tile_len; q_local += num_warps) {
    const int q_abs = tile_q_start + q_local;

    // Register-resident slice of Q for this lane
    const int elems_per_lane = D / 32;
    float q_reg[4];  // up to D=128 supported (128/32=4)
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      const int d_idx = c*32 + lane_id;
      q_reg[c] = __half2float(smem_Q[q_local * D + d_idx]);
    }

    // Online softmax state per row
    float m_i = -INFINITY;
    float l_i = 0.f;
    // Output accumulator (FP32)
    float out_acc[4];  // up to D=128
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) out_acc[c] = 0.f;

    // Stream over K/V tiles
    buf = 0;
    for (k_start = 0; k_start < S; k_start += N_TILE) {
      const int k_len = min(N_TILE, S - k_start);

      // Prefetch next tile (simple synchronous)
      if (k_start + N_TILE < S) {
        const int next_len = min(N_TILE, S - (k_start + N_TILE));
        const int elems = next_len * D;
        half* nxtK = (buf ^ 1) ? smem_K1 : smem_K0;
        half* nxtV = (buf ^ 1) ? smem_V1 : smem_V0;
        for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
          const int j = idx / D;
          const int d = idx % D;
          nxtK[j * D + d] = K_bh[(k_start + N_TILE + j) * D + d];
          nxtV[j * D + d] = V_bh[(k_start + N_TILE + j) * D + d];
        }
      }
      __syncthreads();

      // Active buffers
      half* Kbuf = buf ? smem_K1 : smem_K0;
      half* Vbuf = buf ? smem_V1 : smem_V0;

      // Iterate over keys in this tile
      #pragma unroll 4
      for (int j = 0; j < k_len; ++j) {
        // Dot(Q_row, K_j)
        float partial = 0.f;
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          const int d_idx = c*32 + lane_id;
          const float kv = __half2float(Kbuf[j * D + d_idx]);
          partial += q_reg[c] * kv;
        }
        float score = warp_sum(partial) * scale;

        // Online softmax update
        float m_new = fmaxf(m_i, score);
        float alpha = __expf(m_i - m_new);
        float beta  = __expf(score - m_new);

        float l_new = l_i * alpha + beta;

        // Update output numerator
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          const int d_idx = c*32 + lane_id;
          float vcomp = __half2float(Vbuf[j * D + d_idx]);
          out_acc[c] = out_acc[c] * alpha + beta * vcomp;
        }

        m_i = m_new;
        l_i = l_new;
      }

      buf ^= 1;
      __syncthreads();
    }

    // Normalize and store
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      const int d_idx = c*32 + lane_id;
      float outv = (l_i > 0.f) ? (out_acc[c] / l_i) : 0.f;
      O_bh[q_abs * D + d_idx] = __float2half(outv);
    }
  }
}

// -------------------- Host Launcher --------------------
void launch_flash3_fused_attention_fp16(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    cudaStream_t stream)
{
  // Softmax scale
  float scale = 1.0f / sqrtf((float)D);

  dim3 block(WARPS_PER_BLOCK * 32);
  dim3 grid( (S + M_TILE - 1) / M_TILE, H, B );

  // Dynamic shared memory
  size_t smem_elems =
      (size_t)M_TILE * D       // Q tile
    + (size_t)N_TILE * D       // K0
    + (size_t)N_TILE * D       // V0
    + (size_t)N_TILE * D       // K1
    + (size_t)N_TILE * D;      // V1

  size_t smem_bytes = smem_elems * sizeof(half);

  // Opt-in for >48KB shared memory
  cudaFuncSetAttribute(
      flash3_fused_attention_fp16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);

  flash3_fused_attention_fp16_kernel<<<grid, block, smem_bytes, stream>>>(
      Q, K, V, O, B, H, S, D, scale);
}

