// flashcore_fa3_v5.cu
// FA-3 v5: OPTIMAL architecture
// - K/V outer loop (load once!)
// - One row at a time per warp (low register pressure!)
// - State in registers (no spills!)
// Expected: ~60-100 μs

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif
#ifndef M_TILE
#define M_TILE 64
#endif
#ifndef N_TILE
#define N_TILE 64
#endif
#ifndef PAD
#define PAD 8
#endif

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask);
  }
  return v;
}

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
flash3_fused_attention_fp16_kernel_v5(
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

  extern __shared__ char smem_bytes[];
  
  // Layout shared memory carefully
  half* smem_Q = reinterpret_cast<half*>(smem_bytes);
  half* smem_K = smem_Q + M_TILE * LDQ;
  half* smem_V = smem_K + N_TILE * LDK;
  
  // Float state after half buffers (ensure 4-byte alignment)
  size_t float_offset = (M_TILE * LDQ + N_TILE * LDK + N_TILE * LDV) * sizeof(half);
  float_offset = (float_offset + 3) & ~3;  // Align to 4 bytes
  
  float* smem_m = reinterpret_cast<float*>(smem_bytes + float_offset);
  float* smem_l = smem_m + M_TILE;
  float* smem_O = smem_l + M_TILE;

  // Load Q tile ONCE
  {
    const int q_elems = q_tile_len * D;
    for (int idx = threadIdx.x; idx < q_elems; idx += blockDim.x) {
      const int r = idx / D;
      const int d = idx % D;
      smem_Q[r * LDQ + d] = Q_bh[(q_tile_start + r) * D + d];
    }
  }

  // Initialize state in shared memory
  for (int r = threadIdx.x; r < q_tile_len; r += blockDim.x) {
    smem_m[r] = -INFINITY;
    smem_l[r] = 0.0f;
  }
  for (int idx = threadIdx.x; idx < q_tile_len * D; idx += blockDim.x) {
    const int r = idx / D;
    const int d = idx % D;
    smem_O[r * (D + PAD) + d] = 0.0f;
  }

  __syncthreads();

  const int elems_per_lane = D / 32;

  // === CORRECT ARCHITECTURE: K/V outer loop ===
  for (int k_start = 0; k_start < S; k_start += N_TILE) {
    const int k_len = min(N_TILE, S - k_start);

    // ALL WARPS LOAD K/V TILE (ONCE!)
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

    // === Inner loop: Each warp processes its rows ===
    for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
      const int q_abs = q_tile_start + q_local;

      // Load Q row and state into registers (THIS warp only)
      float q_reg[4];  // For D=64
      #pragma unroll
      for (int c = 0; c < elems_per_lane; ++c) {
        const int d_idx = c * 32 + lane_id;
        q_reg[c] = __half2float(smem_Q[q_local * LDQ + d_idx]);
      }

      float m_i = smem_m[q_local];
      float l_i = smem_l[q_local];

      float out_acc[4];
      #pragma unroll
      for (int c = 0; c < elems_per_lane; ++c) {
        const int d_idx = c * 32 + lane_id;
        out_acc[c] = smem_O[q_local * (D + PAD) + d_idx];
      }

      // Process each key in this tile
      for (int j = 0; j < k_len; ++j) {
        const int key_abs = k_start + j;

        // Q·K[j]
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

        // Online softmax
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float beta = expf(score - m_new);
        float l_new = l_i * alpha + beta;

        // Update output
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          const int d_idx = c * 32 + lane_id;
          float v_val = __half2float(smem_V[j * LDV + d_idx]);
          out_acc[c] = out_acc[c] * alpha + beta * v_val;
        }

        m_i = m_new;
        l_i = l_new;
      }

      // Write state back to shared memory
      if (lane_id == 0) {
        smem_m[q_local] = m_i;
        smem_l[q_local] = l_i;
      }

      #pragma unroll
      for (int c = 0; c < elems_per_lane; ++c) {
        const int d_idx = c * 32 + lane_id;
        smem_O[q_local * (D + PAD) + d_idx] = out_acc[c];
      }
    }

    __syncthreads();  // Wait before loading next K/V tile
  }

  // === Normalize and write output ===
  for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
    const int q_abs = q_tile_start + q_local;
    float l_i = smem_l[q_local];

    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      const int d_idx = c * 32 + lane_id;
      float out_val = smem_O[q_local * (D + PAD) + d_idx];
      out_val = (l_i > 0.0f) ? (out_val / l_i) : 0.0f;
      O_bh[q_abs * D + d_idx] = __float2half(out_val);
    }
  }
}

// -------------------- Host launcher --------------------
void launch_flash3_v5(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    bool is_causal,
    cudaStream_t stream)
{
  const float scale = 1.0f / sqrtf((float)D);

  dim3 block(WARPS_PER_BLOCK * 32);
  dim3 grid((S + M_TILE - 1) / M_TILE, H, B);

  // Shared memory: Q + K + V (half) + state m, l, O (float)
  const size_t half_elems =
      (size_t)M_TILE * (D + PAD)       // Q
    + (size_t)N_TILE * (D + PAD)       // K
    + (size_t)N_TILE * (D + PAD);      // V
  
  const size_t float_elems =
      (size_t)M_TILE                   // m
    + (size_t)M_TILE                   // l
    + (size_t)M_TILE * (D + PAD);      // O accumulator

  const size_t smem_bytes = half_elems * sizeof(half) + float_elems * sizeof(float) + 4;  // +4 for alignment

  cudaFuncSetAttribute(
      flash3_fused_attention_fp16_kernel_v5,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);

  flash3_fused_attention_fp16_kernel_v5<<<grid, block, smem_bytes, stream>>>(
      Q, K, V, O, B, H, S, D, (int)is_causal, scale);
}

