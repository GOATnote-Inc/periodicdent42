#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// --- Tunables (keep these for L4 correctness path) ---
#define HEAD_DIM 64
#define TILE_M   32
#define TILE_N   64
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define D_PAD (HEAD_DIM + 8)  // 16B bank-friendly padding

// Warp reductions
__device__ __forceinline__ float warp_reduce_max(float v){
    #pragma unroll
    for (int o=16; o>0; o>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v){
    #pragma unroll
    for (int o=16; o>0; o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// Your "sim-FP8" dequant (matches your test harness)
__device__ __forceinline__ float dequant_sim_fp8(uint8_t u, float s){
    float x = (float(u) / 255.0f) * (2.0f * 448.0f) - 448.0f;
    return x * s;
}

__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void sdpa_fp8_fixed_kernel(
    const uint8_t* __restrict__ Q,   // [B,H,S,D]
    const uint8_t* __restrict__ K,   // [B,H,S,D]
    const uint8_t* __restrict__ V,   // [B,H,S,Dv] with Dv==D here
    const float*   __restrict__ Qs,  // [H]
    const float*   __restrict__ Ks,  // [H]
    const float*   __restrict__ Vs,  // [H]
    half*          __restrict__ O,   // [B,H,S,D]
    int B, int H, int S, int D, float softmax_scale
){
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int q_block = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int q_start = q_block * TILE_M;
    const int q_end   = min(q_start + TILE_M, S);
    const int rows_in_tile = q_end - q_start;
    if (rows_in_tile <= 0) return;

    const size_t BHSD = (size_t)S * D;
    const uint8_t* Qbh = Q + ((size_t)b * H + h) * BHSD;
    const uint8_t* Kbh = K + ((size_t)b * H + h) * BHSD;
    const uint8_t* Vbh = V + ((size_t)b * H + h) * BHSD;
    half*         Obh  = O + ((size_t)b * H + h) * BHSD;

    const float q_s = Qs[h], k_s = Ks[h], v_s = Vs[h];

    // --- Shared memory ---
    __shared__ uint8_t sQ[TILE_M][D_PAD];        // Q tile (uint8)
    __shared__ uint8_t sK[TILE_N][D_PAD];        // double-buffer not required for correctness
    __shared__ uint8_t sV[TILE_N][D_PAD];
    __shared__ float   m_smem[TILE_M];
    __shared__ float   l_smem[TILE_M];
    __shared__ float   U_smem[TILE_M][D_PAD];    // unnormalized numerator

    // Load Q tile
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        const uint8_t* qrow = Qbh + (size_t)(q_start + r) * D;
        for (int d0 = lane; d0 < D; d0 += 32) {
            sQ[r][d0] = qrow[d0];
        }
    }

    // Init per-row stats and U accumulator
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        if (lane == 0) { m_smem[r] = -INFINITY; l_smem[r] = 0.f; }
        for (int d0 = lane; d0 < D; d0 += 32) {
            U_smem[r][d0] = 0.f;
        }
    }
    __syncthreads();

    const int nTiles = (S + TILE_N - 1) / TILE_N;

    for (int t = 0; t < nTiles; ++t) {
        const int kv_start = t * TILE_N;
        const int kv_end   = min(kv_start + TILE_N, S);
        const int kv_len   = kv_end - kv_start;

        // Load K/V tile
        for (int n = tid; n < kv_len * D; n += blockDim.x) {
            int row = n / D;
            int d0  = n % D;
            sK[row][d0] = Kbh[(size_t)(kv_start + row) * D + d0];
            sV[row][d0] = Vbh[(size_t)(kv_start + row) * D + d0];
        }
        __syncthreads();

        // Each warp processes 4 rows: r = warp_id, warp_id+NUM_WARPS, ...
        for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
            // 1) compute scores for this row vs current K tile
            float S_row[TILE_N];

            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                float acc = 0.f;
                for (int d0 = lane; d0 < D; d0 += 32) {
                    float q = dequant_sim_fp8(sQ[r][d0], q_s);
                    float k = dequant_sim_fp8(sK[n][d0], k_s);
                    acc += q * k;
                }
                acc = warp_reduce_sum(acc);
                if (lane == 0) acc *= softmax_scale;
                // Broadcast lane 0 result so all lanes have the same scalar
                S_row[n] = __shfl_sync(0xffffffff, acc, 0);
            }

            // 2) online softmax stats update (m,l) and scale old U
            float m_old = m_smem[r];
            float m_new = m_old;
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) m_new = fmaxf(m_new, S_row[n]);

            float l_old = l_smem[r];
            float l_add = 0.f;
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                S_row[n] = __expf(S_row[n] - m_new); // reuse S_row to hold p (unnormalized)
                l_add += S_row[n];
            }
            float rescale = __expf(m_old - m_new);
            float l_new   = l_old * rescale + l_add;

            // Scale previous U by exp(m_old - m_new)
            for (int d0 = lane; d0 < D; d0 += 32) {
                U_smem[r][d0] *= rescale;
            }

            // 3) accumulate P·V (still unnormalized)
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                float p = S_row[n];  // unnormalized
                for (int d0 = lane; d0 < D; d0 += 32) {
                    float v = dequant_sim_fp8(sV[n][d0], v_s);
                    U_smem[r][d0] += p * v;
                }
            }

            if (lane == 0) { m_smem[r] = m_new; l_smem[r] = l_new; }
        }
        __syncthreads();
    }

    // 4) write O = U / l
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        float l_final = l_smem[r];
        half* out = Obh + (size_t)(q_start + r) * D;
        for (int d0 = lane; d0 < D; d0 += 32) {
            float o = U_smem[r][d0] / l_final;
            out[d0] = __float2half(o);
        }
    }
}

// Launcher
extern "C" void launch_sdpa_fp8_fixed(
    const void* Q,
    const void* K,
    const void* V,
    const float* Q_scale,
    const float* K_scale,
    const float* V_scale,
    half* O,
    int B, int H, int S, int D,
    float softmax_scale,
    cudaStream_t stream
) {
    dim3 grid((S + TILE_M - 1) / TILE_M, H, B);
    dim3 block(THREADS_PER_BLOCK);
    
    sdpa_fp8_fixed_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(Q),
        reinterpret_cast<const uint8_t*>(K),
        reinterpret_cast<const uint8_t*>(V),
        Q_scale,
        K_scale,
        V_scale,
        O,
        B, H, S, D,
        softmax_scale
    );
}

