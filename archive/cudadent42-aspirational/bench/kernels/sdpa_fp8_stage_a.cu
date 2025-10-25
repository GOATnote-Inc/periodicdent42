#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// --- Tunables (L4 optimized, Stage A) ---
#define HEAD_DIM 64
#define TILE_M   32
#define TILE_N   64
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define D_PAD (((HEAD_DIM + 15) / 16) * 16 + 16)  // 16B aligned + padding = 80

// Warp reductions
__device__ __forceinline__ float warp_reduce_sum(float v){
    #pragma unroll
    for (int o=16; o>0; o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// Sim-FP8 dequant (for Q only, K/V use LUT)
__device__ __forceinline__ float dequant_sim_fp8(uint8_t u, float s){
    float x = (float(u) / 255.0f) * (2.0f * 448.0f) - 448.0f;
    return x * s;
}

// cp.async helpers (16B preferred, 8B/4B fallback)
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__
void cp_async_16B(void* smem_dst, const void* gmem_src) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem_src));
}

__device__ __forceinline__
void cp_async_8B(void* smem_dst, const void* gmem_src) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(smem), "l"(gmem_src));
}

__device__ __forceinline__
void cp_async_4B(void* smem_dst, const void* gmem_src) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
#endif

__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void sdpa_fp8_stage_a_kernel(
    const uint8_t* __restrict__ Q,   // [B,H,S,D]
    const uint8_t* __restrict__ K,   // [B,H,S,D]
    const uint8_t* __restrict__ V,   // [B,H,S,D]
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

    // --- Shared memory (double-buffered K/V + LUTs) ---
    __shared__ alignas(16) uint8_t sQ[TILE_M][D_PAD];
    __shared__ alignas(16) uint8_t sK[2][TILE_N][D_PAD];  // 2-stage
    __shared__ alignas(16) uint8_t sV[2][TILE_N][D_PAD];  // 2-stage
    __shared__ float   kLUT[256];  // K dequant lookup
    __shared__ float   vLUT[256];  // V dequant lookup
    __shared__ float   m_smem[TILE_M];
    __shared__ float   l_smem[TILE_M];
    __shared__ alignas(16) float U_smem[TILE_M][D_PAD];

    // --- Build LUTs once per block ---
    if (tid < 256) {
        const int u = tid;
        float x = (float(u) / 255.0f) * (2.0f * 448.0f) - 448.0f;
        kLUT[u] = x * k_s;
        vLUT[u] = x * v_s;
    }

    // --- Load Q tile (coalesced) ---
    for (int idx = tid; idx < rows_in_tile * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        sQ[r][d] = Qbh[(size_t)(q_start + r) * D + d];
    }

    // Init per-row stats and U accumulator
    for (int idx = tid; idx < rows_in_tile * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        U_smem[r][d] = 0.f;
    }
    for (int r = tid; r < rows_in_tile; r += blockDim.x) {
        m_smem[r] = -INFINITY;
        l_smem[r] = 0.f;
    }
    __syncthreads();

    // --- Prefetch helper (16B cp.async when possible) ---
    auto prefetch_tile = [&](int t, int stage) {
#if __CUDA_ARCH__ >= 800
        const int kv_start = t * TILE_N;
        const int kv_len   = min(TILE_N, S - kv_start);
        const int CHUNK = 16;  // prefer 16B
        const int total_bytes = kv_len * D;

        for (int off = tid * CHUNK; off < total_bytes; off += blockDim.x * CHUNK) {
            int n = off / D;
            int d = off % D;

            const uint8_t* k_src = Kbh + (size_t)(kv_start + n) * D + d;
            const uint8_t* v_src = Vbh + (size_t)(kv_start + n) * D + d;

            if (d + CHUNK <= D && n < kv_len) {
                // Safe 16B copy
                cp_async_16B(&sK[stage][n][d], k_src);
                cp_async_16B(&sV[stage][n][d], v_src);
            } else if (d + 8 <= D && n < kv_len) {
                // 8B fallback
                cp_async_8B(&sK[stage][n][d], k_src);
                cp_async_8B(&sV[stage][n][d], v_src);
            } else if (n < kv_len) {
                // Tail: scalar
                for (int dd = d; dd < D && dd < d + CHUNK; ++dd) {
                    sK[stage][n][dd] = k_src[dd - d];
                    sV[stage][n][dd] = v_src[dd - d];
                }
            }
        }
        cp_async_commit_group();
#else
        // Synchronous fallback
        const int kv_start = t * TILE_N;
        const int kv_len   = min(TILE_N, S - kv_start);
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int n = idx / D, d = idx % D;
            sK[stage][n][d] = Kbh[(size_t)(kv_start + n) * D + d];
            sV[stage][n][d] = Vbh[(size_t)(kv_start + n) * D + d];
        }
#endif
    };

    // --- Pipeline: prefetch tile 0, then loop with overlap ---
    const int nTiles = (S + TILE_N - 1) / TILE_N;
    int stage = 0;
    
    // Prefetch tile 0
    if (nTiles > 0) {
        prefetch_tile(0, stage);
#if __CUDA_ARCH__ >= 800
        cp_async_wait_group<0>();  // Ensure tile 0 visible
#endif
        __syncthreads();
    }

    for (int t = 0; t < nTiles; ++t) {
        const int read = stage;
        const int next = stage ^ 1;
        const int kv_start = t * TILE_N;
        const int kv_end   = min(kv_start + TILE_N, S);
        const int kv_len   = kv_end - kv_start;

        // Prefetch next tile while computing current
        if (t + 1 < nTiles) {
            prefetch_tile(t + 1, next);
        }

        // --- Compute using sK[read], sV[read] with LUT dequant ---
        for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
            // Q dequant once per lane for this row
            float q_lane[2];
            q_lane[0] = dequant_sim_fp8(sQ[r][lane], q_s);
            q_lane[1] = (lane + 32 < D) ? dequant_sim_fp8(sQ[r][lane + 32], q_s) : 0.f;

            // 1) Compute scores using K LUT
            float S_row[TILE_N];
            for (int n = 0; n < kv_len; ++n) {
                float acc = q_lane[0] * kLUT[sK[read][n][lane]];
                if (lane + 32 < D) {
                    acc += q_lane[1] * kLUT[sK[read][n][lane + 32]];
                }
                
                acc = warp_reduce_sum(acc);
                if (lane == 0) acc *= softmax_scale;
                S_row[n] = __shfl_sync(0xffffffff, acc, 0);
            }

            // 2) Online softmax stats update
            float m_old = m_smem[r];
            float m_new = m_old;
            for (int n = 0; n < kv_len; ++n) {
                m_new = fmaxf(m_new, S_row[n]);
            }

            float l_old = l_smem[r];
            float l_add = 0.f;
            for (int n = 0; n < kv_len; ++n) {
                S_row[n] = __expf(S_row[n] - m_new);
                l_add += S_row[n];
            }
            float rescale = __expf(m_old - m_new);
            float l_new   = l_old * rescale + l_add;

            // Scale previous U
            for (int d0 = lane; d0 < D; d0 += 32) {
                U_smem[r][d0] *= rescale;
            }

            // 3) Accumulate P·V using V LUT
            for (int n = 0; n < kv_len; ++n) {
                float p = S_row[n];  // unnormalized
                for (int d0 = lane; d0 < D; d0 += 32) {
                    float v = vLUT[sV[read][n][d0]];
                    U_smem[r][d0] += p * v;
                }
            }

            if (lane == 0) { 
                m_smem[r] = m_new; 
                l_smem[r] = l_new; 
            }
        }

        // Wait for prefetch to complete before next iteration
#if __CUDA_ARCH__ >= 800
        cp_async_wait_group<0>();
#endif
        __syncthreads();
        stage = next;
    }

    // 4) Write O = U / l
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        float l_final = l_smem[r];
        l_final = (l_final > 0.f) ? l_final : 1e-8f;
        
        half* out = Obh + (size_t)(q_start + r) * D;
        for (int d0 = lane; d0 < D; d0 += 32) {
            float o = U_smem[r][d0] / l_final;
            out[d0] = __float2half(o);
        }
    }
}

// Launcher
extern "C" void launch_sdpa_fp8_stage_a(
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
    
    sdpa_fp8_stage_a_kernel<<<grid, block, 0, stream>>>(
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

