#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cassert>

// --- Tunables (L4 optimized) ---
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

// Sim-FP8 dequant (E4M3 simulation)
__device__ __forceinline__ float dequant_sim_fp8(uint8_t u, float s){
    float x = (float(u) / 255.0f) * (2.0f * 448.0f) - 448.0f;
    return x * s;
}

// cp.async helper (address-space correct for Ampere/Ada)
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__
void cp_async_4B(void* smem_dst, const void* gmem_src) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem), "l"(gmem_src)
    );
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
__global__ void sdpa_fp8_coalesced_kernel(
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

    // --- Shared memory ---
    __shared__ uint8_t sQ[TILE_M][D_PAD];
    __shared__ uint8_t sK[TILE_N][D_PAD];
    __shared__ uint8_t sV[TILE_N][D_PAD];
    __shared__ float   m_smem[TILE_M];
    __shared__ float   l_smem[TILE_M];
    __shared__ float   U_smem[TILE_M][D_PAD];

    // --- Load Q tile (coalesced scalar loads) ---
    // Each thread loads consecutive elements → hardware coalesces into 128B transactions
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

    const int nTiles = (S + TILE_N - 1) / TILE_N;

    for (int t = 0; t < nTiles; ++t) {
        const int kv_start = t * TILE_N;
        const int kv_end   = min(kv_start + TILE_N, S);
        const int kv_len   = kv_end - kv_start;

        // --- Load K/V tile (coalesced with cp.async for Ampere/Ada) ---
#if __CUDA_ARCH__ >= 800
        // cp.async path: 4-byte granularity (uint32 = 4× uint8)
        // Each thread copies 4 consecutive uint8 elements
        const int elems_per_thread = 4;
        const int total_4B_chunks = (kv_len * D + elems_per_thread - 1) / elems_per_thread;
        
        for (int idx = tid; idx < total_4B_chunks; idx += blockDim.x) {
            int elem_offset = idx * elems_per_thread;
            int n = elem_offset / D;
            int d = elem_offset % D;
            
            if (n < kv_len && d + elems_per_thread <= D) {
                // Safe 4-byte copy
                const uint8_t* k_src = Kbh + (size_t)(kv_start + n) * D + d;
                const uint8_t* v_src = Vbh + (size_t)(kv_start + n) * D + d;
                
                cp_async_4B(&sK[n][d], k_src);
                cp_async_4B(&sV[n][d], v_src);
            } else if (n < kv_len) {
                // Tail: scalar copy
                for (int dd = d; dd < D && dd < d + elems_per_thread; ++dd) {
                    sK[n][dd] = Kbh[(size_t)(kv_start + n) * D + dd];
                    sV[n][dd] = Vbh[(size_t)(kv_start + n) * D + dd];
                }
            }
        }
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();
#else
        // Synchronous coalesced loads (fallback for sm_75)
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int n = idx / D;
            int d = idx % D;
            sK[n][d] = Kbh[(size_t)(kv_start + n) * D + d];
            sV[n][d] = Vbh[(size_t)(kv_start + n) * D + d];
        }
        __syncthreads();
#endif

        // --- Each warp processes multiple rows ---
        for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
            // Optimization: dequantize Q once per lane for this row
            float q_lane[2];  // D=64, 2 elements per lane (lane, lane+32)
            q_lane[0] = dequant_sim_fp8(sQ[r][lane], q_s);
            q_lane[1] = (lane + 32 < D) ? dequant_sim_fp8(sQ[r][lane + 32], q_s) : 0.f;

            // 1) compute scores for this row vs current K tile
            float S_row[TILE_N];

            for (int n = 0; n < kv_len; ++n) {
                float acc = 0.f;
                
                // Split D across lanes
                acc += q_lane[0] * dequant_sim_fp8(sK[n][lane], k_s);
                if (lane + 32 < D) {
                    acc += q_lane[1] * dequant_sim_fp8(sK[n][lane + 32], k_s);
                }
                
                acc = warp_reduce_sum(acc);
                if (lane == 0) acc *= softmax_scale;
                // Broadcast lane 0 result to all lanes
                S_row[n] = __shfl_sync(0xffffffff, acc, 0);
            }

            // 2) online softmax stats update (m,l) and scale old U
            float m_old = m_smem[r];
            float m_new = m_old;
            for (int n = 0; n < kv_len; ++n) {
                m_new = fmaxf(m_new, S_row[n]);
            }

            float l_old = l_smem[r];
            float l_add = 0.f;
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
            for (int n = 0; n < kv_len; ++n) {
                float p = S_row[n];  // unnormalized
                
                for (int d0 = lane; d0 < D; d0 += 32) {
                    float v = dequant_sim_fp8(sV[n][d0], v_s);
                    U_smem[r][d0] += p * v;
                }
            }

            if (lane == 0) { 
                m_smem[r] = m_new; 
                l_smem[r] = l_new; 
            }
        }
        __syncthreads();
    }

    // 4) write O = U / l
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        float l_final = l_smem[r];
        // Numerical robustness: guard against pathological inputs
        l_final = (l_final > 0.f) ? l_final : 1e-8f;
        
        half* out = Obh + (size_t)(q_start + r) * D;
        
        for (int d0 = lane; d0 < D; d0 += 32) {
            float o = U_smem[r][d0] / l_final;
            out[d0] = __float2half(o);
        }
    }
}

// Launcher
extern "C" void launch_sdpa_fp8_coalesced(
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
    
    sdpa_fp8_coalesced_kernel<<<grid, block, 0, stream>>>(
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

