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

static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be 16-byte aligned for int4 loads");

// half4 helper (not in cuda_fp16.h)
struct __align__(8) half4 {
    half x, y, z, w;
};

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
void cp_async_16B(void* smem_dst, const void* gmem_src) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}
#endif

__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void sdpa_fp8_vectorized_kernel(
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

    // --- Shared memory (explicitly aligned) ---
    __shared__ __align__(16) uint8_t sQ[TILE_M][D_PAD];
    __shared__ __align__(16) uint8_t sK[TILE_N][D_PAD];
    __shared__ __align__(16) uint8_t sV[TILE_N][D_PAD];
    __shared__ float   m_smem[TILE_M];
    __shared__ float   l_smem[TILE_M];
    __shared__ __align__(16) float U_smem[TILE_M][D_PAD];

    // --- Load Q tile (vectorized with int4) ---
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        const uint8_t* qrow = Qbh + (size_t)(q_start + r) * D;
        // Load 16 bytes per iteration (int4 = 4 × int32 = 16 bytes)
        for (int d0 = lane * 16; d0 < D; d0 += 32 * 16) {
            if (d0 + 16 <= D) {
                *reinterpret_cast<int4*>(&sQ[r][d0]) = 
                    *reinterpret_cast<const int4*>(&qrow[d0]);
            } else {
                // Tail handling
                for (int dd = d0; dd < D; ++dd) {
                    sQ[r][dd] = qrow[dd];
                }
            }
        }
    }

    // Init per-row stats and U accumulator
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        if (lane == 0) { 
            m_smem[r] = -INFINITY; 
            l_smem[r] = 0.f; 
        }
        // Vectorized zero init for U
        for (int d0 = lane * 16; d0 < D; d0 += 32 * 16) {
            if (d0 + 16 <= D) {
                float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(&U_smem[r][d0]) = zero;
            } else {
                for (int dd = d0; dd < D; ++dd) {
                    U_smem[r][dd] = 0.f;
                }
            }
        }
    }
    __syncthreads();

    const int nTiles = (S + TILE_N - 1) / TILE_N;

    for (int t = 0; t < nTiles; ++t) {
        const int kv_start = t * TILE_N;
        const int kv_end   = min(kv_start + TILE_N, S);
        const int kv_len   = kv_end - kv_start;

        // --- Load K/V tile (vectorized with int4) ---
#if __CUDA_ARCH__ >= 800
        // cp.async path (Ampere/Ada)
        for (int idx = tid; idx < kv_len * (D / 16); idx += blockDim.x) {
            int n = idx / (D / 16);
            int d_chunk = (idx % (D / 16)) * 16;
            
            const uint8_t* k_src = Kbh + (size_t)(kv_start + n) * D + d_chunk;
            const uint8_t* v_src = Vbh + (size_t)(kv_start + n) * D + d_chunk;
            
            cp_async_16B(&sK[n][d_chunk], k_src);
            cp_async_16B(&sV[n][d_chunk], v_src);
        }
        cp_async_commit_group();
        cp_async_wait_group(0);
        __syncthreads();
#else
        // Synchronous vectorized loads (fallback for sm_75)
        for (int idx = tid; idx < kv_len * (D / 16); idx += blockDim.x) {
            int n = idx / (D / 16);
            int d_chunk = (idx % (D / 16)) * 16;
            
            if (d_chunk + 16 <= D) {
                *reinterpret_cast<int4*>(&sK[n][d_chunk]) = 
                    *reinterpret_cast<const int4*>(&Kbh[(size_t)(kv_start + n) * D + d_chunk]);
                *reinterpret_cast<int4*>(&sV[n][d_chunk]) = 
                    *reinterpret_cast<const int4*>(&Vbh[(size_t)(kv_start + n) * D + d_chunk]);
            }
        }
        __syncthreads();
#endif

        // --- Each warp processes multiple rows ---
        for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
            // Optimization: dequantize Q once per lane for this row
            float q_lane[4];  // Up to 4 × 32 = 128 positions (covers D=64 with 2)
            const int num_chunks = (D + 31) / 32;
            #pragma unroll
            for (int c = 0; c < num_chunks; ++c) {
                int d0 = lane + c * 32;
                q_lane[c] = (d0 < D) ? dequant_sim_fp8(sQ[r][d0], q_s) : 0.f;
            }

            // 1) compute scores for this row vs current K tile
            float S_row[TILE_N];

            #pragma unroll 4
            for (int n = 0; n < kv_len; ++n) {
                float acc = 0.f;
                #pragma unroll
                for (int c = 0; c < num_chunks; ++c) {
                    int d0 = lane + c * 32;
                    if (d0 < D) {
                        float k = dequant_sim_fp8(sK[n][d0], k_s);
                        acc += q_lane[c] * k;
                    }
                }
                acc = warp_reduce_sum(acc);
                if (lane == 0) acc *= softmax_scale;
                // Broadcast lane 0 result to all lanes
                S_row[n] = __shfl_sync(0xffffffff, acc, 0);
            }

            // 2) online softmax stats update (m,l) and scale old U
            float m_old = m_smem[r];
            float m_new = m_old;
            #pragma unroll 4
            for (int n = 0; n < kv_len; ++n) {
                m_new = fmaxf(m_new, S_row[n]);
            }

            float l_old = l_smem[r];
            float l_add = 0.f;
            #pragma unroll 4
            for (int n = 0; n < kv_len; ++n) {
                S_row[n] = __expf(S_row[n] - m_new); // reuse S_row to hold p (unnormalized)
                l_add += S_row[n];
            }
            float rescale = __expf(m_old - m_new);
            float l_new   = l_old * rescale + l_add;

            // Scale previous U by exp(m_old - m_new)
            #pragma unroll
            for (int c = 0; c < num_chunks; ++c) {
                int d0 = lane + c * 32;
                if (d0 < D) {
                    U_smem[r][d0] *= rescale;
                }
            }

            // 3) accumulate P·V (still unnormalized)
            #pragma unroll 4
            for (int n = 0; n < kv_len; ++n) {
                float p = S_row[n];  // unnormalized
                #pragma unroll
                for (int c = 0; c < num_chunks; ++c) {
                    int d0 = lane + c * 32;
                    if (d0 < D) {
                        float v = dequant_sim_fp8(sV[n][d0], v_s);
                        U_smem[r][d0] += p * v;
                    }
                }
            }

            if (lane == 0) { 
                m_smem[r] = m_new; 
                l_smem[r] = l_new; 
            }
        }
        __syncthreads();
    }

    // 4) write O = U / l (vectorized with float4)
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        float l_final = l_smem[r];
        // Numerical robustness: guard against pathological inputs
        l_final = (l_final > 0.f) ? l_final : 1e-8f;
        
        half* out = Obh + (size_t)(q_start + r) * D;
        
        // Vectorized write (4 × half = 8 bytes, use float2 for 2 × half2)
        for (int d0 = lane * 4; d0 < D; d0 += 32 * 4) {
            if (d0 + 4 <= D) {
                half4 h4;
                h4.x = __float2half(U_smem[r][d0 + 0] / l_final);
                h4.y = __float2half(U_smem[r][d0 + 1] / l_final);
                h4.z = __float2half(U_smem[r][d0 + 2] / l_final);
                h4.w = __float2half(U_smem[r][d0 + 3] / l_final);
                *reinterpret_cast<half4*>(&out[d0]) = h4;
            } else {
                // Tail
                for (int dd = d0; dd < D; ++dd) {
                    out[dd] = __float2half(U_smem[r][dd] / l_final);
                }
            }
        }
    }
}

// Launcher
extern "C" void launch_sdpa_fp8_vectorized(
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
#ifdef DEBUG_SDPA
    // Safety canaries (compile with -DDEBUG_SDPA)
    assert(D % 16 == 0 && "HEAD_DIM must be 16-byte aligned");
    assert((reinterpret_cast<uintptr_t>(Q) % 16) == 0 && "Q must be 16-byte aligned");
    assert((reinterpret_cast<uintptr_t>(K) % 16) == 0 && "K must be 16-byte aligned");
    assert((reinterpret_cast<uintptr_t>(V) % 16) == 0 && "V must be 16-byte aligned");
    assert((reinterpret_cast<uintptr_t>(O) % 16) == 0 && "O must be 16-byte aligned");
#endif

    dim3 grid((S + TILE_M - 1) / TILE_M, H, B);
    dim3 block(THREADS_PER_BLOCK);
    
    sdpa_fp8_vectorized_kernel<<<grid, block, 0, stream>>>(
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

