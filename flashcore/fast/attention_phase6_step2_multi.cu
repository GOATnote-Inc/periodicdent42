// Step 2: Multiple WGMMAs (4× K-dimension loop)
// Target: 10-15 TFLOPS
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if __CUDA_ARCH__ < 900
#error "H100 only"
#endif

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;
constexpr int HEAD_DIM = 64;  // Full head dimension
constexpr int NUM_K_TILES = HEAD_DIM / TILE_K;  // 4 tiles

__device__ __forceinline__ 
uint64_t make_desc(const void* ptr, uint32_t ld, uint32_t swizzle = 3) {
    uint32_t addr = __cvta_generic_to_shared(ptr);
    uint32_t ld_units = (ld * sizeof(__half)) / 16;
    return (addr & 0xFFFFF) | ((uint64_t)(ld_units & 0x3FFF) << 32) | ((uint64_t)(swizzle & 0x7) << 46);
}

__device__ __forceinline__ 
void wgmma_m64n64k16(float acc[32], uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33;\n"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(desc_a), "l"(desc_b)
    );
}

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

__global__ void __launch_bounds__(256)
attention_step2_multi_wgmma(
    const __half* Q, const __half* K, const __half* V, float* O,
    int M, int N, int D
) {
    __shared__ __align__(128) __half smem_Q[64][32];
    __shared__ __align__(128) __half smem_K[64][32];
    
    int tid = threadIdx.x;
    int warp_group_id = tid / 128;
    
    if (warp_group_id == 0 && tid < 128) {
        float acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) acc[i] = 0.0f;
        
        // Loop over K dimension (4 tiles for D=64)
        for (int k_tile = 0; k_tile < NUM_K_TILES; k_tile++) {
            // Load Q tile: [64, 16] at offset k_tile*16
            for (int idx = tid; idx < 64*16; idx += 128) {
                int row = idx / 16;
                int col = idx % 16;
                int global_col = k_tile * 16 + col;
                smem_Q[row][col] = (global_col < D) ? Q[row * D + global_col] : __float2half(0.0f);
            }
            
            // Load K tile transposed: [64, 16] at offset k_tile*16
            for (int idx = tid; idx < 64*16; idx += 128) {
                int row = idx / 16;
                int col = idx % 16;
                int global_col = k_tile * 16 + col;
                smem_K[col][row] = (global_col < D) ? K[row * D + global_col] : __float2half(0.0f);
            }
            
            __syncthreads();
            
            wgmma_fence();
            uint64_t desc_q = make_desc(&smem_Q[0][0], 32, 3);
            uint64_t desc_k = make_desc(&smem_K[0][0], 32, 3);
            wgmma_m64n64k16(acc, desc_q, desc_k);
            wgmma_commit();
            wgmma_wait<0>();
            
            __syncthreads();
        }
        
        // Write output (simplified - use proper mapping in production)
        for (int i = 0; i < 32; i += 2) {
            int idx = tid * 32 + i;
            if (idx < 64*64) {
                O[idx] = acc[i];
                O[idx+1] = acc[i+1];
            }
        }
    }
}

extern "C" void launch_step2_multi(
    const void* Q, const void* K, const void* V, void* O,
    int M, int N, int D, cudaStream_t stream
) {
    attention_step2_multi_wgmma<<<1, 256, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, (float*)O, M, N, D
    );
}

