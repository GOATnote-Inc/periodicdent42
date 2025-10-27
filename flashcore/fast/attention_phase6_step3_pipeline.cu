// Step 3: Software Pipeline + Double Buffer
// Target: 30-40 TFLOPS
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if __CUDA_ARCH__ < 900
#error "H100 only"
#endif

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define HEAD_DIM 64
#define NUM_K_TILES (HEAD_DIM / TILE_K)

__device__ __forceinline__ 
uint64_t make_desc(const void* p, uint32_t ld) {
    uint32_t a = __cvta_generic_to_shared(p);
    return (a & 0xFFFFF) | (((uint64_t)((ld*2)/16) & 0x3FFF) << 32) | ((uint64_t)3 << 46);
}

__device__ __forceinline__ 
void wgmma(float a[32], uint64_t dA, uint64_t dB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33;\n"
        : "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]),
          "+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]),
          "+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]),
          "+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31])
        : "l"(dA), "l"(dB)
    );
}

__device__ __forceinline__ void fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__
void cp_async_cg_shared_global(__half* dst, const __half* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}

__device__ __forceinline__
void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template<int N>
__device__ __forceinline__
void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__global__ void __launch_bounds__(256)
attention_step3_pipeline(
    const __half* Q, const __half* K, float* O, int M, int N, int D
) {
    // Double-buffered shared memory
    __shared__ __align__(128) __half smem_Q[2][64][32];
    __shared__ __align__(128) __half smem_K[2][64][32];
    
    int tid = threadIdx.x;
    int buf = 0;  // Current buffer
    
    if (tid < 128) {
        float acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) acc[i] = 0.0f;
        
        // Prefetch first tile
        for (int idx = tid; idx < 64*16; idx += 128) {
            int r = idx / 16, c = idx % 16;
            cp_async_cg_shared_global(&smem_Q[0][r][c], &Q[r*D + c]);
            cp_async_cg_shared_global(&smem_K[0][c][r], &K[r*D + c]);
        }
        cp_async_commit();
        
        // Pipeline loop
        for (int k = 0; k < NUM_K_TILES; k++) {
            // Prefetch next tile while computing current
            if (k+1 < NUM_K_TILES) {
                for (int idx = tid; idx < 64*16; idx += 128) {
                    int r = idx / 16, c = idx % 16;
                    int next_buf = (buf + 1) & 1;
                    int k_offset = (k+1) * 16 + c;
                    cp_async_cg_shared_global(&smem_Q[next_buf][r][c], &Q[r*D + k_offset]);
                    cp_async_cg_shared_global(&smem_K[next_buf][c][r], &K[r*D + k_offset]);
                }
                cp_async_commit();
            }
            
            // Wait for current tile
            cp_async_wait<0>();
            __syncthreads();
            
            // Compute on current buffer
            fence();
            uint64_t dQ = make_desc(&smem_Q[buf][0][0], 32);
            uint64_t dK = make_desc(&smem_K[buf][0][0], 32);
            wgmma(acc, dQ, dK);
            commit();
            wait<0>();
            
            buf = (buf + 1) & 1;  // Flip buffer
        }
        
        // Write output
        for (int i = 0; i < 32; i++) {
            int idx = tid * 32 + i;
            if (idx < 64*64) O[idx] = acc[i];
        }
    }
}

extern "C" void launch_step3(const void* Q, const void* K, void* O, int M, int N, int D, cudaStream_t s) {
    attention_step3_pipeline<<<1, 256, 0, s>>>((const __half*)Q, (const __half*)K, (float*)O, M, N, D);
}

