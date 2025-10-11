// BF16-suppressing macros OK here (FP16-only TU)
#define CUDA_NO_BFLOAT16
#define __CUDA_NO_BFLOAT16_OPERATORS__

// 1. DTYPE HEADER FIRST (before any templates)
#include <cuda_fp16.h>

// 2. Then template header
#include "flash_attention_core.h"

// Defensive: dtype size sanity checks (catches toolchain issues)
static_assert(sizeof(half) == 2, "unexpected half size - check CUDA version");

namespace flashmoe {

// Specialize MathOps for half
template<>
struct MathOps<half> {
    // Defensive: ensure device compilation for MathOps
    #ifdef __CUDA_ARCH__
    static_assert(__CUDA_ARCH__ >= 700, "MathOps<half> requires SM70+");
    #endif
    
    __device__ __forceinline__ static half add(half a, half b) {
        return __hadd(a, b);
    }
    __device__ __forceinline__ static half mul(half a, half b) {
        return __hmul(a, b);
    }
    __device__ __forceinline__ static half sub(half a, half b) {
        return __hsub(a, b);
    }
    __device__ __forceinline__ static half div(half a, half b) {
        return __hdiv(a, b);
    }
    __device__ __forceinline__ static float to_float(half x) {
        return __half2float(x);
    }
    __device__ __forceinline__ static half from_float(float x) {
        return __float2half(x);
    }
};

// Explicit instantiation for half
template __global__ void flash_attention_kernel<half>(
    const half*, const half*, const half*, half*,
    int, int, int, int
);

template void flash_attention_forward<half>(
    const half*, const half*, const half*, half*,
    int, int, int, int, cudaStream_t
);

} // namespace flashmoe

// C-linkage wrapper with explicit symbol visibility
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("default")))
#endif
extern "C" void flash_attention_forward_fp16(
    const void* Q, const void* K, const void* V, void* O,
    int M, int N, int K_dim, int tile_size, cudaStream_t stream
) {
    flashmoe::flash_attention_forward<half>(
        reinterpret_cast<const half*>(Q),
        reinterpret_cast<const half*>(K),
        reinterpret_cast<const half*>(V),
        reinterpret_cast<half*>(O),
        M, N, K_dim, tile_size, stream
    );
}

