// ⚠️ CRITICAL: NO BF16-suppressing macros here!
// Guard against accidental macro leakage from build system
#ifdef CUDA_NO_BFLOAT16
#error "CUDA_NO_BFLOAT16 leaked into BF16 TU - check setup.py per-file flags!"
#endif
#ifdef __CUDA_NO_BFLOAT16_OPERATORS__
#error "__CUDA_NO_BFLOAT16_OPERATORS__ leaked into BF16 TU - check setup.py!"
#endif

// 1. BF16 HEADER FIRST
#include <cuda_bf16.h>

// 2. Template header
#include "flash_attention_core.h"

// Defensive: dtype size sanity checks
static_assert(sizeof(__nv_bfloat16) == 2, "unexpected bfloat16 size - check CUDA version");

namespace flashmoe {

// Specialize for __nv_bfloat16
template<>
struct MathOps<__nv_bfloat16> {
    // Defensive: ensure device compilation for MathOps
    #ifdef __CUDA_ARCH__
    static_assert(__CUDA_ARCH__ >= 800, "MathOps<__nv_bfloat16> requires SM80+ (Ampere)");
    #endif
    
    __device__ __forceinline__ static __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
        return __hadd(a, b);  // BF16 variant
    }
    __device__ __forceinline__ static __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
        return __hmul(a, b);
    }
    __device__ __forceinline__ static __nv_bfloat16 sub(__nv_bfloat16 a, __nv_bfloat16 b) {
        return __hsub(a, b);
    }
    __device__ __forceinline__ static __nv_bfloat16 div(__nv_bfloat16 a, __nv_bfloat16 b) {
        return __hdiv(a, b);
    }
    __device__ __forceinline__ static float to_float(__nv_bfloat16 x) {
        return __bfloat162float(x);
    }
    __device__ __forceinline__ static __nv_bfloat16 from_float(float x) {
        return __float2bfloat16(x);
    }
};

// Explicit instantiation
template __global__ void flash_attention_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int
);

template void flash_attention_forward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, cudaStream_t
);

} // namespace flashmoe

// Unique symbol with explicit visibility
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("default")))
#endif
extern "C" void flash_attention_forward_bf16(
    const void* Q, const void* K, const void* V, void* O,
    int M, int N, int K_dim, int tile_size, cudaStream_t stream
) {
    flashmoe::flash_attention_forward<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(Q),
        reinterpret_cast<const __nv_bfloat16*>(K),
        reinterpret_cast<const __nv_bfloat16*>(V),
        reinterpret_cast<__nv_bfloat16*>(O),
        M, N, K_dim, tile_size, stream
    );
}

