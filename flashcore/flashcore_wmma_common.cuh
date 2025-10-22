#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace flashcore {

constexpr int kWarpSize = 32;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

#if __CUDA_ARCH__ >= 800
#define FLASHCORE_CP_ASYNC_SUPPORTED 1
#else
#define FLASHCORE_CP_ASYNC_SUPPORTED 0
#endif

#if FLASHCORE_CP_ASYNC_SUPPORTED
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    constexpr int bytes = 16;
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n" ::
            "r"(smem_addr), "l"(src), "n"(bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N = 0>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.wait_all;\n" ::);
}
#else
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

__device__ __forceinline__ void cp_async_commit() {}

template <int N = 0>
__device__ __forceinline__ void cp_async_wait() {}

__device__ __forceinline__ void cp_async_fence() {}
#endif

}  // namespace flashcore

