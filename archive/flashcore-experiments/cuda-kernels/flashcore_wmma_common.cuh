#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "detail/cp_async.hpp"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define FLASHCORE_CP_ASYNC_SUPPORTED 1
#else
#define FLASHCORE_CP_ASYNC_SUPPORTED 0
#endif

namespace flashcore {

constexpr int kWarpSize = 32;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr bool kCpAsyncSupported = FLASHCORE_CP_ASYNC_SUPPORTED != 0;

__device__ __forceinline__ void cp_async_commit() {
    if constexpr (kCpAsyncSupported) {
        detail::cp_async_commit_group();
    }
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    if constexpr (kCpAsyncSupported) {
        detail::cp_async_wait_group<N>();
    }
}

__device__ __forceinline__ void cp_async_fence() {
    if constexpr (kCpAsyncSupported) {
        detail::cp_async_wait_all();
    }
}

__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    if constexpr (kCpAsyncSupported) {
        detail::cp_async_cg<16>(dst, src);
    } else {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    }
}

}  // namespace flashcore
