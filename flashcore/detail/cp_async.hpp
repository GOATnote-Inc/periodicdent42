#pragma once

#include <cuda_runtime.h>

namespace flashcore {
namespace detail {

template <int BYTES>
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                  "cp.async only supports 4, 8, or 16 byte transfers");
    
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n" ::
            "r"(smem_addr), "l"(src), "n"(BYTES));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

}  // namespace detail
}  // namespace flashcore

