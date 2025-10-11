#pragma once
// Pure host header - NEVER includes CUDA dtype headers

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// Forward declare C-linkage functions with explicit visibility
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("default")))
#endif
extern "C" void flash_attention_forward_fp16(
    const void*, const void*, const void*, void*,
    int, int, int, int, cudaStream_t
);

#ifdef FLASHMOE_HAS_BF16
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("default")))
#endif
extern "C" void flash_attention_forward_bf16(
    const void*, const void*, const void*, void*,
    int, int, int, int, cudaStream_t
);
#endif

namespace flashmoe {

// Host-side dispatcher (no device types)
// FlashAttention uses dtype/arch-specific CUDA files (same pattern).
// Ref: https://github.com/Dao-AILab/flash-attention/tree/main/csrc
inline void flash_attention_dispatch(
    const void* Q, const void* K, const void* V, void* O,
    int M, int N, int K_dim, int tile_size,
    int dtype_id,  // 0=fp16, 1=bf16
    cudaStream_t stream
) {
    if (dtype_id == 0) {
        flash_attention_forward_fp16(Q, K, V, O, M, N, K_dim, tile_size, stream);
    }
#ifdef FLASHMOE_HAS_BF16
    else if (dtype_id == 1) {
        flash_attention_forward_bf16(Q, K, V, O, M, N, K_dim, tile_size, stream);
    }
#endif
    else {
        // Fail fast with clear diagnostic
        std::string error_msg = "Unsupported dtype_id=" + std::to_string(dtype_id) +
                                " (built with FLASHMOE_HAS_BF16=" +
#ifdef FLASHMOE_HAS_BF16
                                "1"
#else
                                "0"
#endif
                                ")";
        throw std::runtime_error(error_msg);
    }
}

} // namespace

