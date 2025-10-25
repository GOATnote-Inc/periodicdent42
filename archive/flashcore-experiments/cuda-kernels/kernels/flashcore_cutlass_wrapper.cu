/*
 * FlashCore: CUTLASS FMHA Wrapper
 * Simplified wrapper around CUTLASS AttentionKernel for PyTorch integration
 * Target: <26 μs on NVIDIA L4 (sm_89, Ada architecture)
 * Shape: B=1, H=8, S=512, D=64
 */

#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

// CUTLASS headers (will be added via include paths)
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/arch/arch.h"

// FMHA kernel from CUTLASS examples
#include "kernel_forward.h"

/*
 * Simplified AttentionKernel instantiation for our use case
 * - FP16 (cutlass::half_t)
 * - sm_80 (Ampere, compatible with sm_89 Ada L4)
 * - 64 queries per block, 64 keys per block
 * - Head dim = 64
 * - No dropout, no bias
 */
using FMHAKernel = AttentionKernel<
    cutlass::half_t,              // FP16
    cutlass::arch::Sm80,          // Ampere (compatible with Ada)
    true,                         // isAligned
    64,                           // kQueriesPerBlock
    64,                           // kKeysPerBlock
    64,                           // kMaxK (head_dim = 64)
    false,                        // kSupportsDropout
    false                         // kSupportsBias
>;

/*
 * Launch CUTLASS FMHA kernel
 * 
 * Input shapes: Q, K, V are [B, H, S, D] in row-major
 * Output shape: O is [B, H, S, D]
 *
 * Note: attention_kernel_batched_impl is defined in kernel_forward.h
 */
void launch_cutlass_fmha(
    const cutlass::half_t* Q,      // [B, H, S, D]
    const cutlass::half_t* K,      // [B, H, S, D]
    const cutlass::half_t* V,      // [B, H, S, D]
    cutlass::half_t* O,            // [B, H, S, D]
    int B,                         // batch size
    int H,                         // num heads
    int S,                         // sequence length
    int D,                         // head dim
    cudaStream_t stream
) {
    // Validate inputs
    if (D != 64) {
        throw std::runtime_error("CUTLASS wrapper requires D=64");
    }
    
    // Setup kernel parameters
    typename FMHAKernel::Params params;
    
    // Input pointers (cast away const for CUTLASS API)
    params.query_ptr = const_cast<cutlass::half_t*>(Q);
    params.key_ptr = const_cast<cutlass::half_t*>(K);
    params.value_ptr = const_cast<cutlass::half_t*>(V);
    
    // Output pointer
    params.output_ptr = O;
    params.output_accum_ptr = nullptr;  // Not needed for our case
    params.logsumexp_ptr = nullptr;     // Not needed for forward
    
    // Softmax scale: 1 / sqrt(D)
    params.scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // Dimensions
    params.num_heads = H;
    params.num_batches = B;
    params.head_dim = D;
    params.head_dim_value = D;
    params.num_queries = S;
    params.num_keys = S;
    
    // Strides for BMHK [B, S, H, D] layout (after permute in bindings)
    // BMHK tensor: [B=1, S=512, H=8, D=64]
    //   To access element [b, s, h, d]:
    //     offset = b * (S*H*D) + s * (H*D) + h * D + d
    //
    // CUTLASS Params expects:
    //   q_strideH: stride to go to next head
    //   q_strideM: stride to go to next sequence position
    //   q_strideB: stride to go to next batch
    //
    // For BMHK [B, S, H, D]:
    params.q_strideH = D;              // next head: skip D elements
    params.k_strideH = D;
    params.v_strideH = D;
    params.q_strideM = H * D;          // next seq pos: skip H*D elements
    params.k_strideM = H * D;
    params.v_strideM = H * D;
    params.q_strideB = S * H * D;      // next batch: skip S*H*D elements
    params.k_strideB = S * H * D;
    params.v_strideB = S * H * D;
    params.o_strideM = H * D;          // output: same as query
    
    // Custom mask (none for now)
    params.custom_mask_type = FMHAKernel::NoCustomMask;
    
    // Kernel function
    constexpr auto kernel_fn = attention_kernel_batched_impl<FMHAKernel>;
    
    // Shared memory
    int smem_bytes = sizeof(typename FMHAKernel::SharedStorage);
    if (smem_bytes > 0xc000) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    
    // Check if supported
    if (!FMHAKernel::check_supported(params)) {
        throw std::runtime_error("CUTLASS FMHA: Kernel does not support these inputs");
    }
    
    // Launch kernel using Params methods
    kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(params);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FMHA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

