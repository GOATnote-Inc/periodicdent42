#pragma once
// TMA Utility Wrappers
// Based on EXPERT_CORRECTIONS.md §1.1

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <cstdint>

// ============================================================================
// TMA Descriptor Helper (Host-side)
// ============================================================================

class TMATensorDescriptor {
public:
    CUtensorMap desc;
    
    void create(
        void* ptr,           // Device pointer to data
        int64_t dim0,        // First dimension (e.g., S_max)
        int64_t dim1,        // Second dimension (e.g., d)
        int64_t tile_dim0,   // Tile size in first dimension (e.g., N)
        int64_t tile_dim1    // Tile size in second dimension (e.g., d)
    ) {
        // EXPERT CORRECTION: Proper TMA descriptor creation
        CUresult result = cuTensorMapEncodeTiled(
            &desc,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,  // rank
            ptr,
            (uint64_t[]){(uint64_t)dim0, (uint64_t)dim1},
            (uint64_t[]){(uint64_t)(dim1 * sizeof(__half)), sizeof(__half)},  // strides
            (uint32_t[]){(uint32_t)tile_dim0, (uint32_t)tile_dim1},  // box (tile size)
            (uint32_t[]){1, 1},  // element strides
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,  // CRITICAL for performance
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("TMA descriptor creation failed");
        }
    }
    
    const CUtensorMap* get_device_ptr() const { 
        return &desc; 
    }
};

// ============================================================================
// TMA Load Synchronization (Device-side)
// ============================================================================

// Wait for TMA completion (Hopper-specific)
__device__ __forceinline__ void tma_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// Fence before TMA operations
__device__ __forceinline__ void tma_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ============================================================================
// Usage Notes (from EXPERT_CORRECTIONS.md):
// - TMA descriptors MUST be created on host
// - Pass descriptor pointers to kernel
// - Only thread 0 in warp should issue TMA loads
// - Use cuda::barrier or mbarrier for synchronization
// ============================================================================

