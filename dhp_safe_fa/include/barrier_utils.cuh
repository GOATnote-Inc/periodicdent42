#pragma once
// Hopper Barrier Utilities
// Based on EXPERT_CORRECTIONS.md §1.3

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Hopper mbarrier (Raw PTX for maximum performance)
// ============================================================================

// Initialize mbarrier
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_addr, uint32_t expected_count) {
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;" 
            :: "r"((unsigned)__cvta_generic_to_shared(mbar_addr)), 
               "r"(expected_count)
        );
    }
}

// Producer: Arrive at barrier
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_addr) {
    asm volatile(
        "mbarrier.arrive.shared.b64 _, [%0];"
        :: "r"((unsigned)__cvta_generic_to_shared(mbar_addr))
    );
}

// Consumer: Wait on barrier
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_addr, uint32_t phase) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  mbarrier.test_wait.shared.b64 p, [%0], %1;\n"
        "  @!p nanosleep.u32 20;\n"  // Backoff if not ready
        "}"
        :: "r"((unsigned)__cvta_generic_to_shared(mbar_addr)),
           "r"(phase)
    );
}

// ============================================================================
// Warp-Group Synchronization (Hopper WGMMA requirements)
// ============================================================================

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// ============================================================================
// Simplified Barrier API (for easier usage in I4)
// ============================================================================

struct DHPBarrier {
    uint64_t mbar[2];  // Double-buffered
    
    __device__ void init(uint32_t expected_count) {
        mbarrier_init(&mbar[0], expected_count);
        mbarrier_init(&mbar[1], expected_count);
    }
    
    __device__ void arrive(int stage) {
        mbarrier_arrive(&mbar[stage]);
    }
    
    __device__ void wait(int stage) {
        mbarrier_wait(&mbar[stage], 0);
    }
};

// ============================================================================
// Usage Notes:
// - mbarrier is Hopper-specific (sm_90a+)
// - Warp-group ops required for WGMMA
// - For fallback to Ampere, use __syncthreads()
// ============================================================================

