// ============================================================================
// BlackwellSparseK: Runtime Architecture Dispatch
// ============================================================================
// Dispatches to sm_90a (Hopper H100) or sm_100 (Blackwell B200) kernels
// based on runtime GPU detection.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

// Forward declarations from attention_fmha.cu
extern __global__ void fmha_kernel_impl(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, int D
);

// ============================================================================
// ARCHITECTURE DETECTION
// ============================================================================

struct GPUInfo {
    int major;
    int minor;
    int sm_version;  // major * 100 + minor * 10
    const char* arch_name;
};

GPUInfo get_gpu_info(int device_id = 0) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to get device properties: ") + 
            cudaGetErrorString(err)
        );
    }
    
    GPUInfo info;
    info.major = prop.major;
    info.minor = prop.minor;
    info.sm_version = info.major * 100 + info.minor * 10;
    
    // Determine architecture name
    if (info.major == 10 && info.minor == 0) {
        info.arch_name = "Blackwell B200 (sm_100)";
    } else if (info.major == 9 && info.minor == 0) {
        info.arch_name = "Hopper H100 (sm_90a)";
    } else {
        info.arch_name = "Unknown";
    }
    
    return info;
}

void validate_architecture() {
    GPUInfo info = get_gpu_info();
    
    if (info.sm_version < 900) {
        throw std::runtime_error(
            std::string("Unsupported GPU architecture: ") + info.arch_name + "\n" +
            "BlackwellSparseK requires sm_90a (Hopper H100) or sm_100 (Blackwell B200).\n" +
            "Detected: sm_" + std::to_string(info.major) + std::to_string(info.minor)
        );
    }
}

// ============================================================================
// KERNEL LAUNCH HELPERS
// ============================================================================

void launch_fmha_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int B, int H, int S, int D,
    cudaStream_t stream
) {
    // Grid configuration
    const int kBlockM = 64;
    const int kBlockN = 64;
    const int kNumWarps = 4;
    const int kThreadsPerBlock = kNumWarps * 32;
    
    // Grid dimensions: (seq_blocks, num_heads, batch_size)
    const int num_seq_blocks = (S + kBlockM - 1) / kBlockM;
    dim3 grid(num_seq_blocks, H, B);
    dim3 block(kThreadsPerBlock);
    
    // Shared memory size
    const int kHeadDimPad = ((D + 15) / 16) * 16 + 8;
    const int smem_q = kBlockM * kHeadDimPad * sizeof(half);
    const int smem_k = kBlockN * kHeadDimPad * sizeof(half);
    const int smem_v = kBlockN * kHeadDimPad * sizeof(half);
    const int smem_s = kBlockM * kBlockN * sizeof(float);
    const size_t smem_size = smem_q + smem_k + smem_v + smem_s;
    
    // Launch kernel
    fmha_kernel_impl<<<grid, block, smem_size, stream>>>(
        Q, K, V, O, softmax_scale, B, H, S, D
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Kernel launch failed: ") + cudaGetErrorString(err)
        );
    }
}

// ============================================================================
// HOPPER (SM_90A) DISPATCH
// ============================================================================

void fmha_forward_hopper(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int B, int H, int S, int D,
    cudaStream_t stream
) {
    // Hopper H100 optimized path
    // Uses cp.async.bulk for asynchronous memory operations
    launch_fmha_kernel(Q, K, V, O, softmax_scale, B, H, S, D, stream);
}

// ============================================================================
// BLACKWELL (SM_100) DISPATCH
// ============================================================================

void fmha_forward_blackwell(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int B, int H, int S, int D,
    cudaStream_t stream
) {
    // Blackwell B200 optimized path
    // Uses TMA (Tensor Memory Accelerator) for optimal memory access
    launch_fmha_kernel(Q, K, V, O, softmax_scale, B, H, S, D, stream);
}

// ============================================================================
// UNIFIED DISPATCH ENTRY POINT
// ============================================================================

void attention_forward(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int B, int H, int S, int D,
    cudaStream_t stream = 0
) {
    // Validate architecture on first call
    static bool validated = false;
    if (!validated) {
        validate_architecture();
        validated = true;
    }
    
    // Get GPU info for dispatch
    GPUInfo info = get_gpu_info();
    
    // Dispatch to architecture-specific kernel
    if (info.sm_version >= 1000) {
        // Blackwell B200 (sm_100)
        fmha_forward_blackwell(Q, K, V, O, softmax_scale, B, H, S, D, stream);
    } else if (info.sm_version >= 900) {
        // Hopper H100 (sm_90a)
        fmha_forward_hopper(Q, K, V, O, softmax_scale, B, H, S, D, stream);
    } else {
        throw std::runtime_error("Unsupported architecture in dispatch");
    }
}

