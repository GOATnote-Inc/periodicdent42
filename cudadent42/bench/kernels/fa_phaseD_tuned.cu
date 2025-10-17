// Phase D: Register Pressure Optimized Attention Kernel
// Goal: Increase occupancy from 9.28% to 20%+ via register reduction
//
// Optimizations applied:
// 1. Launch bounds to guide compiler
// 2. Reduced per-thread arrays (move to SMEM)
// 3. Bounded unrolls (not aggressive)
// 4. __restrict__ pointers
// 5. De-inlined heavy functions
// 6. Smaller tile sizes for better occupancy

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Configurable via environment variables
#ifndef TILE_M
#define TILE_M 32
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 192  // Default: 192 (vs 256)
#endif

#ifndef MIN_BLOCKS_PER_SM
#define MIN_BLOCKS_PER_SM 2
#endif

#define HEAD_DIM 64
#define NUM_WARPS (THREADS_PER_BLOCK / 32)

// Warp-level reductions (de-inlined to save registers)
__device__ float warp_reduce_max(float val) {
    #pragma unroll 4  // Bounded unroll (not full 16)
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    #pragma unroll 4  // Bounded unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Phase D Kernel: Optimized for occupancy
// Use launch bounds to control register allocation
#ifdef LAUNCH_BOUNDS_THREADS
__launch_bounds__(LAUNCH_BOUNDS_THREADS, LAUNCH_BOUNDS_MIN)
#endif
__global__ void phaseD_attention_kernel(
    const half* __restrict__ Q,      // [B, H, M, D]
    const half* __restrict__ K,      // [B, H, N, D]
    const half* __restrict__ V,      // [B, H, N, D]
    half* __restrict__ O,            // [B, H, M, D]
    const int B,
    const int H,
    const int M,
    const int N,
    const int D,
    const float scale
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int row_block = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int row_start = row_block * TILE_M;
    const int rows_this_block = min(TILE_M, M - row_start);
    
    // OPTIMIZATION 1: Move large per-thread arrays to shared memory
    // OLD: float O_row[HEAD_DIM];  // 64 floats = 256 bytes per thread!
    // NEW: Use shared memory buffer
    extern __shared__ char smem_buffer[];
    
    // Shared memory layout (carefully sized to fit in 48KB limit)
    half* Q_smem = reinterpret_cast<half*>(smem_buffer);
    half* K_smem = Q_smem + TILE_M * HEAD_DIM;
    half* V_smem = K_smem + N * HEAD_DIM;
    float* S_smem = reinterpret_cast<float*>(V_smem + N * HEAD_DIM);
    float* O_smem = S_smem + TILE_M * N;
    float* m_smem = O_smem + TILE_M * HEAD_DIM;
    float* l_smem = m_smem + TILE_M;
    
    // Load Q tile to SMEM (coalesced access)
    const half* Q_base = Q + b * H * M * D + h * M * D + row_start * D;
    for (int row = warp_id; row < rows_this_block; row += NUM_WARPS) {
        const half* Q_ptr = Q_base + row * D;
        half* Q_smem_ptr = Q_smem + row * HEAD_DIM;
        
        // OPTIMIZATION 2: Vectorized loads (4 elements at a time)
        #pragma unroll 2  // Bounded unroll
        for (int d = lane_id * 4; d < D; d += 32 * 4) {
            if (d + 3 < D) {
                *reinterpret_cast<float2*>(&Q_smem_ptr[d]) = 
                    *reinterpret_cast<const float2*>(&Q_ptr[d]);
            }
        }
    }
    
    // Load K to SMEM
    const half* K_base = K + b * H * N * D + h * N * D;
    for (int n = tid; n < N; n += THREADS_PER_BLOCK) {
        const half* K_ptr = K_base + n * D;
        half* K_smem_ptr = K_smem + n * HEAD_DIM;
        
        #pragma unroll 2
        for (int d = 0; d < D; d += 4) {
            if (d + 3 < D) {
                *reinterpret_cast<float2*>(&K_smem_ptr[d]) = 
                    *reinterpret_cast<const float2*>(&K_ptr[d]);
            }
        }
    }
    
    // Load V to SMEM
    const half* V_base = V + b * H * N * D + h * N * D;
    for (int n = tid; n < N; n += THREADS_PER_BLOCK) {
        const half* V_ptr = V_base + n * D;
        half* V_smem_ptr = V_smem + n * HEAD_DIM;
        
        #pragma unroll 2
        for (int d = 0; d < D; d += 4) {
            if (d + 3 < D) {
                *reinterpret_cast<float2*>(&V_smem_ptr[d]) = 
                    *reinterpret_cast<const float2*>(&V_ptr[d]);
            }
        }
    }
    __syncthreads();
    
    // OPTIMIZATION 3: Minimal live ranges for accumulators
    // Process each row independently to reduce register pressure
    for (int row = warp_id; row < rows_this_block; row += NUM_WARPS) {
        // Q@K^T (one row at a time)
        float* S_row = S_smem + row * N;
        const half* Q_row = Q_smem + row * HEAD_DIM;
        
        for (int n = lane_id; n < N; n += 32) {
            float score = 0.0f;
            
            // OPTIMIZATION 4: Bounded unroll (not full unroll)
            #pragma unroll 8
            for (int d = 0; d < D; d++) {
                score += __half2float(Q_row[d]) * __half2float(K_smem[n * HEAD_DIM + d]);
            }
            
            S_row[n] = score * scale;
        }
        __syncwarp();
        
        // Online softmax
        float m_row = -INFINITY;
        for (int n = lane_id; n < N; n += 32) {
            m_row = fmaxf(m_row, S_row[n]);
        }
        m_row = warp_reduce_max(m_row);
        
        // Broadcast max via shared memory
        if (lane_id == 0) m_smem[row] = m_row;
        __syncwarp();
        m_row = m_smem[row];
        
        // Exp and sum
        float l_row = 0.0f;
        for (int n = lane_id; n < N; n += 32) {
            float p = expf(S_row[n] - m_row);
            S_row[n] = p;
            l_row += p;
        }
        l_row = warp_reduce_sum(l_row);
        
        if (lane_id == 0) l_smem[row] = l_row;
        __syncwarp();
        l_row = l_smem[row];
        
        // P@V (accumulate directly to shared memory to reduce registers)
        float* O_row = O_smem + row * HEAD_DIM;
        
        // Initialize O_row to zero
        for (int d = lane_id; d < D; d += 32) {
            O_row[d] = 0.0f;
        }
        __syncwarp();
        
        // Accumulate P@V
        for (int n = 0; n < N; n++) {
            float p_normalized = S_row[n] / l_row;
            const half* V_row = V_smem + n * HEAD_DIM;
            
            #pragma unroll 4  // Bounded unroll
            for (int d = lane_id; d < D; d += 32) {
                O_row[d] += p_normalized * __half2float(V_row[d]);
            }
        }
        __syncwarp();
        
        // Write output
        int global_row = row_start + row;
        half* O_ptr = O + b * H * M * D + h * M * D + global_row * D;
        
        #pragma unroll 2  // Bounded unroll
        for (int d = lane_id * 4; d < D; d += 32 * 4) {
            if (d + 3 < D) {
                float4 vals;
                vals.x = O_row[d];
                vals.y = O_row[d+1];
                vals.z = O_row[d+2];
                vals.w = O_row[d+3];
                
                half2 h0 = __float22half2_rn(make_float2(vals.x, vals.y));
                half2 h1 = __float22half2_rn(make_float2(vals.z, vals.w));
                
                *reinterpret_cast<half2*>(&O_ptr[d]) = h0;
                *reinterpret_cast<half2*>(&O_ptr[d+2]) = h1;
            }
        }
    }
}

// Launcher
extern "C" void launch_phaseD_tuned(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int M, int N, int D,
    float scale,
    cudaStream_t stream
) {
    // Calculate shared memory size
    size_t smem_size = 0;
    smem_size += TILE_M * HEAD_DIM * sizeof(half);  // Q
    smem_size += N * HEAD_DIM * sizeof(half);        // K
    smem_size += N * HEAD_DIM * sizeof(half);        // V
    smem_size += TILE_M * N * sizeof(float);         // S
    smem_size += TILE_M * HEAD_DIM * sizeof(float);  // O
    smem_size += TILE_M * sizeof(float);             // m
    smem_size += TILE_M * sizeof(float);             // l
    
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        H,
        B
    );
    dim3 block(THREADS_PER_BLOCK);
    
    phaseD_attention_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, O, B, H, M, N, D, scale
    );
}

