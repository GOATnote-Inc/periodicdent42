// I7: Tensor Core acceleration with WMMA
// Expert approach: proven tile sizes, standard patterns, focus on working first

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../include/dhp_ct_enhanced.cuh"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64  // 4 WMMA tiles
#define BN 64  // 4 WMMA tiles
#define BK 64  // 4 WMMA tiles

#define WARP_SIZE 32

__global__ void __launch_bounds__(128)  // 4 warps
dhp_i7_wmma(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
) {
    const int batch_head = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (batch_head >= batch_size) return;
    
    const int row_start = tile_m * BM;
    
    // Shared memory for tiles
    __shared__ __half Qs[BM][BK];
    __shared__ __half Ks[BN][BK];
    __shared__ __half Vs[BN][BK];
    __shared__ __half S_tile[BM][BN];
    
    // Each warp handles WMMA_M=16 rows
    const int warp_row = warp_id * WMMA_M;
    const int global_row_base = row_start + warp_row;
    
    // Per-warp online softmax state (16 rows per warp)
    float m[WMMA_M];
    float l[WMMA_M];
    float acc[WMMA_M][BK];
    
    #pragma unroll
    for (int i = 0; i < WMMA_M; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < BK; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Load Q tile (cooperative across all threads)
    for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
        int row = i / BK;
        int col = i % BK;
        int global_row = row_start + row;
        
        uint32_t valid = ct_lt_u32(global_row, S_actual);
        int q_idx = batch_head * S_max * BK + global_row * BK + col;
        __half val = Q[q_idx];
        Qs[row][col] = ct_select_half(__float2half(0.0f), val, valid);
    }
    __syncthreads();
    
    // Loop over K/V tiles
    for (int tile_n = 0; tile_n < (S_max + BN - 1) / BN; ++tile_n) {
        int col_start = tile_n * BN;
        
        // Load K tile
        for (int i = threadIdx.x; i < BN * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int global_col = col_start + row;
            
            uint32_t valid = ct_lt_u32(global_col, S_actual);
            int k_idx = batch_head * S_max * BK + global_col * BK + col;
            __half val = K[k_idx];
            Ks[row][col] = ct_select_half(__float2half(0.0f), val, valid);
        }
        __syncthreads();
        
        // Compute S = Q @ K^T using WMMA
        // Each warp computes 16x64 chunk of S
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
        
        for (int n_tile = 0; n_tile < BN / WMMA_N; ++n_tile) {
            wmma::fill_fragment(c_frag, __float2half(0.0f));
            
            // Accumulate over K dimension
            for (int k_tile = 0; k_tile < BK / WMMA_K; ++k_tile) {
                wmma::load_matrix_sync(a_frag, &Qs[warp_row][k_tile * WMMA_K], BK);
                wmma::load_matrix_sync(b_frag, &Ks[n_tile * WMMA_N][k_tile * WMMA_K], BK);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            
            // Store to shared memory
            wmma::store_matrix_sync(&S_tile[warp_row][n_tile * WMMA_N], c_frag, BN, wmma::mem_row_major);
        }
        __syncthreads();
        
        // Scale and apply causal mask (each thread handles one element)
        if (warp_id == 0) {  // Use one warp to avoid conflicts
            for (int i = lane_id; i < BM * BN; i += WARP_SIZE) {
                int row = i / BN;
                int col = i % BN;
                int global_row = row_start + row;
                int global_col = col_start + col;
                
                float s = __half2float(S_tile[row][col]) * 0.125f;
                
                uint32_t causal = ct_le_u32(global_col, global_row);
                uint32_t row_valid = ct_lt_u32(global_row, S_actual);
                uint32_t col_valid = ct_lt_u32(global_col, S_actual);
                uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal);
                
                s = ct_select_f32(-INFINITY, s, valid);
                S_tile[row][col] = __float2half(s);
            }
        }
        __syncthreads();
        
        // Online softmax update (each warp handles its 16 rows)
        for (int i = 0; i < WMMA_M; ++i) {
            int row = warp_row + i;
            if (row >= BM) break;
            
            // Find max in this row (warp-collaborative)
            float row_max = -INFINITY;
            for (int col = lane_id; col < BN; col += WARP_SIZE) {
                float s = __half2float(S_tile[row][col]);
                uint32_t gt = ct_gt_f32(s, row_max);
                row_max = ct_select_f32(row_max, s, gt);
            }
            // Warp reduce
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(0xffffffff, row_max, offset);
                uint32_t gt = ct_gt_f32(other, row_max);
                row_max = ct_select_f32(row_max, other, gt);
            }
            
            // Update global max
            uint32_t gt = ct_gt_f32(row_max, m[i]);
            float m_new = ct_select_f32(m[i], row_max, gt);
            
            // Rescale
            float alpha = expf(m[i] - m_new);
            l[i] *= alpha;
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                acc[i][k] *= alpha;
            }
            
            // Compute exp and sum (warp-collaborative)
            float row_sum = 0.0f;
            for (int col = lane_id; col < BN; col += WARP_SIZE) {
                float s = __half2float(S_tile[row][col]);
                float p = safe_exp(s - m_new);
                S_tile[row][col] = __float2half(p);  // Store P for P@V
                row_sum += p;
            }
            // Warp reduce sum
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
            }
            l[i] += row_sum;
            m[i] = m_new;
        }
        __syncthreads();
        
        // Load V tile
        for (int i = threadIdx.x; i < BN * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int global_col = col_start + row;
            
            uint32_t valid = ct_lt_u32(global_col, S_actual);
            int v_idx = batch_head * S_max * BK + global_col * BK + col;
            __half val = V[v_idx];
            Vs[row][col] = ct_select_half(__float2half(0.0f), val, valid);
        }
        __syncthreads();
        
        // Accumulate P @ V (warp-collaborative, no WMMA for now - just naive)
        for (int i = 0; i < WMMA_M; ++i) {
            int row = warp_row + i;
            if (row >= BM) break;
            
            for (int k = lane_id; k < BK; k += WARP_SIZE) {
                float sum = 0.0f;
                #pragma unroll
                for (int col = 0; col < BN; ++col) {
                    sum += __half2float(S_tile[row][col]) * __half2float(Vs[col][k]);
                }
                acc[i][k] += sum;
            }
        }
        __syncthreads();
    }
    
    // Write output
    for (int i = 0; i < WMMA_M; ++i) {
        int global_row = global_row_base + i;
        if (global_row >= S_max) break;
        
        uint32_t valid = ct_lt_u32(global_row, S_actual);
        float l_safe = ct_select_f32(1.0f, l[i], valid);
        
        for (int k = lane_id; k < BK; k += WARP_SIZE) {
            float val = acc[i][k] / l_safe;
            val = ct_select_f32(0.0f, val, valid);
            int out_idx = batch_head * S_max * BK + global_row * BK + k;
            out[out_idx] = __float2half(val);
        }
    }
}

