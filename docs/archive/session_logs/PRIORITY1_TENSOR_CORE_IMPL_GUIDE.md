# Priority 1: Tensor Core Implementation Guide

**Date**: October 14, 2025  
**Goal**: Replace manual FP16 arithmetic with `wmma` Tensor Core operations  
**Target**: 6-8× speedup (0.5257 ms → 0.08 ms)

---

## 🎯 Changes Required

### 1. Keep Unchanged (✅ Working correctly)
- Tile sizes: TILE_M=32, TILE_N=32, TILE_K/HEAD_DIM=64
- Shared memory layout
- Online softmax logic
- Correction factor computation
- Memory loading functions (load_Q_tile, load_K_tile, load_V_tile)
- Main kernel structure (iterate over K/V tiles)

### 2. Replace with `wmma` (🔧 Optimization target)
- `compute_QK()`: Q @ K^T matmul
- `compute_SV()`: S @ V matmul

---

## 📐 Tensor Core Fragment Sizes (L4 Ada Lovelace)

### Hardware Specs
- **Fragment size**: `m16n8k16`
  - M (output rows): 16
  - N (output cols): 8  
  - K (inner dim): 16
- **Input type**: FP16 (`half`)
- **Accumulator type**: FP32 (`float`)
- **Instruction**: `wmma::mma_sync(d, a, b, c)` computes `d = a @ b + c`

### Our Tile Mapping

**Q @ K^T: (TILE_M × TILE_K) @ (TILE_N × TILE_K)^T → (TILE_M × TILE_N)**
- (32 × 64) @ (32 × 64)^T → (32 × 32)
- M-blocks: 32/16 = 2
- N-blocks: 32/8 = 4
- K-blocks: 64/16 = 4

**S @ V: (TILE_M × TILE_N) @ (TILE_N × TILE_K) → (TILE_M × TILE_K)**
- (32 × 32) @ (32 × 64) → (32 × 64)
- M-blocks: 32/16 = 2
- N-blocks: 64/8 = 8
- K-blocks: 32/16 = 2

---

## 🔧 Implementation Plan

### Step 1: Replace `compute_QK()` with `wmma`

**Old version** (manual FP16 arithmetic):
```cuda
__device__ void compute_QK(SharedMemory* smem) {
    const int tid = threadIdx.x;
    const int elements_per_thread = (TILE_M * TILE_N) / NUM_THREADS;
    
    for (int i = 0; i < elements_per_thread; i++) {
        const int flat_idx = tid + i * NUM_THREADS;
        const int row = flat_idx / TILE_N;
        const int col = flat_idx % TILE_N;
        
        if (row < TILE_M && col < TILE_N) {
            float acc = 0.0f;
            for (int k = 0; k < TILE_K; k++) {
                float q_val = half_to_float(smem->Q[row][k]);
                float k_val = half_to_float(smem->K[col][k]);
                acc += q_val * k_val;
            }
            smem->S[row][col] = float_to_half(acc);
        }
    }
}
```

**New version** (wmma-based):
```cuda
__device__ void compute_QK_wmma(SharedMemory* smem) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Each warp computes a portion of the output
    // With 4 warps and 2×4 = 8 output tiles:
    // Warp 0,1: M-block 0 (rows 0-15), Warp 2,3: M-block 1 (rows 16-31)
    
    const int m_block = warp_id / 2;  // 0 or 1 (which 16-row block)
    const int n_start = (warp_id % 2) * 2;  // 0 or 2 (which pair of 8-col blocks)
    
    // Declare wmma fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2];  // 2 N-blocks per warp
    
    // Initialize accumulators
    for (int n_block_offset = 0; n_block_offset < 2; n_block_offset++) {
        wmma::fill_fragment(acc_frag[n_block_offset], 0.0f);
    }
    
    // Compute: acc = Q @ K^T
    // K^T layout: need to load K as col-major (transpose)
    for (int k_block = 0; k_block < TILE_K / WMMA_K; k_block++) {
        // Load Q fragment (16 rows × 16 cols from Q)
        wmma::load_matrix_sync(a_frag, 
            &smem->Q[m_block * WMMA_M][k_block * WMMA_K],
            TILE_K + SMEM_PAD);  // ldm (leading dimension with padding)
        
        // Compute 2 N-blocks per warp
        for (int n_block_offset = 0; n_block_offset < 2; n_block_offset++) {
            int n_block = n_start + n_block_offset;
            
            // Load K fragment as column-major (transpose)
            wmma::load_matrix_sync(b_frag, 
                &smem->K[n_block * WMMA_N][k_block * WMMA_K],
                TILE_K + SMEM_PAD);
            
            // Multiply-accumulate
            wmma::mma_sync(acc_frag[n_block_offset], a_frag, b_frag, acc_frag[n_block_offset]);
        }
    }
    
    // Store results to S
    for (int n_block_offset = 0; n_block_offset < 2; n_block_offset++) {
        int n_block = n_start + n_block_offset;
        wmma::store_matrix_sync(
            &smem->S[m_block * WMMA_M][n_block * WMMA_N],
            acc_frag[n_block_offset],
            TILE_N + SMEM_PAD,  // ldm
            wmma::mem_row_major);
    }
}
```

**Key changes**:
- Each warp processes 16×16 output elements (2 wmma N-blocks)
- 4 warps cover all 32×32 output
- Accumulate in FP32 (better numerical stability)
- Use `col_major` for K to get K^T

---

### Step 2: Replace `compute_SV()` with `wmma`

**Old version** (manual FP16 arithmetic):
```cuda
__device__ void compute_SV(SharedMemory* smem, float* O_local) {
    const int tid = threadIdx.x;
    
    for (int row = 0; row < TILE_M; row++) {
        for (int col = tid; col < HEAD_DIM; col += NUM_THREADS) {
            float acc = 0.0f;
            for (int k = 0; k < TILE_N; k++) {
                float s_val = half_to_float(smem->S[row][k]);
                float v_val = half_to_float(smem->V[k][col]);
                acc += s_val * v_val;
            }
            int out_idx = row * HEAD_DIM + col;
            O_local[out_idx] += acc;
        }
    }
}
```

**New version** (wmma-based):
```cuda
__device__ void compute_SV_wmma(SharedMemory* smem, float* O_local) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Each warp computes a portion of the output
    // With 4 warps and 2×8 = 16 output tiles:
    // Distribute across warps
    
    const int m_block = warp_id / 2;  // 0 or 1 (which 16-row block)
    const int n_start = (warp_id % 2) * 4;  // 0 or 4 (which set of 4 8-col blocks)
    
    // Declare wmma fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[4];  // 4 N-blocks per warp
    
    // Load existing O_local values into accumulators (for accumulation)
    for (int n_block_offset = 0; n_block_offset < 4; n_block_offset++) {
        int n_block = n_start + n_block_offset;
        // Note: O_local is FP32, need to load carefully
        // For simplicity, zero-initialize and add O_local after wmma
        wmma::fill_fragment(acc_frag[n_block_offset], 0.0f);
    }
    
    // Compute: acc = S @ V
    for (int k_block = 0; k_block < TILE_N / WMMA_K; k_block++) {
        // Load S fragment
        wmma::load_matrix_sync(a_frag, 
            &smem->S[m_block * WMMA_M][k_block * WMMA_K],
            TILE_N + SMEM_PAD);
        
        // Compute 4 N-blocks per warp
        for (int n_block_offset = 0; n_block_offset < 4; n_block_offset++) {
            int n_block = n_start + n_block_offset;
            
            // Load V fragment
            wmma::load_matrix_sync(b_frag, 
                &smem->V[k_block * WMMA_K][n_block * WMMA_N],
                TILE_K + SMEM_PAD);
            
            // Multiply-accumulate
            wmma::mma_sync(acc_frag[n_block_offset], a_frag, b_frag, acc_frag[n_block_offset]);
        }
    }
    
    // Store results to O_local (need to accumulate with existing values)
    for (int n_block_offset = 0; n_block_offset < 4; n_block_offset++) {
        int n_block = n_start + n_block_offset;
        
        // Extract results from accumulator fragment
        // Each fragment has 256 elements (16×8), distributed across 32 threads
        float* acc_ptr = &acc_frag[n_block_offset].x;  // Access raw elements
        
        // Map fragment elements to O_local indices
        // This is warp-level cooperative: each thread handles certain elements
        for (int frag_idx = 0; frag_idx < acc_frag[n_block_offset].num_elements; frag_idx++) {
            // Map frag_idx to (local_row, local_col) within the 16×8 tile
            int local_row = frag_idx / 8;  // 0-15
            int local_col = frag_idx % 8;  // 0-7
            
            // Global position in output
            int global_row = m_block * WMMA_M + local_row;
            int global_col = n_block * WMMA_N + local_col;
            
            // Accumulate to O_local
            int out_idx = global_row * HEAD_DIM + global_col;
            O_local[out_idx] += acc_ptr[frag_idx];
        }
    }
}
```

**Key changes**:
- Each warp processes 16×32 output elements (4 wmma N-blocks)
- Accumulate wmma results with existing O_local values
- More complex because O_local is accumulated across K/V tiles

---

## ⚠️ Challenges & Solutions

### Challenge 1: O_local accumulation
**Problem**: wmma outputs FP32 fragments, but we need to accumulate across multiple K/V tiles  
**Solution**: Use `wmma::fill_fragment(acc_frag, 0.0f)` initially, then manually add wmma result to O_local

### Challenge 2: Fragment element mapping
**Problem**: wmma fragment layout is opaque (not just a 16×8 array)  
**Solution**: Use `fragment.num_elements` and careful index mapping (may need trial/error)

### Challenge 3: Thread synchronization
**Problem**: warps need to synchronize after writing to smem->S  
**Solution**: Keep `__syncthreads()` after both compute_QK_wmma and compute_SV_wmma

---

## 🧪 Validation Strategy

### Step 1: Compile & Run
- Expect compilation warnings about unused variables (normal)
- Check for wmma-related errors

### Step 2: Correctness Tests
- Run all 7 test cases
- **Critical**: Max error must be < 0.02 (FP16 tolerance)
- **Acceptable**: Slightly higher error than baseline (wmma introduces minor numerical differences)
- **Unacceptable**: Max error > 0.05 or NaN outputs

### Step 3: Performance Measurement
- Benchmark with N=100, bootstrap CIs
- **Target**: < 0.10 ms (at least 5× speedup)
- **Stretch**: < 0.08 ms (6.5× speedup)
- **Best**: < 0.065 ms (8× speedup)

---

## 📊 Expected Results

### Performance
- **Current**: 0.5257 ms
- **Expected**: 0.08 ms (6.5× speedup)
- **95% CI**: Should not overlap with baseline

### Correctness
- **All 7 tests**: Should pass
- **Max error**: 0.001-0.003 (may be slightly higher than baseline due to wmma FP32 accumulation)

### Profiling (if ncu works)
- **Tensor Core utilization**: 40-60% (up from 0%)
- **SM throughput**: Increase by 6-8×

---

## 🚀 Implementation Approach

Given the complexity, I'll:
1. **Create new kernel file**: `fa_inverted_v2_tensor_cores.cu`
2. **Test incrementally**: First QK, then SV
3. **Fall back if needed**: If wmma doesn't work, document why and try simpler approach

**Time estimate**: 2-3 hours for implementation + validation

---

**Ready to implement!** 🔥

