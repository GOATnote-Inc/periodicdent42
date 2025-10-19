# Reusable Code Snippets for EvoEngineer SDPA

### cp.async 16B legality (Ada)
- Use 16B bursts (`int4`) when both src & dst are 16‑byte aligned.
- Map half elements: 8 half = 16B. Loop `idx += 32*8` per warp so lanes stride in 16B units.
- Tail: scalar fallback loads to SMEM.

```cuda
__device__ __forceinline__ void cp_async_16B_if_aligned(
    void* smem_ptr, const void* global_ptr, bool predicate
) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    
    // Check 16B alignment
    if (predicate && 
        (((uintptr_t)smem_ptr & 0xF) == 0) && 
        (((uintptr_t)global_ptr & 0xF) == 0)) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem_addr), "l"(global_ptr));
    } else if (predicate) {
        // Fallback: scalar load
        *(int4*)smem_ptr = __ldg((const int4*)global_ptr);
    }
}
```

### Stage ring policy
- Preload stage 0 (commit+wait+barrier).
- Each tile t:
  - Producer: enqueue (t+1) → `write_stage`; `commit_group`
  - Consumers: compute on `read_stage` immediately
  - Fence: `wait_group<0|1>` (2‑stage vs 3‑stage), then `__syncthreads()` and swap

```cuda
// 2-stage example
int write_stage = 0, read_stage = 0;

// Preload stage 0
producer_load_tile(0, write_stage);
cp_async_commit_group();
cp_async_wait_group<0>();
__syncthreads();

for (int t = 0; t < num_tiles; ++t) {
    // Producer: async load next tile (t+1)
    if (is_producer && t + 1 < num_tiles) {
        write_stage = (write_stage + 1) % STAGES;
        producer_load_tile(t + 1, write_stage);
        cp_async_commit_group();
    }
    
    // Consumers: compute on current tile
    if (is_compute_warp) {
        compute_on_tile(read_stage);
    }
    
    // Wait for next stage and sync
    cp_async_wait_group<STAGES - 1>();
    __syncthreads();
    
    read_stage = (read_stage + 1) % STAGES;
}
```

### Interleaved column‑major (for v7b)
- Use a **column‑interleaved<8>** layout (CUTLASS‑style) so that WMMA col‑major loads remain contiguous:
  - Physical column = `c' = (c & ~7) + ((c ^ (r>>3)) & 7)`
  - Keeps per‑column contiguity; avoids 32‑way bank conflicts without breaking `ld=Dpad`.
- Apply same mapping at **store time only**; **load with `wmma::matrix_b,col_major`** using the *physical* base pointer.

```cuda
// Interleaved column-major store for K^T
__device__ __forceinline__ int interleaved_col_idx(int row, int col, int ld) {
    // Break bank conflicts while keeping columns contiguous for WMMA
    int col_swizzled = (col & ~7) + ((col ^ (row >> 3)) & 7);
    return col_swizzled * ld + row;
}

// Store K^T with interleaving
for (int n = lane; n < kv_len; n += 32) {
    for (int c = 0; c < HEAD_DIM; c += 8) {
        int idx = interleaved_col_idx(c, n, HEAD_DIM_PAD);
        sK[idx] = __ldg(&K_bh[(kv_start + n) * d + c]);
    }
}

// Load with WMMA (still col-major, ld=HEAD_DIM_PAD)
const half* kt_ptr = &sK[n0 * HEAD_DIM_PAD + k0];
wmma::load_matrix_sync(kt_frag, kt_ptr, HEAD_DIM_PAD);
```

### Epilogue micro‑fusion
- Normalize & cast inside the last tile store: `o *= (1/l[r]); __float2half_rn(o);`
- Optional: `st.global.cs` for streaming stores (PTX)

```cuda
// Fused normalize + cast epilogue
for (int r = warp_id; r < num_q_rows; r += NUM_WARPS) {
    float l_final = l_smem[r];
    float inv_l = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
    
    half* out_row = Obh + (q_start + r) * d;
    
    for (int c = lane; c < HEAD_DIM; c += 32) {
        float o_val = O_accum[r * HEAD_DIM_PAD + c] * inv_l;
        out_row[c] = __float2half(o_val);
    }
}
```

### Warp Specialization (Producer/Consumer)
```cuda
const int NUM_WARPS = 5;  // 4 compute + 1 producer
const int warp_id = threadIdx.x >> 5;
const int lane = threadIdx.x & 31;

const bool is_producer = (warp_id == 4);
const bool is_compute = (warp_id < 4);

if (is_producer) {
    // Producer: async copy tiles
    for (int t = 0; t < num_tiles; ++t) {
        load_kv_tile_async(t);
        cp_async_commit_group();
    }
} else if (is_compute) {
    // Compute warps: WMMA operations
    const int warp_m0 = warp_id * 16;  // Each warp owns 16 rows
    
    for (int t = 0; t < num_tiles; ++t) {
        cp_async_wait_group<STAGES - 1>();
        __syncthreads();
        
        // WMMA Q@K^T
        wmma_qk_transpose(warp_m0);
        
        // Streaming softmax
        streaming_softmax_16_rows(warp_m0);
        
        // WMMA P@V
        wmma_pv(warp_m0);
    }
}
```

### WMMA Store → Softmax → Rebuild Pattern
```cuda
// Per-warp scratch in SMEM
float* sS_frag = sS_all + warp_tile_id * (16 * 16);  // 16x16 scores
half*  sP_frag = sP_all + warp_tile_id * (16 * 16);  // 16x16 probs

// Store Q@K^T fragment to scratch
wmma::fragment<wmma::accumulator, 16, 16, 16, float> qk_frag;
// ... compute qk_frag with wmma::mma_sync ...
wmma::store_matrix_sync(sS_frag, qk_frag, 16, wmma::mem_row_major);

// Row-wise streaming softmax (standard array indexing)
for (int r = 0; r < 16; ++r) {
    int r_global = warp_m0 + r;
    
    // Find max
    float row_max = -FLT_MAX;
    for (int c = 0; c < 16; ++c) {
        float score = sS_frag[r * 16 + c];
        if (causal && (kv_start + n0 + c > q_start + r_global)) {
            score = -FLT_MAX;
        }
        sS_frag[r * 16 + c] = score;
        row_max = fmaxf(row_max, score);
    }
    
    // Update m, l
    float m_old = m_smem[r_global];
    float m_new = fmaxf(m_old, row_max);
    
    float tile_sum = 0.0f;
    for (int c = 0; c < 16; ++c) {
        float p = __expf(sS_frag[r * 16 + c] - m_new);
        sP_frag[r * 16 + c] = __float2half(p);
        tile_sum += p;
    }
    
    float rescale = __expf(m_old - m_new);
    float l_new = l_smem[r_global] * rescale + tile_sum;
    
    // Rescale O_accum
    for (int c = lane; c < HEAD_DIM; c += 32) {
        O_accum[r_global * HEAD_DIM_PAD + c] *= rescale;
    }
    
    m_smem[r_global] = m_new;
    l_smem[r_global] = l_new;
}
__syncwarp();

// Rebuild P fragment for WMMA P@V
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
wmma::load_matrix_sync(p_frag, sP_frag, 16);

// Now use p_frag with WMMA to compute P@V
// ...
```

