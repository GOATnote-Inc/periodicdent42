# FlashAttention-2 Split-K Design Document
## Session N+7 - Priority 1 Implementation

**Goal**: Implement parallel K/V tiles for 10× speedup improvement

---

## Architecture Overview

### Current (FA-1): Sequential K/V Loop
```
Grid: (H, B, Q_tiles) = 3D
Each block processes Q_tile:
  for each KV_tile (SERIAL):
    Load Q, K, V
    Compute attention
    Update running stats
  Write final output
```

**Problem**: No parallelism across KV dimension

---

### Target (FA-2): Parallel K/V with Split-K
```
Pass 1: Compute Partial Attention
Grid: (H, B, Q_tiles, KV_tiles) = 4D
Each block processes ONE (Q_tile, KV_tile) pair:
  Load Q[q_tile], K[kv_tile], V[kv_tile]
  Compute scores: S = Q @ K^T
  Compute softmax: P = softmax(S) [local to this tile]
  Compute partial output: O_partial = P @ V
  Store: O_partial[h,b,q,kv], max[h,b,q,kv], sum[h,b,q,kv]

Pass 2: Reduce Partial Results
Grid: (H, B, Q_tiles) = 3D
Each block reduces over all KV_tiles for one Q_tile:
  Load all O_partial[h,b,q,:], max[h,b,q,:], sum[h,b,q,:]
  Compute global_max = max(max[:])
  Reweight and sum: O = sum(O_partial[:] * exp(max[:] - global_max) * sum[:]) / global_sum
  Write final O[h,b,q]
```

---

## Memory Layout

### Partial Results Buffers
```cpp
// Dimensions: [B, H, Q_tiles, KV_tiles, TILE_SIZE_M, head_dim]
T* partial_O;  // Partial outputs

// Dimensions: [B, H, Q_tiles, KV_tiles, TILE_SIZE_M]
float* partial_max;  // Local max for each (q_tile, kv_tile) pair
float* partial_sum;  // Local sum for each (q_tile, kv_tile) pair
```

**Memory Cost** (for S=128, D=64, B=1, H=1):
- Q_tiles = 2, KV_tiles = 2
- partial_O: 1 × 1 × 2 × 2 × 64 × 64 × 2 bytes = 32,768 bytes = 32 KB
- partial_max: 1 × 1 × 2 × 2 × 64 × 4 bytes = 2,048 bytes = 2 KB
- partial_sum: 1 × 1 × 2 × 2 × 64 × 4 bytes = 2,048 bytes = 2 KB
- **Total: 36 KB per (B=1, H=1, S=128)** ✅ Acceptable

---

## Grid Configuration

### Pass 1: Compute Partials
```cpp
const int num_query_tiles = (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M;
const int num_kv_tiles = (seq_len + TILE_SIZE_N - 1) / TILE_SIZE_N;

// Option A: 4D grid (if CUDA supports, needs compute capability 7.0+)
dim3 grid(num_heads, batch_size, num_query_tiles, num_kv_tiles);

// Option B: Flatten last two dimensions (safer, works everywhere)
const int total_tiles = num_query_tiles * num_kv_tiles;
dim3 grid(num_heads, batch_size, total_tiles);
// In kernel: query_tile_idx = blockIdx.z / num_kv_tiles
//            kv_tile_idx = blockIdx.z % num_kv_tiles
```

**Decision**: Use **Option B** (flattened 3D) for compatibility

### Pass 2: Reduce
```cpp
dim3 grid(num_heads, batch_size, num_query_tiles);
dim3 block(THREADS_PER_BLOCK);
```

---

## Kernel Signatures

### Pass 1: Compute Partial Attention
```cpp
template<typename T>
__global__ void flash_attention_forward_split_k_partial(
    const T* Q,
    const T* K,
    const T* V,
    T* partial_O,              // Output: [B,H,Q_tiles,KV_tiles,TILE_SIZE_M,D]
    float* partial_max,        // Output: [B,H,Q_tiles,KV_tiles,TILE_SIZE_M]
    float* partial_sum,        // Output: [B,H,Q_tiles,KV_tiles,TILE_SIZE_M]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int num_query_tiles,
    const int num_kv_tiles,
    const float softmax_scale,
    const bool causal
);
```

### Pass 2: Reduce Partial Results
```cpp
template<typename T>
__global__ void flash_attention_forward_split_k_reduce(
    const T* partial_O,        // Input: [B,H,Q_tiles,KV_tiles,TILE_SIZE_M,D]
    const float* partial_max,  // Input: [B,H,Q_tiles,KV_tiles,TILE_SIZE_M]
    const float* partial_sum,  // Input: [B,H,Q_tiles,KV_tiles,TILE_SIZE_M]
    T* O,                      // Output: [B,H,S,D]
    float* softmax_lse,        // Output: [B,H,S]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int num_query_tiles,
    const int num_kv_tiles
);
```

---

## Implementation Steps

### Step 1: Create Partial Kernel (Pass 1)
1. Copy current kernel as template
2. Remove `for (kv_tile)` loop
3. Extract single `kv_tile_idx` from `blockIdx.z`
4. Compute local softmax (not online)
5. Store partial results to global memory

### Step 2: Create Reduction Kernel (Pass 2)
1. Load all partial results for this query tile
2. Find global max across all kv_tiles
3. Reweight and sum partial outputs
4. Write final output

### Step 3: Update Host Function
1. Allocate partial buffers
2. Launch Pass 1 (4× more blocks for S=128)
3. Launch Pass 2 (original block count)
4. Free partial buffers

### Step 4: Validate Correctness
1. Run all 7 test configs
2. Verify max_diff < 0.1
3. Compare to Session N+5 results

### Step 5: Measure Performance
1. Run benchmark suite
2. Compare to Session N+6 baseline
3. Calculate speedup improvement

---

## Expected Performance

### Session N+6 Baseline
- S=128: 0.543 ms (0.045× vs PyTorch)
- S=512: 2.133 ms (0.015× vs PyTorch)

### Session N+7 Target (After Priority 1)
- S=128: 0.054 ms (0.45× vs PyTorch) → **10× improvement** ✅
- S=512: 0.213 ms (0.15× vs PyTorch) → **10× improvement** ✅

**Conservative Estimate**: 5-10× speedup from parallelizing K/V tiles

---

## Risk Mitigation

### Risk 1: Increased Memory Usage
**Mitigation**: Partial buffers scale with num_tiles², but still manageable
- For S=512: 8 × 8 = 64 tile pairs → 64 × 36 KB = 2.3 MB per (B,H) ✅

### Risk 2: Reduction Overhead
**Mitigation**: Reduction is parallel (all queries reduced simultaneously)
- Expected overhead: 5-10% of total time

### Risk 3: Numerical Accuracy
**Mitigation**: Use same online softmax math, just applied in reduction
- Reweight formula: `O = sum(O_partial[i] * exp(max[i] - global_max) * sum[i]) / global_sum`
- This is mathematically equivalent to online softmax

---

## Testing Strategy

### Phase 1: Partial Kernel Only
1. Launch Pass 1 only
2. Manually verify partial_O, partial_max, partial_sum on CPU
3. Check: sum(partial_O[all kv_tiles]) ≈ expected output

### Phase 2: Add Reduction
4. Implement Pass 2
5. Run correctness tests (all 7 configs)
6. Verify: max_diff < 0.1 (same as Session N+5)

### Phase 3: Performance
7. Run benchmark suite
8. Compare to Session N+6 baseline
9. Verify: 5-10× speedup achieved

---

## Success Criteria

✅ **Correctness**: All 7 test configs pass (max_diff < 0.1)  
✅ **Performance**: 5-10× speedup vs Session N+6 baseline  
✅ **Scalability**: Performance improves with sequence length  
✅ **Memory**: Partial buffers fit in GPU memory for S≤512  

---

**Status**: Design complete, ready for implementation  
**Next**: Implement Pass 1 (Partial Kernel) in Sub-Session N+7A  
**Time**: 6:58 PM UTC, 3 minutes elapsed

