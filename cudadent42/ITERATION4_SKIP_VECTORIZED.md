# Iteration 4: Vectorized Memory Access - Already Implemented

**Date**: October 13, 2025  
**Duration**: 10 minutes  
**Status**: Skipped (already present)

## Finding

Code inspection reveals the kernel already implements vectorized memory access:

**Q loading** (line 217-224):
```cuda
const float4* Q_vec = reinterpret_cast<const float4*>(Q_base + query_idx * head_dim);
float4* smem_Q_vec = reinterpret_cast<float4*>(&smem_Q[query_idx_in_tile][0]);
for (int d = 0; d < head_dim / 8; ++d) {
    smem_Q_vec[d] = Q_vec[d];  // 128-bit loads
}
```

**K, V loading** (line 250-263):
```cuda
const float4* K_vec = reinterpret_cast<const float4*>(K_base + kv_idx * head_dim);
const float4* V_vec = reinterpret_cast<const float4*>(V_base + kv_idx * head_dim);
for (int d = 0; d < head_dim / 8; ++d) {
    smem_K_vec[d] = K_vec[d];
    smem_V_vec[d] = V_vec[d];
}
```

## Analysis

With head_dim=64 and FP16 data:
- 64 elements × 2 bytes = 128 bytes per row
- float4 = 16 bytes
- Loads per row: 128 / 16 = 8 float4 operations
- Bandwidth efficiency: Good (128-bit aligned loads)

## Conclusion

Vectorized memory access does NOT explain the 6.8-73× performance gap, as this optimization is already implemented.

## Revised Bottleneck Hypothesis

Given vectorized loads are present but GFLOP/s remains at 0.02-1.2% of peak:

**Primary suspects**:
1. **Shared memory bank conflicts** (64-column arrays on 32-bank memory)
2. **Insufficient parallelism** (2-512 CTAs vs 58 SMs)
3. **Register pressure** (limiting occupancy)
4. **Algorithmic inefficiency** (sequential K/V tile processing)

## Next: Iteration 5

Implement shared memory padding to eliminate bank conflicts.

**Target**: 1.3-1.5× speedup  
**Implementation**: Add padding to smem arrays  
**Time**: 15-20 minutes

