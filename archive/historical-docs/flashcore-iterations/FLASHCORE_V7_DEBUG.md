# FlashCore v7 Debug Report

**Date**: October 22, 2025  
**Issue**: CUDA launch failure in WMMA PV implementation

---

## 🔍 **Problem**

v7 kernel crashes with "unspecified launch failure" during execution.

**Root Cause**: Complex fragment indexing in WMMA PV accumulation:
```cuda
// BUGGY CODE:
float o_tile[8];
wmma::store_matrix_sync(o_tile, o_frag, WMMA_N, wmma::mem_row_major);

for (int i = 0; i < 8; ++i) {
  int row_offset = i / 4;
  int col_offset = (i % 4);
  int m_idx = warp_m_start + row_offset + (lane_id / 4) * 2;
  int d_idx = d_wmma + col_offset * 4 + (lane_id % 4);
  atomicAdd(&sO[m_idx * (D + PAD) + d_idx], o_tile[i]);
}
```

**Issues**:
1. Fragment layout mapping is incorrect
2. Atomic add overhead
3. Index calculations don't match WMMA fragment structure

---

## 🔧 **Solution: Store to Shared Memory First**

Instead of complex fragment indexing, store WMMA output to a temp buffer in shared memory, then accumulate:

```cuda
// Allocate temp buffer for WMMA PV output
__shared__ float sPV_temp[M_TILE][D + PAD];

// WMMA PV to temp buffer
for (int d_wmma = 0; d_wmma < D; d_wmma += WMMA_N) {
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
  wmma::fill_fragment(o_frag, 0.0f);
  
  for (int n_wmma = 0; n_wmma < k_len; n_wmma += WMMA_K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
    
    wmma::load_matrix_sync(p_frag, &sP[warp_m_start * LDP + n_wmma], LDP);
    wmma::load_matrix_sync(v_frag, &sV[n_wmma * LDV + d_wmma], LDV);
    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
  }
  
  // Store to temp buffer (correct WMMA layout)
  wmma::store_matrix_sync(&sPV_temp[warp_m_start][d_wmma], o_frag, D + PAD, wmma::mem_row_major);
}
__syncthreads();

// Accumulate into sO (simple, no atomics!)
for (int m = warp_id; m < q_tile_len; m += WARPS_PER_BLOCK) {
  for (int d = lane_id; d < D; d += 32) {
    sO[m * (D + PAD) + d] += sPV_temp[m][d];
  }
}
```

---

## 📊 **Status**

- ✅ v6 (WMMA QK^T): 447 μs, **VALIDATED**
- ❌ v7 (WMMA PV): Launch failure, **NEEDS FIX**
- 🔄 v7.1 (Fixed WMMA PV): **Next iteration**

---

## 🎯 **Alternative: Skip Full WMMA PV for Now**

Since v6 already delivers 4.74× speedup (2122 → 447 μs), we could:

**Option A**: Fix WMMA PV properly (1-2 hours)
- Expected: 447 → 150-200 μs (2-3× more)

**Option B**: Focus on other optimizations first
- Vectorize global loads
- Tune tile sizes
- Add cp.async
- Come back to WMMA PV later

**Recommendation**: Fix WMMA PV now (proven technique, just needs correct implementation)

