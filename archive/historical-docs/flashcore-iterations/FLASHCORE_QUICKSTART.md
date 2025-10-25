# FlashCore Quick Start Guide

**Status**: ✅ GREEN BASELINE READY FOR OPTIMIZATION  
**Location**: `cudadent42-l4-dev:~/flashcore/`

---

## 🎯 Current Status

```
✅ FP16 Baseline:      1398 μs (100% correct, 20/20 tests)
✅ PyTorch Target:     45 μs (31× speedup needed)
⚠️ WMMA Kernel:        8835 μs (correct but needs optimization)
```

---

## 🚀 Quick Commands

### Connect to L4
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c
cd ~/flashcore
```

### Test Baseline
```bash
python3 -m pytest tests/test_correctness.py -v  # All 20 tests pass
python3 benchmarks/benchmark_latency.py --shape mission --iters 100
```

### Build & Test WMMA
```bash
python3 build_wmma.py
python3 - << 'PY'
from build_wmma import build_wmma
import torch

ext = build_wmma()
B, H, S, D = 1, 8, 512, 64
Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
O = ext.forward(Q, K, V, 1.0/8.0)
print("WMMA kernel ran successfully!")
PY
```

---

## 📁 Key Files

```
flashcore/
├── kernels/flashcore_baseline.cu      ← GREEN (1398 μs)
├── kernels/flashcore_wmma.cu          ← FIX THIS (line 110-135)
├── build.py                           ← Baseline build
├── build_wmma.py                      ← WMMA build
└── tests/test_correctness.py          ← 20/20 passing
```

---

## 🔧 Next: Fix WMMA (Phase 1)

**Edit**: `kernels/flashcore_wmma.cu` lines 110-135

**Current (BROKEN)**:
```cuda
// TODO: Full WMMA implementation
// Compute Q @ K^T using WMMA (each warp computes 16×16 tile)
// For simplicity, use scalar fallback for now
```

**Replace with**:
```cuda
// Q @ K^T with WMMA
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

// Load Q tile (16×16 from 16×64 Q_tile)
for (int k_tile = 0; k_tile < 4; ++k_tile) {  // 64/16 = 4 tiles
    wmma::load_matrix_sync(q_frag, &Q_tile[0][k_tile*16], 64);
    wmma::load_matrix_sync(k_frag, &K_tile[warp_id*16][k_tile*16], 64);
    wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
}

// Store result
wmma::store_matrix_sync(&S_tile[0][warp_id*16], s_frag, 64, wmma::mem_row_major);
```

**Then rebuild**: `python3 build_wmma.py`

**Expected**: ~140-280 μs (10× faster than baseline)

---

## 📊 Phase Roadmap

| Phase | Target | Status | Techniques |
|-------|--------|--------|------------|
| Phase 0: Baseline | 1398 μs | ✅ DONE | Scalar FP16, online softmax |
| Phase 1: WMMA | <100 μs | ⏳ IN PROGRESS | Tensor Cores Q@K^T + P@V |
| Phase 2: Fusion | <60 μs | ⏳ PLANNED | FlashAttention tiling |
| Phase 3: Advanced | <45 μs | ⏳ STRETCH | Warp specialization, persistent CTAs |

**Current Focus**: Fix Phase 1 WMMA implementation

---

## 🎓 References

- **FlashCore Results**: `/Users/kiteboard/periodicdent42/FLASHCORE_SESSION1_RESULTS.md`
- **L4 Findings**: `/Users/kiteboard/periodicdent42/FLASHCORE_L4_FINDINGS.md`
- **Launch Plan**: `/Users/kiteboard/periodicdent42/FLASHCORE_LAUNCH_PLAN.md`
- **Existing WMMA Code**: `~/periodicdent42/cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`

---

**Time Estimate**: 8 hours to complete Phase 1  
**GPU Cost**: $6 at $0.75/hour  
**Next Session**: Implement proper WMMA fragments and iterate to <100 μs!

🚀 **Let's get that 15× speedup!**

