# NCU-DRIVEN OPTIMIZATION - BURN METHODOLOGY

## Iteration 0: Baseline (Single Head)
```
Q@K^T:  17.76 μs (3.72% SM)
Softmax: 9.22 μs (16.20% SM)
P@V:    17.92 μs (3.06% SM)
───────────────────────────────
TOTAL:  44.9 μs per head
```

**Problem:** Extremely low SM utilization → Problem too small for H100

---

## Iteration 1: Batched (16 Heads Together)
```
Q@K^T:  140.74 μs (7.67% SM, 5.38% DRAM) - ALL 16 heads
Softmax: 46.72 μs (50.27% SM, 49.23% DRAM) - ALL 16 heads  
P@V:     47.78 μs (18.69% SM, 36.54% DRAM) - ALL 16 heads
────────────────────────────────────────────────────────────
TOTAL:   235 μs for 16 heads
Per-head: 14.7 μs/head
```

**Result: 3× SPEEDUP! (44.9 → 14.7 μs/head)**

**Key Findings:**
- ✅ Softmax now 50% SM (memory-bound, properly saturated)
- ✅ P@V improved to 18.69% SM (better but still low)
- ⚠️ Q@K^T still only 7.67% SM (problem size limitation)

---

## NCU Ground Truth Comparison

| Metric | Iteration 0 | Iteration 1 | Improvement |
|--------|------------|------------|-------------|
| **Per-head latency** | 44.9 μs | 14.7 μs | **3.0× faster** |
| **Softmax SM %** | 16.20% | 50.27% | **3.1× better** |
| **P@V SM %** | 3.06% | 18.69% | **6.1× better** |
| **Q@K^T SM %** | 3.72% | 7.67% | 2.1× better |

---

## Analysis

### What Worked:
1. **Batching all 16 heads** saturates softmax kernel (memory-bound op)
2. **Larger problem size** improves GEMM utilization
3. **H100's parallelism** fully utilized with proper batching

### What's Still Slow:
1. **GEMMs still <20% SM** - Need larger tiles or more parallelism
2. **Problem size fundamentally small** - S=1024, D=64 is tiny for H100

### Next Steps:
- **Iteration 2:** Increase batch size (B×H together)
- **Iteration 3:** Larger tile sizes for GEMM
- **Iteration 4:** Try different CUTLASS CollectiveBuilder modes

---

## Honest Assessment

**Against PyTorch SDPA (26 μs for 16 heads):**
- Our result: 235 μs for 16 heads
- **9× slower** than PyTorch

**Why still slower?**
1. **3 kernel launches** vs PyTorch's fused kernel
2. **Global memory traffic** between kernels
3. **Low GEMM utilization** (7-18% vs should be >70%)

**To beat PyTorch, need:**
- Single fused kernel
- OR much larger batch sizes (B=32+)
- OR different workload characteristics

---

**NCU = GROUND TRUTH. All other measurements are bullshit.**
