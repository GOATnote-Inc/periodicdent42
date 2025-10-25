# FlashCore: Evidence-Based Optimization Session

**Date**: October 22, 2025  
**Approach**: Profile → Identify → Fix → Measure  
**Current**: 279 μs (working kernel)  
**Target**: <26 μs (beat PyTorch SDPA)  
**Status**: 🔬 **PROFILING IN PROGRESS**

---

## 🎯 **Mission**

**Stop guessing. Start measuring.**

1. ✅ Profile working kernel with NCU
2. ⏳ Find #1 bottleneck (memory? TC? stalls?)
3. ⏳ Apply proven fix (cp.async / warp spec / fusion)
4. ⏳ Measure speedup
5. ⏳ Repeat until <26 μs

---

## 📊 **Step 1: NCU Profiling** (NOW)

### **Command**
```bash
cd ~/flashcore
ncu --set full \
    --launch-skip 10 \
    --launch-count 1 \
    --target-processes all \
    python3 -c "
import torch
from build_fused import build_fused
fc = build_fused()
Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
softmax_scale = 1.0 / (64 ** 0.5)
for _ in range(20): O = fc.forward(Q, K, V, softmax_scale)  # Warmup
torch.cuda.synchronize()
O = fc.forward(Q, K, V, softmax_scale)  # Profile this
torch.cuda.synchronize()
"
```

### **Key Metrics to Extract**

| Metric | What It Tells Us | Action If Low |
|--------|------------------|---------------|
| **sm__pipe_tensor_cycles_active.pct** | Tensor Core utilization | Add WMMA, improve tiling |
| **dram__throughput.pct** | Memory bandwidth usage | Add cp.async, coalesce |
| **smsp__warp_issue_stalled_.*_pct** | What's blocking warps | Fix specific stall reason |
| **l2_cache_hit_rate** | Cache efficiency | Improve data reuse |
| **achieved_occupancy** | Active warps | Reduce registers/SMEM |

---

## 🔍 **Expected Bottlenecks** (Hypotheses)

### **Hypothesis 1: Memory Bound** (70% probability)
**Symptoms**: 
- `dram__throughput.pct` > 50%
- `smsp__warp_issue_stalled_mio_throttle.pct` high

**Fix**: cp.async double-buffering (overlap compute + memory)
**Expected Gain**: 1.5-2.0× speedup → 140-190 μs

---

### **Hypothesis 2: Low Tensor Core Utilization** (60% probability)
**Symptoms**:
- `sm__pipe_tensor_cycles_active.pct` < 30%
- Many scalar ops instead of WMMA

**Fix**: More WMMA coverage, better fragment reuse
**Expected Gain**: 1.3-1.7× speedup → 165-215 μs

---

### **Hypothesis 3: Warp Stalls** (50% probability)
**Symptoms**:
- `smsp__warp_issue_stalled_barrier.pct` > 10%
- Too many `__syncthreads()`

**Fix**: Warp-level sync (`__syncwarp`), reduce barriers
**Expected Gain**: 1.2-1.4× speedup → 200-230 μs

---

### **Hypothesis 4: Register/SMEM Pressure** (30% probability)
**Symptoms**:
- `achieved_occupancy` < 50%
- Register spills

**Fix**: Reduce live variables, reuse buffers
**Expected Gain**: 1.1-1.3× speedup → 215-255 μs

---

## 📝 **Profiling Log**

### **Run 1: Baseline Profile**
```
[Waiting for NCU output...]
```

**Findings**:
- [To be filled after NCU run]

**Bottleneck Identified**:
- [Primary issue]

**Recommended Fix**:
- [Specific optimization]

---

## 🚀 **Optimization Iteration Plan**

### **Iteration 1: Fix #1 Bottleneck**
- **Before**: 279 μs
- **Fix**: [TBD based on NCU]
- **Expected**: [TBD]
- **Actual**: [TBD]

### **Iteration 2: Fix #2 Bottleneck**
- **Before**: [Result from Iter 1]
- **Fix**: [Next bottleneck]
- **Expected**: [TBD]
- **Actual**: [TBD]

### **Iteration 3: Polish**
- **Before**: [Result from Iter 2]
- **Fix**: [Micro-optimizations]
- **Expected**: <50 μs
- **Target**: <26 μs ✅

---

## ⏱️ **Time Tracking**

| Step | Est. Time | Actual | Status |
|------|-----------|--------|--------|
| **NCU Profile** | 30 min | ⏳ | In progress |
| **Analysis** | 30 min | - | Pending |
| **Fix #1** | 60 min | - | Pending |
| **Test #1** | 15 min | - | Pending |
| **Fix #2** | 60 min | - | Pending |
| **Test #2** | 15 min | - | Pending |
| **Polish** | 30 min | - | Pending |
| **Total** | **4h** | - | - |

---

**Current Step**: Running NCU profile...
**Next**: Analyze metrics and identify #1 bottleneck

