# Session N+7D: Split-K Implementation Progress

**Duration**: 45 minutes  
**Cost**: $0.15 GPU + $0.75 AI = $0.90  
**Status**: ⏸️ PAUSED - Correctness bug in Split-K partial kernel

---

## ✅ Achievements

### 1. NaN Fix Validated & Applied
- **FA-1**: Perfect correctness (diff < 0.00002) ✅
- Applied to both FA-1 and Split-K kernels
- Fix: Skip fully-masked K/V tiles where `m_tile == -INFINITY`

### 2. Split-K Build Successful
- Restored Split-K implementation from Session N+7A
- Applied NaN fixes to both kernels
- Build successful: 6 kernels compiled (FA-1 + Split-K, FP16 + BF16)
- Binary size: 379 KB

### 3. Correctness Issue Identified
- **FA-1**: diff=0.000015 ✅ PASS
- **Split-K**: diff=0.19-0.32 ❌ FAIL

**Critical Finding**: Bug exists even with S=64 (1 tile × 1 tile, no reduction needed)
→ **Bug is in Split-K partial kernel, not reduction kernel**

---

## 🔍 Debug Analysis

### Test Results
```
S=64: q_tiles=1, kv_tiles=1
  FA-1: diff=0.000008 ✅
  Split-K: diff=0.191650 ❌  ← No reduction involved!

S=128: q_tiles=2, kv_tiles=2
  FA-1: diff=0.000015 ✅
  Split-K: diff=0.224976 ❌
```

### Hypotheses (In Order of Likelihood)

1. **Online Softmax Scaling Error** (80% likely)
   - Split-K partial computes local softmax
   - May not be storing max/sum correctly for reduction
   - Or applying incorrect scaling to output

2. **Memory Indexing Bug** (15% likely)
   - Partial output indexing: `partial_O[b][h][q_tile][kv_tile][query][d]`
   - Could be writing to wrong offset

3. **Tile Boundary Handling** (5% likely)
   - Edge case when seq_len not divisible by TILE_SIZE

---

## 📋 Next Session: Fix Split-K Partial (30-45 min)

### Phase 1: Isolate Bug (10 min)
```python
# Test with S=64, print intermediate values
- Print partial_max, partial_sum after partial kernel
- Print O values after reduction
- Compare with FA-1 intermediate values
```

### Phase 2: Fix Computation (15 min)
**Most Likely Fix**:
```cpp
// In partial kernel, ensure correct output scaling
// Current: acc_o[d] contains sum(P[kv] * V[kv][d])
// Should: acc_o[d] = sum(exp(S[kv] - local_max) * V[kv][d])
// Then reduction will apply: out = sum(partial * exp(max_partial - global_max))
```

### Phase 3: Validate & Measure (15 min)
- Test 4 configs: S=64,128,256,512
- Verify all pass (diff < 0.1)
- Measure performance vs FA-1 and PyTorch
- **Expected**: 5-10× speedup over FA-1 @ S=128

---

## 📊 Performance Baseline (FA-1)

| S | PyTorch (ms) | FA-1 (ms) | Speedup |
|---|--------------|-----------|---------|
| 64 | 0.051 | 0.912 | 0.056× |
| 128 | 0.052 | 1.811 | 0.028× |
| 256 | 0.057 | 3.529 | 0.016× |
| 512 | 0.053 | 6.987 | 0.008× |

**Target with Split-K**: 
- S=128: < 0.5 ms (4× faster than FA-1, 10× faster than current best)
- Speedup vs PyTorch: 0.1-0.2× (still slower, but much better)

---

## 💰 Cost Tracking

### Session N+7D
- GPU: $0.15
- AI: $0.75
- Total: $0.90

### Cumulative (Sessions N through N+7D)
| Total Sessions | Total Time | GPU Cost | AI Cost | Grand Total |
|----------------|------------|----------|---------|-------------|
| 9 sessions | 929 min | $3.09 | $13.01 | $16.10 |

---

## 🎓 Pattern 15: Validate Before Optimize

**Lesson Learned**: 
- Built Split-K successfully
- But didn't validate correctness before performance testing
- **Should**: Test correctness IMMEDIATELY after build
- **Result**: Found bug early (S=64 test), saving performance debugging time

**New Workflow**:
```
1. Build → 2. Quick correctness test → 3. If pass, benchmark → 4. If fail, debug
         ↓ (current: fail)
         Fix computation bug → validate → measure
```

---

**Status**: ⏸️ **PAUSED - 80% COMPLETE**

**GPU**: RUNNING (L4, 34.172.98.137)  
**Next**: Debug Split-K partial kernel (30-45 min)  
**Expected**: Fix → validate → 5-10× speedup  

---

*Generated: October 12, 2025 8:40 PM UTC*  
*Result: Split-K built, correctness bug isolated to partial kernel*
