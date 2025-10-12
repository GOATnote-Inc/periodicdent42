# Session N+7C PAUSED: NaN Bug Root Cause Found (Fix Placement Issue)

**Duration**: 2h 30min (exceeded 90min budget)  
**Cost**: $0.50 GPU + $2.00 AI = $2.50  
**Status**: ⏸️ PAUSED - Root cause found, fix needs correct placement

---

## 🎯 Objective
Debug and fix NaN issue (tile 0 produces NaN for S≥65, tile 1+ correct)

---

## ✅ Achievements

### 1. Root Cause Identified

**The Bug**:
```cuda
// When all attention scores are -INFINITY (causal masking):
m_tile = -INFINITY
exp_val = expf(score - m_tile) = expf(-INFINITY - (-INFINITY)) 
        = expf(NaN) = NaN
```

**Why It Happens**:
- Query tile 0 (queries 0-63) processes K/V tile 1 (keys 64-127)
- With causal masking: all scores are -INFINITY (keys > queries)
- Computing exp with two -INFINITY values produces NaN

**Evidence**:
```
DEBUG: query_tile=0, query=0, l_i=nan, m_i=-0.000085, acc_o[0]=nan
```

### 2. Fix Designed

**Solution**: Skip fully-masked K/V tiles
```cuda
if (isinf(m_tile) && m_tile < 0.0f) {
    __syncthreads();
    continue;  // Skip this K/V tile
}
```

### 3. Validated on Test Kernel

Applied fix to Session N+5 kernel → **All 7 configs passed**:
```
S=4    D=4    max_diff=0.000008  ✅ PASS
S=64   D=64   max_diff=0.000008  ✅ PASS
S=65   D=64   max_diff=0.000008  ✅ PASS
S=128  D=64   max_diff=0.000015  ✅ PASS
S=192  D=64   max_diff=0.000008  ✅ PASS
S=256  D=64   max_diff=0.000015  ✅ PASS
S=512  D=64   max_diff=0.000008  ✅ PASS
```

---

## ⚠️ Current Blocker: Fix Placement

**Issue**: NaN check placed BEFORE `m_tile` is computed

**Wrong**:
```cuda
float m_tile = -INFINITY;
if (isinf(m_tile) ...) continue;  // ← Checks before loop!
for (int kv = 0; kv < tile_size; ++kv) {
    m_tile = fmaxf(m_tile, smem_S[...]);
}
```

**Correct**:
```cuda
float m_tile = -INFINITY;
for (int kv = 0; kv < tile_size; ++kv) {
    m_tile = fmaxf(m_tile, smem_S[...]);
}
// ← Fix goes here (after loop completes)
if (isinf(m_tile) && m_tile < 0.0f) {
    __syncthreads();
    continue;
}
```

---

## 📊 Time Breakdown

| Phase | Duration | Status |
|-------|----------|--------|
| Compare with N+5 | 30 min | ✅ Complete |
| Isolate bug | 45 min | ✅ Complete |
| Design & test fix | 30 min | ✅ Complete |
| Apply to Split-K | 45 min | ⚠️ Placement issue |
| **Total** | **150 min** | **85% complete** |

**Why Exceeded**: Multiple rebuild cycles due to fix placement debugging

---

## 🔍 Key Findings

### Pattern 13: Debug Printf is Critical

**What Worked** ✅:
```cuda
if (query_idx_in_tile == 0 && query_tile_idx == 0) {
    printf("DEBUG: l_i=%f, m_i=%f\n", l_i, m_i);
}
```
Immediately revealed `l_i=nan` before normalization

**Lesson**: Add debug printfs early, not after hours of speculation

### Configuration Change Risk

**Discovery**: Session N+5's kernel (validated with 384 threads, 128 tiles) failed with L4 config (256 threads, 64 tiles)

**Root Cause**: Not the config itself, but causal masking exposed latent bug

**Lesson**: Configuration changes can expose edge cases in tested code

---

## 📋 Next Session: N+7D (30 min)

**Objective**: Apply fix correctly, validate, measure

**Steps**:
1. **Fix Placement** (5 min)
   - Move NaN check after `m_tile` loop
   - Verify with `sed -n` before rebuild
   
2. **Rebuild & Validate** (10 min)
   - Clean build
   - Test 7 configs
   - Verify all pass
   
3. **Performance Measurement** (10 min)
   - Run bench_correctness_and_speed.py
   - Compare to Session N+6 baseline
   - Document speedup (if any from L4 config)
   
4. **Document & Commit** (5 min)
   - SESSION_N7_COMPLETE.md
   - Update GPU status
   - Push to GitHub

**Expected**: ✅ Correctness validated, performance measured, Priority 1 paused (Split-K deferred to N+7E)

---

## 💰 Cost

| Item | Cost |
|------|------|
| GPU (150 min @ $0.20/hr) | $0.50 |
| AI/Cursor | $2.00 |
| **Total** | **$2.50** |

### Cumulative (Sessions N through N+7C)
| Total Sessions | Total Time | GPU Cost | AI Cost | Grand Total |
|----------------|------------|----------|---------|-------------|
| 8 sessions | 884 min | $2.94 | $12.26 | $15.20 |

**ROI**: $15.20 investment, NaN bug root cause found + fix validated ✅

---

## 🎓 Meta-Learning

### What Worked ✅
1. **Reverting to known-good kernel** (Session N+5) to isolate issue
2. **Printf debugging** revealed NaN source immediately
3. **Systematic testing** (S=64 works, S=65 fails → multi-tile bug)

### What to Improve ⚠️
1. **Test fix placement** before commit (visual inspection of code context)
2. **Use smaller rebuild cycles** (test after each change, not after full implementation)
3. **Add unit tests** for edge cases (fully-masked tiles)

---

**Status**: ⏸️ **PAUSED - 85% COMPLETE**

**GPU**: RUNNING (ready for N+7D)  
**Next**: Apply fix correctly → validate → measure (30 min)  
**Blocker**: Fix placement (trivial, 5 min to resolve)

---

*Generated: October 12, 2025 9:03 PM UTC*  
*Duration: 2h 30min*  
*Cost: $2.50*  
*Result: Root cause found, fix validated on test kernel, placement issue to resolve*
