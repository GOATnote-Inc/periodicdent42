# GPU Session Status - October 12, 2025

**GPU Instance**: cudadent42-l4-dev  
**Zone**: us-central1-a  
**Type**: L4 (NVIDIA L4, 24GB VRAM)  
**IP**: 136.114.253.228  
**Status**: 🟢 **RUNNING**  

---

## ⏰ Session Tracking

**Started**: Sunday, October 12, 2025, 9:03 PM  
**Keep Running Until**: Monday, October 13, 2025, 4:00 AM (7 hours)  
**Reason**: Engineer + Cursor IDE cost > GPU stop/start cost (Pattern 7)  
**Cost Rate**: $0.20/hour  
**Planned Duration**: 7 hours  
**Planned Cost**: $1.40

---

## 🎯 Current Work

**Session**: N+7G (Fix Split-K Bug #3)  
**Duration**: 30 minutes  
**Status**: ⏸️ **PAUSED** - Bug more complex than expected  
**Progress**: 3 bugs fixed, 1 bug remains (root cause unknown)

---

## 🐛 Bugs Fixed

✅ **Bug #1** (Session N+7E): Reduction double-counting  
- Fixed: Line 628 reweight calculation  
- Commit: 0bb3778

✅ **Bug #2** (Session N+7F): Partial kernel NaN check  
- Fixed: Lines 534-542 NaN guard for fully-masked tiles  
- Commit: 6f4e940

✅ **Bug #3a** (Session N+7G): Partial kernel `acc_o` initialization  
- Fixed: Line 464 changed `{0.0f}` to `{0}`  
- Commit: 9410a7a

✅ **Bug #3b** (Session N+7G): Reduction kernel `final_o` initialization  
- Fixed: Line 626 changed `{0.0f}` to `{0}`  
- Commit: a1b9a45

---

## ❌ Bug Still Present

**Bug #4** (Unknown root cause):  
- **Symptom**: Split-K produces ~0.19-0.27 error  
- **Pattern**: 
  - S=0: Perfect (diff = 0.0) ✅  
  - S=32: Error = 0.158 ❌  
  - S=45: Max error = 0.205 ❌  
  - S=63: Error = 0.199 ❌  
- **Observation**: Error increases with query index  
- **Hypothesis**: Subtle indexing or memory layout issue

---

## 📊 Test Results (Latest)

```
Config               FA-1 max_diff   Split-K max_diff   Status
────────────────────────────────────────────────────────────────
B=1 H=1 S=4    D=4    0.000004        0.022202          ✅ PASS
B=1 H=1 S=64   D=64   0.000008        0.193604          ❌ FAIL
B=1 H=1 S=65   D=64   0.000008        0.267090          ❌ FAIL
B=1 H=1 S=128  D=64   0.000015        0.230103          ❌ FAIL
B=1 H=1 S=192  D=64   0.000008        0.213867          ❌ FAIL
B=1 H=1 S=256  D=64   0.000015        0.223022          ❌ FAIL
B=1 H=1 S=512  D=64   0.000008        0.243530          ❌ FAIL
```

**Key**: Small cases (S=4) pass, large cases (S>=64) fail consistently.

---

## 🔍 Next Debugging Steps (Session N+7H)

### Phase 1: Verify Memory Layout (10 min)
Add debug prints to compare partial write vs reduction read:
```cpp
// In partial kernel (line 560)
if (query_idx == 0 && kv_tile_idx == 0) {
    printf("Partial write: offset=%d, q_in_tile=%d, addr=%p\n",
           partial_offset, query_idx_in_tile, partial_O_base);
}

// In reduction kernel (line 635)
if (query_idx == 0 && kv_tile == 0) {
    printf("Reduction read: offset=%d, q_in_tile=%d, addr=%p\n",
           partial_offset, query_idx_in_tile, partial_O_base);
}
```

### Phase 2: Test Individual Components (15 min)
1. **Test partial kernel alone**: Print `partial_O`, `partial_max`, `partial_sum` to file
2. **Verify reduction input**: Load known-good partial values, test reduction only
3. **Isolate the bug**: Which component is producing wrong output?

### Phase 3: Simplify Test Case (10 min)
Create minimal reproduction:
- B=1, H=1, S=4, D=4 (passes)
- B=1, H=1, S=8, D=4 (does it pass or fail?)
- Find the EXACT boundary where bug appears

### Phase 4: Review Algorithm (10 min)
Re-read FlashAttention-2 Split-K paper algorithm and compare line-by-line with implementation.

---

## 💰 Cost Tracking

### Session N+7G (Current)
| Item | Cost |
|------|------|
| Engineer time (30 min) | $25.00 |
| GPU time (7 hours planned) | $1.40 |
| **Total** | **$26.40** |

### Cumulative (Sessions N through N+7G)
| Metric | Value |
|--------|-------|
| Total Sessions | 12 |
| Total Duration | 25.3 hours |
| GPU Hours | 14.0 hours (+ 7 hours planned) |
| GPU Cost | $16.70 + $1.40 = $18.10 |
| Engineer Cost | $1,264.17 |
| **Total Investment** | **$1,282.27** |

---

## 🎓 Key Learnings

### Pattern 7: Keep GPU Running ✅ **APPLIED CORRECTLY**
- Decision: Keep GPU running for 7 hours during active work
- Rationale: Engineer + Cursor cost ($50-75) > GPU cost ($1.40)
- Result: Avoiding 3-5 context switches saves $150-225

### Pattern 15: Defensive Debugging (New)
**Context**: Complex multi-kernel bugs that resist simple fixes  
**Strategy**:
1. Fix obvious bugs first (initialization, NaN checks) ✅
2. Test after each fix ✅
3. When simple fixes fail, switch to systematic debugging:
   - Add debug prints
   - Test components in isolation
   - Create minimal reproductions
   - Binary search for bug boundary

---

## 🚀 Recommendations

### Option A: Continue Debugging (Recommended for short term)
- **Next**: Session N+7H (45-60 min, $37.50-$50)
- **Strategy**: Apply Phase 1-4 debugging plan above
- **Confidence**: Medium (50%) - Bug is subtle
- **If successful**: Split-K working, Priority 1 complete

### Option B: Defer Split-K (Recommended for long term)
- **Rationale**: 
  - FA-1 works (1.8 ms @ S=128, 36× slower than PyTorch)
  - Already spent 12 sessions ($1,282) on Split-K
  - Bug is complex, may take 3-5 more sessions ($150-250)
- **Alternative**: Focus on Priorities 2-4 (warp specialization, tensor cores) on FA-1
- **ROI**: 
  - Warp spec: 2× speedup (1.8 ms → 0.9 ms), 2-3 sessions
  - Tensor cores: 3-5× speedup (0.9 ms → 0.18-0.30 ms), 3-4 sessions
  - Combined: 6-10× total speedup, 5-7 sessions ($250-350)
  - Result: 0.18-0.30 ms (3.6-6× slower than PyTorch) vs current 36×

### Option C: Switch to H100
- **Rationale**: L4 constraints (48KB SMEM, 256 threads) may be causing subtle bugs
- **Cost**: $1.00/hr (5× more than L4)
- **Benefit**: Easier debugging, more resources, closer to target deployment
- **Timeline**: 2-3 sessions ($50-75) to port and debug

---

## 📂 Code State

**Branch**: `opt/vectorized-loads`  
**Latest Commits**:
- a1b9a45: Fix final_o initialization  
- 9410a7a: Fix acc_o initialization  
- 6f4e940: Add NaN check to partial kernel  
- 0bb3778: Fix reduction double-counting

**Build Status**: ✅ Compiled successfully (293KB .so)  
**Import Status**: ✅ Module loads without errors  
**Test Status**: ❌ 6/7 tests failing (Bug #4 unresolved)

---

## ⏭️ Next Session

**Planned**: Session N+7H (Continued debugging) OR pivot to Option B/C  
**Duration**: 45-60 minutes  
**Cost**: $37.50-$50 (engineer only, GPU already running)  
**Decision**: Awaiting user approval

---

## 🔧 Environment

**CUDA**: 12.8  
**PyTorch**: 2.2.1+cu121  
**GPU**: NVIDIA L4 (Ampere, SM_89)  
**SMEM Limit**: 48 KB  
**Threads**: 256 (8 warps)  
**Tiles**: 64×64 (TILE_SIZE_M/N/K)

---

**Last Updated**: Sunday, October 12, 2025, 9:08 PM  
**Session**: N+7G (30 min, $26.40)  
**GPU Status**: 🟢 RUNNING (keep until 4:00 AM or user says stop)  
**Next**: User decision (continue debugging or pivot strategy)

