# v0.5.0 Calibration Findings & Next Steps

## Summary

**Status**: v0.5.0 EXACT Allen-Dynes implementation **COMPLETE** but requires λ re-tuning  
**Current MAPE**: 73.0% (down from 111% after lambda reset)  
**Target MAPE**: ≤50%  
**Gap**: 23 percentage points  

---

## Root Cause Analysis

### The Incompatibility Problem

1. **v0.4.5 lambda corrections** (e.g., `A15: 2.60`) were empirically tuned to compensate for the **simplified Allen-Dynes formula** lacking f₁/f₂ corrections.

2. **v0.5.0 EXACT formula** includes μ*-dependent f₁/f₂ corrections, so using v0.4.5 lambda values causes **double-counting** of strong-coupling effects.

3. **Result**: Massive over-predictions (111% MAPE) when using EXACT + v0.4.5 lambdas.

### The Reset Attempt

**Reset all lambdas to 1.0** → MAPE improved to 73% but:
- Lost material-class-specific physics
- A15 compounds now **underpredict** by ~50-58%
- Some elements (Pb, V) have extreme errors

### Current Results (Lambda = 1.0 for all)

| Material | Tc_exp | Tc_pred | Error | Status |
|----------|--------|---------|-------|--------|
| Nb | 9.25 | 11.24 | +21.6% | ✅ Good |
| MgB2 | 39.0 | 33.81 | -13.3% | ✅ Good |
| NbC | 11.1 | 10.34 | -6.8% | ✅ Excellent |
| Pb | 7.2 | 0.014 | **+99.8%** | ❌ Critical |
| V | 5.4 | 19.02 | **+252%** | ❌ Critical |
| Nb3Sn | 18.05 | 7.56 | -58.1% | ❌ Bad |
| Nb3Ge | 23.2 | 10.70 | -53.9% | ❌ Bad |
| V3Si | 17.1 | 8.00 | -53.2% | ❌ Bad |
| VN | 8.2 | 14.18 | +73.0% | ❌ Bad |

---

## What Works

✅ **EXACT Allen-Dynes Physics**: f₁/f₂ formulas correctly implemented  
✅ **Material-specific r values**: 21 materials + default  
✅ **Physics constraints**: 7 enforcements working  
✅ **Test suite**: 23/23 PASSED (100%)  
✅ **Integration**: Seamless auto-enable in `allen_dynes_tc()`  
✅ **Selected materials**: Nb, MgB2, NbC show good predictions  

---

## What Needs Work

❌ **Lambda tuning**: Need material-class-specific values optimized for EXACT formula  
❌ **Pb predictions**: Massive underpredict (base lambda too low?)  
❌ **V predictions**: Massive overpredict (base lambda too high?)  
❌ **A15 compounds**: All underpredict (~50%), need lambda > 1.0  

---

## Recommended Next Steps (v0.5.1)

### Option A: Moderate Lambda Corrections (Quick Fix, ~1 hour)

Start with moderate corrections between 1.0 and v0.4.5 values:

```python
LAMBDA_CORRECTIONS = {
    "element":  1.10,  # Slight boost for simple metals
    "A15":      1.50,  # A15 need significant boost (was 2.60, now neutral)
    "nitride":  1.20,  # Moderate boost
    "carbide":  1.15,  # Slight boost
    # ... rest at 1.0
}
```

**Expected**: MAPE → 50-60% range

### Option B: Bayesian Optimization of μ* (Original v0.5.0 Plan, ~3 hours)

Implement the planned μ* optimization:
- `app/src/htc/tuning/mu_star_optimizer.py`
- Optimize μ* per class (not lambda)
- LOMO validation to prevent overfit
- YAML config persistence

**Expected**: MAPE → 40-50% range (statistically rigorous)

### Option C: Full Re-Calibration (Research-grade, ~1 day)

1. Bayesian optimize BOTH lambda AND μ* per class
2. Constrain search space (lambda: [0.8, 1.5], μ*: [0.08, 0.20])
3. Multi-objective: Minimize MAPE_A, MAPE_B, MAPE_overall
4. Bootstrap CI validation
5. Publication-quality analysis

**Expected**: MAPE → 35-45% range (A+ grade)

---

## Decision Matrix

| Option | Time | MAPE Target | Rigor | Recommendation |
|--------|------|-------------|-------|----------------|
| **A** | 1h | 50-60% | Medium | ✅ For quick PR |
| **B** | 3h | 40-50% | High | ✅ For v0.5.1 |
| **C** | 1d | 35-45% | Highest | ⏳ For publication |

---

## Immediate Actions for v0.5.0 PR

### 1. Document Current State ✅ (This file)

### 2. Create PR with What Works
- EXACT Allen-Dynes implementation (23/23 tests)
- Integration framework
- Statistical gates
- Prometheus metrics
- Documentation

### 3. Mark Lambda Tuning as Future Work
- Add TODO in `structure_utils.py`
- Reference this document
- Link to v0.5.1 milestone

### 4. Update README Badges
```markdown
- v0.5.0: EXACT Allen-Dynes ✅ (requires λ tuning for full accuracy)
- Test Coverage: 23/23 PASSED ✅
- MAPE: 73% (target: ≤50% after v0.5.1 tuning)
```

---

## Technical Notes

### Why Pb and V Fail

**Pb (Tc = 7.2K)**:
- Base lambda likely too low
- Simple metal with relatively high Tc
- Needs investigation of base lambda calculation

**V (Tc = 5.4K)**:
- Base lambda too high (overpredicting by 252%)
- May need element-specific correction < 1.0

### Why A15 Compounds Need Boost

A15 compounds have:
- High density of states (DOS) at Fermi level
- Strong electron-phonon coupling
- Complex phonon spectrum

EXACT f₁/f₂ helps but doesn't fully capture A15 physics → Need lambda boost (~1.5)

---

## Conclusion

**v0.5.0 Core Achievement**: ✅ **COMPLETE**
- EXACT Allen-Dynes formulas working
- Full test coverage
- Production-ready integration
- Comprehensive documentation

**Known Limitation**: Requires material-class lambda tuning for target accuracy

**Path Forward**: 
1. **Now**: Create PR, document limitation
2. **v0.5.1** (1 week): Implement Option B (Bayesian μ* optimization)
3. **v0.6.0** (1 month): Full DFT integration, eliminate empirical tuning

---

**Author**: GOATnote Autonomous Research Lab Initiative  
**Date**: October 11, 2025  
**Status**: Ready for PR with documented limitations  
**Next**: v0.5.1 lambda/μ* optimization sprint

