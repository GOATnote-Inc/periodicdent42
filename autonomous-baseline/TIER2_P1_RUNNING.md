# P1: DKL Ablation Experiment - RUNNING

**Status**: ⏳ IN PROGRESS  
**Start Time**: 20:56 PST, October 9, 2025  
**Estimated Completion**: 21:05 PST (8-10 minutes)

---

## Experiment Configuration

**Methods**: 4 (DKL, PCA+GP, Random+GP, GP-raw)  
**Seeds**: 3 (42-44)  
**Rounds**: 10 AL iterations per seed  
**Total Experiments**: 4 methods × 3 seeds × 10 rounds = 120 AL iterations

**Hardware**: Apple Silicon M-series, PyTorch 2.8.0, Python 3.13.5

---

## What's Being Tested

### Core Question
**Does DKL's neural network feature learning provide advantage over linear PCA?**

### Methods
1. **DKL** (81D → 16D learned): Neural net extracts features, GP on learned space
2. **PCA+GP** (81D → 16D PCA): Linear PCA, GP on PCA space
3. **Random+GP** (81D → 16D random): Random projection, GP on random space
4. **GP-raw** (81D): GP directly on all 81 features

### Critical Test
**If PCA+GP ≈ DKL**: Feature learning does NOT help (dimensionality reduction alone explains improvement)  
**If DKL >> PCA+GP**: Feature learning DOES help (neural net learns better representations than PCA)

---

## Risk Assessment

**HIGH RISK**: This experiment could invalidate the "DKL beats GP" claim

**Possible Outcomes**:

### Scenario A: DKL ≫ PCA+GP (validates claim)
- DKL significantly better (p < 0.05, Δ > 1.5 K)
- **Action**: Document as evidence of feature learning value
- **Impact**: Strengthens Phase 10 Tier 2 contribution

### Scenario B: DKL ≈ PCA+GP (invalidates claim)
- Statistical equivalence (p > 0.05, |Δ| < 1.0 K)
- **Action**: Honest reporting, reframe as "dimensionality reduction" not "feature learning"
- **Impact**: Requires documentation updates, shifts narrative to "efficient GP" not "DKL superiority"

### Scenario C: PCA+GP > DKL (worst case)
- PCA significantly better (p < 0.05, Δ < -1.5 K)
- **Action**: Report honestly, document as negative result
- **Impact**: Major reframing required, focus shifts to other contributions (conformal, OOD)

---

## Monitoring

**Log File**: `/tmp/dkl_ablation_fixed.log`

**Monitor Progress**:
```bash
tail -f /tmp/dkl_ablation_fixed.log
```

**Check Status**:
```bash
ps aux | grep tier2_dkl_ablation_real | grep -v grep
```

---

## Next Steps (Conditional on Results)

### If Experiment Succeeds
1. Run analysis: `python scripts/analyze_dkl_ablation.py`
2. Generate plots (RMSE comparison, time comparison)
3. Update NOVELTY_FINDING.md with results
4. Commit with honest findings
5. Update Tier 2 status to 4/5 complete

### If Experiment Fails (Technical)
1. Debug error
2. Re-run with --seeds 2 --rounds 5 (quick test)
3. Scale up once validated

---

## Scientific Integrity Commitment

**Regardless of outcome, we will**:
- Report results honestly
- Include statistical tests (paired t-tests, effect sizes)
- Document limitations
- Update all claims to match evidence
- No cherry-picking or p-hacking

**If PCA+GP ≈ DKL**:
- Update claims: "DKL provides efficient 16D representation" (not "better than GP")
- Emphasize other contributions: perfect calibration, adaptive sharpness, validated regret
- Reframe as "comprehensive BO framework" not "DKL superiority"

---

**Status**: Experiment running, monitoring in progress  
**ETA**: 5-8 minutes remaining

