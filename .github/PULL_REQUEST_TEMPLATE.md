<!-- 
Periodic Labs - Epistemic CI
PR Template for Ready-to-Merge Changes
-->

## 📋 Summary

<!-- One-sentence description of the change -->

**Rationale:**  
<!-- Why is this change necessary? What problem does it solve? -->

**Impact:** ⭐⭐⭐  
<!-- Rate 1-5 stars for business/scientific impact -->

**Effort:** **[S/M/L]**  
<!-- S=<4h, M=4-16h, L=>16h -->

---

## 🎯 Changes

<!-- Bullet list of key changes -->

- [ ] Change 1
- [ ] Change 2
- [ ] Change 3

---

## ✅ Acceptance Criteria

<!-- Checklist of requirements for merge -->

- [ ] All tests pass (`pytest -v`)
- [ ] Coverage ≥85% (`pytest --cov-fail-under=85`)
- [ ] Secrets scan clean (trufflehog)
- [ ] Reproducible with `--seed 42`
- [ ] Documentation updated (README, docstrings)
- [ ] Performance within budget (time/cost caps)

### ⚡ Performance Checks (if touching `csrc/`, `bench/`, `cudadent42/`)

- [ ] Correctness fuzzing passes (`python cudadent42/bench/correctness_fuzz.py`)
- [ ] No regression > 3% vs baseline
- [ ] If claiming improvement: ≥10% speedup with non-overlapping 95% CIs
- [ ] Nsight evidence attached (`.ncu-rep` or screenshot)

---

## ⚡ Performance Intent & Hypothesis

<!-- Required if touching CUDA kernels or benchmarks -->

**Target Shape(s):** <!-- e.g., B=32, H=8, S=512, D=64 -->

**Hypothesis (Bottleneck):** <!-- What bottleneck does this PR address? -->

**Nsight Evidence:** 
<!-- Link to Nsight report or attach screenshot -->
- Before: `artifacts/ncu/baseline_*.ncu-rep`
- After: `artifacts/ncu/optimized_*.ncu-rep`

**Result:**
- Median Δ: <!-- e.g., +12.5% faster -->
- CI Overlap: <!-- Yes/No -->
- Cliff's δ: <!-- e.g., 0.42 (medium effect) -->
- Verdict: <!-- ✅ Improvement / ⚠️ No significant difference / ❌ Regression -->

---

## 🧪 Test Plan

<!-- How was this tested? -->

**Local Testing:**
```bash
# Commands used to verify
make mock SEED=42
pytest tests/test_*.py -v
```

**Expected Output:**
```
✅ All tests pass
✅ Artifacts match expected (bit-identical if seeded)
✅ No regressions in CI metrics
```

---

## 📊 Evidence

<!-- Links to artifacts, logs, screenshots -->

- **Experiment Ledger:** `experiments/ledger/{run_id}.json`
- **CI Metrics:** `artifact/ci_metrics.json`
- **Coverage Report:** `coverage.json`

---

## 🔗 Related Issues

<!-- Link to related issues/PRs -->

Closes #<!-- issue number -->  
Related: #<!-- related issue -->

---

## 📝 Checklist (for reviewer)

- [ ] Code follows project style (PEP 8, 4-space indent)
- [ ] No secrets committed (.env.example only)
- [ ] Error handling present (no exposed stack traces)
- [ ] Logging added for key paths
- [ ] Schema validated (if data model changes)
- [ ] Backward compatible (or migration provided)

---

## 🚀 Deployment Notes

<!-- Any special deployment considerations -->

**Breaking Changes:**  
<!-- None / List breaking changes -->

**Required Actions:**  
<!-- Steps needed post-merge (e.g., run migrations, update secrets) -->

---

**Signed-off-by:** <!-- Your name <email@example.com> -->  
**Date:** <!-- YYYY-MM-DD -->  
**Review Requested:** @<!-- reviewer GitHub username -->
