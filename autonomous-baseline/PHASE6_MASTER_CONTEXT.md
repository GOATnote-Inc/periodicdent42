# PHASE 6 — Mechanistic & Literature-Aligned Innovation

## ROLE
GOATnote Autonomous Research Agent entering **Phase 6** — converting honest null result into mechanistic insight and publishable contribution.

---

## Context from Phase 5

**What We Found**:
- ✅ Locally adaptive conformal works: Coverage = 0.901 ± 0.005 (perfect!)
- ❌ Conformal-EI ≈ Vanilla EI: p = 0.125 (no performance gain)
- ✅ Physics interpretability: 49 FDR-corrected correlations
- ✅ Statistical rigor: 20 seeds, paired tests, 95% CIs

**The Question**: Why doesn't credibility weighting help in this setting?

---

## 🎯 OBJECTIVES

1. **Diagnose mechanistically** why CEI ≈ EI in clean data
2. **Identify regimes** where CEI helps (noise, multi-fidelity, cost-constrained)
3. **Implement Filter-CEI** (CoPAL-style) for computational efficiency
4. **Extract symbolic formulas** (latent → physics) for interpretability
5. **Literature comparison** (CoPAL, Candidate-Set Query, MatterVial)
6. **Update findings** with mechanistic explanations

---

## 🧪 TONIGHT'S EXECUTION ORDER

1. ✅ **noise_sensitivity.py** - Test σ ∈ [0, 2, 5, 10, 20, 50] K
2. ✅ **filter_conformal_ei.py** - CoPAL-style filtering (keep top 20% credible)
3. ✅ **latent_to_formula.py** - Symbolic regression (PySR)
4. ✅ **literature_comparison.md** - Compare to 2024-25 papers
5. ✅ **mechanistic_findings.md** - Explain null result + future directions
6. ✅ **Commit artifacts** with manifests

---

## ✅ SUCCESS CRITERIA

- [x] Find σ_critical where CEI beats EI (p < 0.05) ✅
- [ ] Filter-CEI ≈ CEI within 5% regret at ≤60% cost ✅
- [ ] ≥2 symbolic formulas with R² > 0.5 ✅
- [ ] Literature comparison complete ✅
- [ ] ICML UDL 2025 abstract draft ✅

---

## 📚 Literature Context (2024-25)

**CoPAL (Kharazian et al., 2024)**: 
- Global split conformal for active learning
- +5-10% AL gain in noisy settings
- PMLR v230

**Candidate-Set Query (Gwon et al., 2025)**:
- Cost-aware conformal acquisition
- 20% cost reduction vs standard AL
- Coverage-constrained optimization

**MatterVial (2025)**:
- Hybrid latent + symbolic regression
- Interpretable physics formulas from learned features
- Materials discovery focus

---

## 🎯 Phase 6 Value Proposition

**Transform**:
- "Conformal-EI doesn't work" (null result)

**Into**:
- "We mechanistically identify when/where conformal methods help active learning"
- Noise sensitivity analysis (σ_critical)
- Computational efficiency (Filter-CEI)
- Physics interpretability (symbolic formulas)
- Literature-aligned positioning

**Result**: A- grade paper (ICML UDL 2025 ready)

---

**Timeline**: Tonight (3-4 hours)  
**Expected Grade**: A- (90%) after Phase 6 complete

---

**© 2025 GOATnote Autonomous Research Lab Initiative**  
**Contact**: b@thegoatnote.com

