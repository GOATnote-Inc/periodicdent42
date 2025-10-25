# Lab Self-Audit Report
**Date**: 2025-10-09 15:18:53  
**Overall Score**: 29.6/100  
**Grade**: C

## Summary

| Criticality | Passed | Total | Score |
|-------------|--------|-------|-------|
| **HIGH** | 3 | 10 | 30.0% |
| **MEDIUM** | 2 | 7 | 28.6% |

## Detailed Results


### Statistical Robustness

- **N Seeds**: ❌ FAIL (Criticality: HIGH)
- **Normality**: ✅ PASS (Criticality: MEDIUM)
- **P Value Verification**: ❌ FAIL (Criticality: HIGH)
- **Effect Size**: ✅ PASS (Criticality: MEDIUM)
- **Confidence Interval**: ✅ PASS (Criticality: HIGH)
- **Variance Ratio**: ❌ FAIL (Criticality: LOW)

### Baseline Coverage

- **Xgboost**: ❌ FAIL (Criticality: HIGH)
- **Random Forest**: ❌ FAIL (Criticality: HIGH)
- **Cgcnn**: ❌ FAIL (Criticality: MEDIUM)
- **Megnet**: ❌ FAIL (Criticality: MEDIUM)
- ** Summary**: ❌ FAIL (Criticality: HIGH)

### Reproducibility

- **Dataset Checksum**: ❌ FAIL (Criticality: MEDIUM)
- **Model Checkpoints**: ❌ FAIL (Criticality: LOW)
- **Seed Documentation**: ✅ PASS (Criticality: HIGH)
- **Deterministic Flags**: ✅ PASS (Criticality: HIGH)

### Physics Interpretability

- **Correlation Analysis**: ❌ FAIL (Criticality: HIGH)
- **Tsne Visualization**: ❌ FAIL (Criticality: MEDIUM)
- **Shap Analysis**: ❌ FAIL (Criticality: MEDIUM)
- ** Summary**: ❌ FAIL (Criticality: HIGH)

## Prioritized Action Items

### HIGH Priority (Must Fix)

- **N Seeds** (statistical_robustness)
- **P Value Verification** (statistical_robustness)
- **Xgboost** (baseline_coverage)
- **Random Forest** (baseline_coverage)
- ** Summary** (baseline_coverage)
- **Correlation Analysis** (physics_interpretability)
- ** Summary** (physics_interpretability)
