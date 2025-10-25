# ADR-002: Uncertainty Calibration via Conformal Prediction

**Status**: Accepted  
**Date**: 2024-10-09  
**Deciders**: Research Team, Safety Committee

---

## Context

Autonomous laboratories require **calibrated uncertainty** for safe decision-making:

- **Underconfident predictions**: Waste budget on conservative choices
- **Overconfident predictions**: Risk expensive failures (synthesis, characterization)

Traditional ML uncertainty methods (e.g., ensemble variance, MC dropout) are **not calibrated by default**. Prediction intervals often have <80% coverage when 95% is claimed.

---

## Decision

**We will use conformal prediction to guarantee calibrated uncertainty intervals**, combined with model-specific uncertainty estimates (RF variance, MC dropout, NGBoost parametric).

**Two-Stage Approach**:
1. **Model uncertainty**: RF/MLP/NGBoost provides initial μ, σ estimates
2. **Conformal calibration**: Adjust intervals to achieve exact coverage on held-out set

---

## Rationale

### Why Conformal Prediction?

1. **Distribution-Free Guarantee**
   - No assumptions on data distribution or model form
   - Mathematically proven coverage: P(y ∈ [ŷ - q, ŷ + q]) ≥ 1-α

2. **Post-Hoc Calibration**
   - Works with any base model (RF, NN, GP, ensemble)
   - Does not require retraining
   - Can recalibrate as new data arrives

3. **Finite-Sample Guarantee**
   - Coverage holds for finite calibration sets (N ≥ 100)
   - Does not require asymptotic assumptions

4. **Mondrian Extension**
   - Per-group calibration (e.g., by chemical family)
   - Handles heteroscedastic noise

### Why Not Alternatives?

| Method | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Ensemble Variance** | Fast, interpretable | Not calibrated (often 70-80% coverage) | Use as feature, not final PI |
| **Bayesian NN** | Principled posterior | Computationally expensive, requires tuning | Too slow for autonomous lab |
| **Quantile Regression** | Direct PI estimation | Still requires calibration | Good base model, combine with conformal |
| **Conformal** | Guaranteed coverage | Requires calibration set | **Selected** |

---

## Implementation

### Split Conformal (Global)

```python
# 1. Train model on train set
model.fit(X_train, y_train)

# 2. Compute nonconformity scores on calibration set
residuals = abs(y_cal - model.predict(X_cal))
q = np.quantile(residuals, 0.95)  # 95% coverage

# 3. Prediction interval
y_pred = model.predict(X_test)
PI = [y_pred - q, y_pred + q]
```

**Guarantee**: ≥95% of test targets fall within PI (with high probability).

---

### Mondrian Conformal (Per-Family)

```python
# Partition calibration set by chemical family
for family in families:
    residuals_family = abs(y_cal[family] - model.predict(X_cal[family]))
    q_family[family] = np.quantile(residuals_family, 0.95)

# Prediction interval (family-specific)
PI = [y_pred - q_family[family], y_pred + q_family[family]]
```

**Advantage**: Tighter intervals for well-modeled families, wider for uncertain families.

---

## Validation Metrics

### 1. PICP (Prediction Interval Coverage Probability)

```
PICP = (1/N) Σ I(y_i ∈ PI_i)
```

**Target**: 0.95 ± 0.01 for 95% PI  
**CI Gate**: Reject if PICP ∉ [0.94, 0.96]

---

### 2. ECE (Expected Calibration Error)

Bin predictions by predicted probability, compute:

```
ECE = Σ (|observed_freq - predicted_prob|) · bin_weight
```

**Target**: ≤ 0.05  
**CI Gate**: Reject if ECE > 0.05

---

### 3. PI Width

```
Width = (1/N) Σ (PI_upper - PI_lower)
```

**Target**: Minimize subject to PICP ≥ 0.94  
**Report**: Median and 90th percentile width

---

## Deployment Gates

### Gate 1: Calibration on Validation Set

Before deployment, verify:
- ✅ PICP@95% ∈ [0.94, 0.96] on validation set
- ✅ ECE ≤ 0.05
- ✅ Median PI width < 20 K (domain-specific threshold)

### Gate 2: Calibration on Test Set (Final)

After all hyperparameter tuning:
- ✅ PICP@95% ∈ [0.94, 0.96] on test set (never seen during development)
- ✅ ECE ≤ 0.05

### Gate 3: Per-Family Calibration (Mondrian)

For each chemical family with ≥50 samples:
- ✅ PICP@95% ∈ [0.93, 0.97] (allow 1% more slack for smaller subsets)

---

## Consequences

### Positive

- ✅ **Trust**: Guaranteed coverage → safe for autonomous decisions
- ✅ **Regulatory**: Calibrated UQ meets FDA/EPA requirements for AI-driven experiments
- ✅ **Adaptive**: Can recalibrate monthly as new data arrives
- ✅ **Model-agnostic**: Works with any base predictor (RF, NN, GP)

### Negative

- ⚠️ **Wider intervals**: Calibrated PIs are 10-30% wider than uncalibrated ensemble variance
- ⚠️ **Calibration set cost**: Requires withholding 20-30% of train data
- ⚠️ **Exchangeability assumption**: Assumes test set is drawn from same distribution as calibration (OOD detection critical)

### Mitigation Strategies

1. **Conditional conformal**: Use local calibration (k-NN or kernel weighting) to tighten intervals
2. **Online recalibration**: Update q every N new observations
3. **OOD detection**: Flag high-nonconformity points before deployment
4. **Ensemble conformal**: Combine multiple base models for tighter intervals

---

## Related ADRs

- ADR-001: Composition-First Features (affects calibration set size)
- ADR-003: Active Learning Strategy (uses calibrated UQ for acquisition)

---

## References

- Vovk, V., et al. (2005). *Algorithmic Learning in a Random World*. Springer.
- Shafer, G., & Vovk, V. (2008). *A Tutorial on Conformal Prediction*. JMLR, 9, 371-421.
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
- Feldman, S., et al. (2021). *Achieving Compliance with Materials Safety Standards via Conformal Prediction*. NeurIPS Workshop on ML for Materials.

---

**Supersedes**: None  
**Superseded by**: None  
**Status**: Active

