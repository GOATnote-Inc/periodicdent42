# GO/NO-GO Policy: Autonomous Deployment Decisions

**For**: Robotic Lab Engineers, Safety Officers, Materials Scientists  
**Version**: 2.0  
**Last Updated**: January 2025  
**Criticality**: SAFETY-CRITICAL

---

## Purpose

This document defines **decision criteria** for autonomous lab deployment of materials predictions. The GO/NO-GO policy ensures:
- ✅ **Safety**: Prevents synthesis of unsafe or unreliable candidates
- ✅ **Efficiency**: Avoids wasting resources on low-confidence predictions
- ✅ **Transparency**: Clear, auditable decision rules

---

## Decision Framework

### Three-Level Decision System

```
┌──────────────────────────────────────────────────────────────┐
│  INPUT: Prediction + Uncertainty                             │
│  • y_pred: Predicted T_c (e.g., 85K)                         │
│  • y_std: Epistemic uncertainty (e.g., 5K)                   │
│  • [y_lower, y_upper]: 95% prediction interval (e.g., [75K, 95K]) │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  Check against thresholds      │
        │  • T_min: Minimum acceptable   │
        │  • T_max: Maximum acceptable   │
        └────────────────┬───────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────┐     ┌─────────┐     ┌──────────┐
    │   GO   │     │  MAYBE  │     │  NO-GO   │
    │ (1)    │     │   (0)   │     │  (-1)    │
    └────┬───┘     └────┬────┘     └────┬─────┘
         │              │               │
         ▼              ▼               ▼
  Deploy        Query for         Skip
  immediately   more info      synthesis
```

---

## Decision Rules

### GO (Decision = 1)

**Criteria**: Prediction interval **entirely within** acceptable range

**Mathematical Condition**:
```
y_lower >= T_min  AND  y_upper <= T_max
```

**Interpretation**:
- High confidence that material meets requirements
- 95% probability that true T_c is within [T_min, T_max]
- Safe to deploy to robotic lab

**Action**: **Proceed with synthesis immediately**

**Example**:
```
T_min = 77K (LN2 temperature)
T_max = 150K (reasonable superconductor range)

Prediction: y_pred = 100K
Interval: [85K, 115K]

Check: 85K >= 77K ✅  AND  115K <= 150K ✅
Decision: GO → Synthesize material
```

---

### MAYBE (Decision = 0)

**Criteria**: Prediction interval **overlaps** thresholds

**Mathematical Condition**:
```
(y_lower < T_min AND y_upper >= T_min)  OR
(y_lower <= T_max AND y_upper > T_max)
```

**Interpretation**:
- Uncertain whether material meets requirements
- Need more information to make confident decision
- Uncertainty is too high for autonomous deployment

**Action**: **Query for more information**

Options for resolving MAYBE:
1. **Run additional simulations** (DFT, MD)
2. **Acquire more training data** (active learning)
3. **Consult domain expert** (human-in-the-loop)
4. **Synthesize small test batch** (validation experiment)

**Example**:
```
T_min = 77K
T_max = 150K

Prediction: y_pred = 80K
Interval: [65K, 95K]

Check: 65K < 77K ✅  AND  95K >= 77K ✅
Decision: MAYBE → Query for more data or run DFT calculation
```

---

### NO-GO (Decision = -1)

**Criteria**: Prediction interval **entirely outside** acceptable range

**Mathematical Condition**:
```
y_upper < T_min  OR  y_lower > T_max
```

**Interpretation**:
- High confidence that material does NOT meet requirements
- >95% probability that true T_c is outside [T_min, T_max]
- Synthesis would waste resources

**Action**: **Skip synthesis entirely**

**Example**:
```
T_min = 77K
T_max = 150K

Prediction: y_pred = 45K
Interval: [30K, 60K]

Check: 60K < 77K ✅
Decision: NO-GO → Do not synthesize (T_c too low)
```

---

## Use Case: Superconductors for LN2 Applications

### Scenario
Search for superconductors operable with liquid nitrogen cooling (T_c > 77K).

### Threshold Settings
```python
T_min = 77.0  # LN2 boiling point
T_max = np.inf  # No upper limit (higher is better)
```

### Example Decisions

| Material | T_c Pred | Interval | Decision | Rationale |
|----------|----------|----------|----------|-----------|
| YBa2Cu3O7 | 92K | [85K, 99K] | **GO** | Entirely above 77K |
| MgB2 | 75K | [68K, 82K] | **MAYBE** | Interval overlaps 77K |
| Nb3Sn | 18K | [15K, 21K] | **NO-GO** | Entirely below 77K |
| HgBa2Ca2Cu3O8+δ | 135K | [120K, 150K] | **GO** | Entirely above 77K |
| La2CuO4 | 40K | [35K, 45K] | **NO-GO** | Entirely below 77K |

### Implementation

```python
from src.active_learning.loop import go_no_go_gate

# Get predictions
y_pred, y_lower, y_upper = model.predict_with_uncertainty(X_candidates)
y_std = model.get_epistemic_uncertainty(X_candidates)

# Apply GO/NO-GO gate
decisions = go_no_go_gate(
    y_pred=y_pred,
    y_std=y_std,
    y_lower=y_lower,
    y_upper=y_upper,
    threshold_min=77.0,  # LN2 temperature
    threshold_max=np.inf,
)

# Filter by decision
go_samples = X_candidates[decisions == 1]
maybe_samples = X_candidates[decisions == 0]
no_go_samples = X_candidates[decisions == -1]

print(f"GO: {len(go_samples)} materials → Synthesize immediately")
print(f"MAYBE: {len(maybe_samples)} materials → Query for more info")
print(f"NO-GO: {len(no_go_samples)} materials → Skip synthesis")
```

---

## Safety Considerations

### 1. Always Use Calibrated Models

**Requirement**: PICP ∈ [0.94, 0.96] and ECE ≤ 0.05

**Rationale**: Miscalibrated models produce unreliable intervals → unsafe decisions

**Verification**:
```python
from src.uncertainty.calibration_metrics import (
    prediction_interval_coverage_probability,
    expected_calibration_error,
)

picp = prediction_interval_coverage_probability(y_val, y_val_lower, y_val_upper)
ece = expected_calibration_error(y_val, y_val_pred, y_val_std)

if picp < 0.94 or picp > 0.96:
    raise ValueError(f"Model not calibrated: PICP={picp:.3f} (target: 0.94-0.96)")
if ece > 0.05:
    raise ValueError(f"Model not calibrated: ECE={ece:.3f} (target: ≤0.05)")
```

### 2. Always Filter OOD Before GO/NO-GO

**Requirement**: OOD detector must flag out-of-distribution samples

**Rationale**: Model predictions are unreliable outside training distribution

**Implementation**:
```python
from src.guards import create_ood_detector

# 1. Create OOD detector
ood_detector = create_ood_detector("mahalanobis", alpha=0.01)
ood_detector.fit(X_train)

# 2. Detect OOD
is_ood = ood_detector.predict(X_candidates)

# 3. Filter OOD samples BEFORE GO/NO-GO
X_filtered = X_candidates[~is_ood]

# 4. Apply GO/NO-GO only to in-distribution samples
y_pred, y_lower, y_upper = model.predict_with_uncertainty(X_filtered)
decisions = go_no_go_gate(y_pred, y_std, y_lower, y_upper, threshold_min=77.0)
```

### 3. Log All Decisions for Audit

**Requirement**: All GO/NO-GO decisions must be logged with timestamps

**Rationale**: Enables post-hoc analysis and regulatory compliance

**Implementation**:
```python
import json
from datetime import datetime

decision_log = {
    "timestamp": datetime.now().isoformat(),
    "model": "RandomForestQRF",
    "threshold_min": 77.0,
    "threshold_max": np.inf,
    "n_candidates": len(X_candidates),
    "n_go": int((decisions == 1).sum()),
    "n_maybe": int((decisions == 0).sum()),
    "n_no_go": int((decisions == -1).sum()),
    "go_samples": go_samples.tolist(),  # For reproducibility
}

with open("artifacts/decision_log.json", "w") as f:
    json.dump(decision_log, f, indent=2)
```

---

## Threshold Selection Guidelines

### Choosing T_min

**Question**: What is the **minimum acceptable** T_c?

**Examples**:
- **LN2 cooling (77K)**: T_min = 77K
- **Room temperature (300K)**: T_min = 300K (aspirational)
- **Liquid helium (4K)**: T_min = 0K (any superconductor)

**Considerations**:
- Cooling cost (LN2 cheap, LHe expensive)
- Application requirements (power transmission, MRI, quantum computing)
- Safety margins (set T_min 5-10K above critical threshold)

### Choosing T_max

**Question**: What is the **maximum acceptable** T_c?

**Examples**:
- **No upper limit**: T_max = np.inf (higher is always better)
- **Stability concerns**: T_max = 200K (avoid metastable phases)
- **Specific application**: T_max = 100K (design constraint)

**Considerations**:
- Material stability at high T_c
- Synthesis difficulty (high T_c often harder to synthesize)
- Cost-benefit analysis (diminishing returns above certain T_c)

---

## Integration with Active Learning

### Scenario: Budget-Constrained Exploration

**Problem**: Limited budget (100 labels), want to maximize discoveries

**Strategy**:
1. **Phase 1**: Use acquisition function (UCB, EI) to explore promising regions
2. **Phase 2**: Apply GO/NO-GO gate to filter candidates
3. **Phase 3**: Synthesize only GO samples, query MAYBE samples

**Implementation**:
```python
from src.pipelines import ActiveLearningPipeline

al_pipeline = ActiveLearningPipeline(
    base_model=model,
    acquisition_method="ucb",
    acquisition_kwargs={"kappa": 2.0, "maximize": True},
    diversity_method="greedy",
    budget=100,
    batch_size=10,
)

# Run active learning
results = al_pipeline.run(
    X_labeled=X_labeled,
    y_labeled=y_labeled,
    X_unlabeled=X_unlabeled,
    y_unlabeled=y_unlabeled,
    go_no_go_threshold_min=77.0,
)

# Extract GO samples
go_decisions = results['go_no_go']['decisions']
go_indices = [i for i, d in enumerate(go_decisions) if d == 1]
go_materials = [formulas[i] for i in go_indices]

print(f"Synthesize {len(go_materials)} materials immediately:")
for mat in go_materials:
    print(f"  - {mat}")
```

---

## Validation & Testing

### Unit Tests

```python
from src.active_learning.loop import go_no_go_gate
import numpy as np

# Test 1: GO decision
y_pred = np.array([100.0])
y_std = np.array([5.0])
y_lower = np.array([90.0])
y_upper = np.array([110.0])

decisions = go_no_go_gate(y_pred, y_std, y_lower, y_upper, threshold_min=77.0, threshold_max=np.inf)
assert decisions[0] == 1, "Should be GO (entirely above 77K)"

# Test 2: NO-GO decision
y_pred = np.array([50.0])
y_std = np.array([5.0])
y_lower = np.array([40.0])
y_upper = np.array([60.0])

decisions = go_no_go_gate(y_pred, y_std, y_lower, y_upper, threshold_min=77.0, threshold_max=np.inf)
assert decisions[0] == -1, "Should be NO-GO (entirely below 77K)"

# Test 3: MAYBE decision
y_pred = np.array([77.0])
y_std = np.array([10.0])
y_lower = np.array([67.0])
y_upper = np.array([87.0])

decisions = go_no_go_gate(y_pred, y_std, y_lower, y_upper, threshold_min=77.0, threshold_max=np.inf)
assert decisions[0] == 0, "Should be MAYBE (overlaps 77K)"
```

### Integration Tests

See `tests/test_phase7_pipelines.py::TestGoNoGoGate` for comprehensive tests.

---

## Regulatory Compliance

### Documentation Requirements

For regulatory approval (FDA, EPA, etc.), maintain:
1. **Decision Log**: All GO/NO-GO decisions with timestamps
2. **Model Validation**: Calibration metrics (PICP, ECE) on validation set
3. **OOD Detection**: Percentage of candidates flagged as OOD
4. **Safety Incidents**: Any false positives/negatives
5. **Model Updates**: Version control for model changes

### Audit Trail

Every deployment must include:
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "model_version": "RandomForestQRF_v2.0",
  "model_checksum": "sha256:abc123...",
  "calibration": {
    "picp": 0.951,
    "ece": 0.032
  },
  "ood_detection": {
    "n_ood_filtered": 15,
    "ood_rate": 0.075
  },
  "decisions": {
    "n_go": 42,
    "n_maybe": 18,
    "n_no_go": 125
  },
  "synthesized_materials": ["YBa2Cu3O7", "HgBa2Ca2Cu3O8+δ"]
}
```

---

## Summary

### Key Takeaways

✅ **Always use three-level decision system** (GO/MAYBE/NO-GO)  
✅ **Verify model calibration** (PICP, ECE) before deployment  
✅ **Filter OOD samples** before applying GO/NO-GO gates  
✅ **Log all decisions** for audit trail  
✅ **Set conservative thresholds** (safety margins)  

### Decision Checklist

Before deploying predictions:
- [ ] Model is calibrated (PICP ∈ [0.94, 0.96], ECE ≤ 0.05)
- [ ] OOD detector is trained and validated
- [ ] Thresholds (T_min, T_max) are appropriate for application
- [ ] Decision logging is enabled
- [ ] Human-in-the-loop review process for MAYBE samples
- [ ] Emergency stop mechanism is available

---

## References

- **Conformal Prediction**: Lei et al. (2018) "Distribution-Free Predictive Inference"
- **Active Learning**: Settles (2009) "Active Learning Literature Survey"
- **OOD Detection**: Hendrycks & Gimpel (2016) "A Baseline for Detecting Misclassified and OOD Examples"

---

**Last Updated**: January 2025  
**Version**: 2.0  
**Criticality**: SAFETY-CRITICAL

