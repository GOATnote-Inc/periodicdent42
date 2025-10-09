# RUNBOOK: Step-by-Step Usage Guide

**For**: Materials Scientists, ML Engineers, Lab Automation  
**Version**: 2.0  
**Last Updated**: January 2025

---

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training Workflow](#training-workflow)
4. [Active Learning Workflow](#active-learning-workflow)
5. [Evaluation & Interpretation](#evaluation--interpretation)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.10+ (tested on 3.13)
- 4GB RAM minimum
- macOS, Linux, or Windows

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/autonomous-baseline.git
cd autonomous-baseline
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Core dependencies
pip install -e .

# Optional: Matminer for advanced features
pip install -e .[materials]

# Development dependencies (if contributing)
pip install -e .[dev]
```

### Step 4: Verify Installation
```bash
# Run tests
pytest tests/ -v

# Expected: 182/182 tests passing, 81% coverage
```

---

## Data Preparation

### Input Format

Your dataset must be a CSV or Pandas DataFrame with:
- **Formula column**: Chemical formulas (e.g., "YBa2Cu3O7", "MgB2")
- **Target column**: T_c values in Kelvin (e.g., 92.0, 39.0)

**Example CSV**:
```csv
formula,Tc
YBa2Cu3O7,92.0
MgB2,39.0
La2CuO4,38.0
Hg0.8Tl0.2Ba2Ca2Cu3O8+δ,138.0
```

### Data Quality Checks

```python
import pandas as pd

# Load data
data = pd.read_csv("superconductor_data.csv")

# Check for missing values
print(data.isnull().sum())

# Check formula validity
from src.features.composition import CompositionFeaturizer

featurizer = CompositionFeaturizer()
for formula in data['formula']:
    try:
        featurizer.featurize_dataframe(pd.DataFrame({'formula': [formula]}), 'formula')
    except Exception as e:
        print(f"Invalid formula: {formula} - {e}")

# Check T_c distribution
print(data['Tc'].describe())
```

### Data Splitting Strategy

The pipeline automatically handles:
- **Family-wise splitting**: Prevents element overlap across splits
- **Stratified sampling**: Balanced T_c distribution
- **Near-duplicate detection**: Cosine similarity < 0.99

**Manual control** (optional):
```python
from src.data.splits import LeakageSafeSplitter

splitter = LeakageSafeSplitter(
    test_size=0.2,
    val_size=0.1,
    near_dup_threshold=0.99,
    enforce_near_dup_check=True,  # Strict for real data
    random_state=42,
)

splits = splitter.split(data, formula_col='formula', target_col='Tc')
```

---

## Training Workflow

### Quickstart: Basic Training

```python
from src.pipelines import TrainingPipeline
from src.models import RandomForestQRF
import pandas as pd

# 1. Load data
data = pd.read_csv("superconductor_data.csv")

# 2. Create pipeline
pipeline = TrainingPipeline(
    random_state=42,
    test_size=0.2,
    val_size=0.1,
    artifacts_dir="artifacts/experiment_001",
)

# 3. Train model
model = RandomForestQRF(n_estimators=100, random_state=42)
results = pipeline.run(
    data=data,
    formula_col='formula',
    target_col='Tc',
    model=model,
    conformal_alpha=0.05,  # 95% confidence intervals
)

# 4. View results
print(f"Dataset: {results['dataset_size']} samples")
print(f"Splits: {results['splits']}")
print(f"Features: {results['features']['n_features']}")
print(f"PICP: {results['calibration']['picp']:.3f}")
print(f"ECE: {results['calibration']['ece']:.3f}")
print(f"Artifacts: {results['artifacts_dir']}")
```

### Advanced: Model Comparison

```python
from src.models import RandomForestQRF, MLPMCD, NGBoostAleatoric

models = {
    "RF+QRF": RandomForestQRF(n_estimators=100, random_state=42),
    "MLP+MCD": MLPMCD(n_ensemble=5, random_state=42),
    "NGBoost": NGBoostAleatoric(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    pipeline = TrainingPipeline(
        random_state=42,
        artifacts_dir=f"artifacts/{name.replace('+', '_')}",
    )
    results[name] = pipeline.run(data, model=model)
    
# Compare calibration
for name, result in results.items():
    print(f"{name}: PICP={result['calibration']['picp']:.3f}, ECE={result['calibration']['ece']:.3f}")
```

### Interpreting Results

**PICP (Prediction Interval Coverage Probability)**:
- Target: 94-96% @ 95% confidence
- Too low (<94%): Model underestimates uncertainty → unsafe
- Too high (>96%): Model overestimates uncertainty → inefficient

**ECE (Expected Calibration Error)**:
- Target: ≤0.05
- >0.05: Model is miscalibrated → adjust hyperparameters or use conformal prediction

**Artifacts Generated**:
- `train.csv`, `val.csv`, `test.csv` - Data splits
- `model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `contracts.json` - Dataset contracts
- `MANIFEST.json` - SHA-256 checksums

---

## Active Learning Workflow

### Quickstart: Basic Active Learning

```python
from src.pipelines import ActiveLearningPipeline
from src.models import RandomForestQRF
import numpy as np

# 1. Prepare data
# X_labeled: (N_labeled, D) - Initial labeled features
# y_labeled: (N_labeled,) - Initial labeled targets
# X_unlabeled: (N_unlabeled, D) - Unlabeled pool features
# y_unlabeled: (N_unlabeled,) - True labels (for simulation only)

# 2. Create AL pipeline
model = RandomForestQRF(n_estimators=50, random_state=42)

al_pipeline = ActiveLearningPipeline(
    base_model=model,
    acquisition_method="ucb",
    acquisition_kwargs={"kappa": 2.0, "maximize": True},
    diversity_method="greedy",
    diversity_kwargs={"alpha": 0.5},
    ood_method="mahalanobis",
    budget=100,
    batch_size=10,
    artifacts_dir="artifacts/al_experiment_001",
)

# 3. Run active learning
results = al_pipeline.run(
    X_labeled=X_labeled,
    y_labeled=y_labeled,
    X_unlabeled=X_unlabeled,
    y_unlabeled=y_unlabeled,  # For simulation
    go_no_go_threshold_min=77.0,  # LN2 temperature
)

# 4. View results
print(f"OOD filtered: {results['ood_filtering']['n_ood']} samples")
print(f"Budget used: {results['active_learning']['budget_used']}")
print(f"Iterations: {results['active_learning']['n_iterations']}")
print(f"GO: {results['go_no_go']['n_go']}")
print(f"MAYBE: {results['go_no_go']['n_maybe']}")
print(f"NO-GO: {results['go_no_go']['n_no_go']}")
```

### Advanced: Acquisition Strategy Comparison

```python
strategies = [
    ("UCB", "ucb", {"kappa": 2.0}),
    ("EI", "ei", {"y_best": 100.0, "xi": 0.01}),
    ("MaxVar", "maxvar", {}),
]

for name, method, kwargs in strategies:
    al_pipeline = ActiveLearningPipeline(
        base_model=RandomForestQRF(n_estimators=50, random_state=42),
        acquisition_method=method,
        acquisition_kwargs=kwargs,
        budget=50,
        batch_size=10,
    )
    
    results = al_pipeline.run(X_labeled, y_labeled, X_unlabeled, y_unlabeled)
    print(f"{name}: Budget used={results['active_learning']['budget_used']}")
```

### Diversity-Aware Selection

```python
# k-Medoids: Cluster-based diversity
al_pipeline = ActiveLearningPipeline(
    base_model=model,
    acquisition_method="ucb",
    diversity_method="k_medoids",
    diversity_kwargs={"metric": "euclidean"},
    budget=100,
    batch_size=10,
)

# Greedy: Iterative diversity
al_pipeline = ActiveLearningPipeline(
    base_model=model,
    acquisition_method="ucb",
    diversity_method="greedy",
    diversity_kwargs={"alpha": 0.5},  # 0.5 = balanced
    budget=100,
    batch_size=10,
)

# DPP: Optimal diversity
al_pipeline = ActiveLearningPipeline(
    base_model=model,
    acquisition_method="ucb",
    diversity_method="dpp",
    diversity_kwargs={"lambda_diversity": 1.0},
    budget=100,
    batch_size=10,
)
```

---

## Evaluation & Interpretation

### Calibration Metrics

```python
from src.uncertainty.calibration_metrics import (
    prediction_interval_coverage_probability,
    expected_calibration_error,
)

# Get predictions with uncertainty
y_pred, y_lower, y_upper = model.predict_with_uncertainty(X_test)
y_std = model.get_epistemic_uncertainty(X_test)

# Compute metrics
picp = prediction_interval_coverage_probability(y_test, y_lower, y_upper)
ece = expected_calibration_error(y_test, y_pred, y_std)

print(f"PICP: {picp:.3f} (target: 0.94-0.96)")
print(f"ECE: {ece:.3f} (target: ≤0.05)")
```

### OOD Detection Evaluation

```python
from src.guards import create_ood_detector

# Create OOD detector
detector = create_ood_detector("mahalanobis", alpha=0.01)
detector.fit(X_train)

# Detect OOD samples
is_ood = detector.predict(X_test)
ood_proba = detector.predict_proba(X_test)

print(f"OOD samples: {is_ood.sum()} / {len(X_test)}")
print(f"OOD rate: {is_ood.mean():.2%}")
```

### GO/NO-GO Gate Analysis

```python
from src.active_learning.loop import go_no_go_gate

# Apply gate
decisions = go_no_go_gate(
    y_pred, y_std, y_lower, y_upper,
    threshold_min=77.0,  # LN2 temperature
    threshold_max=np.inf,
)

# Analyze decisions
n_go = (decisions == 1).sum()
n_maybe = (decisions == 0).sum()
n_no_go = (decisions == -1).sum()

print(f"GO: {n_go} ({n_go / len(decisions):.1%})")
print(f"MAYBE: {n_maybe} ({n_maybe / len(decisions):.1%})")
print(f"NO-GO: {n_no_go} ({n_no_go / len(decisions):.1%})")
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**: Install package in editable mode
```bash
pip install -e .
```

### Issue: "matminer not available"

**Solution**: Install optional materials dependencies
```bash
pip install -e .[materials]
```

Or use lightweight featurizer (automatically falls back).

### Issue: "Leakage detected: Near-duplicates across splits"

**Solution**: Adjust threshold or disable strict check
```python
pipeline = TrainingPipeline(
    near_dup_threshold=0.995,  # More lenient
    enforce_near_dup_check=False,  # For synthetic data
)
```

### Issue: "PICP too low (<0.94)"

**Causes**:
- Model underestimates uncertainty
- Conformal prediction not applied

**Solutions**:
1. Use conformal prediction (automatic in pipeline)
2. Increase quantile width for QRF
3. Increase MC-Dropout samples for MLP

### Issue: "Active learning not improving performance"

**Causes**:
- Acquisition function not suited to problem
- No diversity in batch selection
- Budget too small

**Solutions**:
1. Try different acquisition functions (UCB, EI, MaxVar)
2. Add diversity selector (k-Medoids, Greedy, DPP)
3. Increase budget or batch size

### Issue: "OOD detector flagging too many samples"

**Causes**:
- Threshold too conservative
- Training data not representative

**Solutions**:
1. Adjust alpha (e.g., 0.05 → 0.1)
2. Use different detector (Mahalanobis → KDE)
3. Retrain on more diverse training set

---

## Best Practices

### 1. Always Use Leakage-Safe Splitting
```python
# ✅ Good: Family-wise splitting
splitter = LeakageSafeSplitter(test_size=0.2)
splits = splitter.split(data, formula_col='formula')

# ❌ Bad: Random splitting
train, test = train_test_split(data, test_size=0.2)  # LEAKAGE RISK
```

### 2. Verify Calibration Before Deployment
```python
# Always check PICP and ECE
picp = prediction_interval_coverage_probability(y_test, y_lower, y_upper)
ece = expected_calibration_error(y_test, y_pred, y_std)

if picp < 0.94 or ece > 0.05:
    print("⚠️  Model not well-calibrated - use conformal prediction")
```

### 3. Use OOD Detection in Active Learning
```python
# Always filter OOD before querying
al_pipeline = ActiveLearningPipeline(
    ood_method="mahalanobis",  # Or "kde", "conformal"
    # ...
)
```

### 4. Apply GO/NO-GO Gates Before Deployment
```python
# Never deploy without safety check
decisions = go_no_go_gate(y_pred, y_std, y_lower, y_upper, threshold_min=77.0)

if (decisions == -1).any():
    print("⚠️  Some samples flagged NO-GO - do not synthesize")
```

### 5. Generate Evidence Packs for Reproducibility
```python
from src.reporting import create_evidence_pack

create_evidence_pack(
    artifacts_dir=Path("artifacts/experiment_001"),
    pipeline_type="train",
    config={"random_state": 42, "model": "RandomForestQRF"},
)
```

---

## Next Steps

1. **Run your first experiment**: Follow [Training Workflow](#training-workflow)
2. **Evaluate calibration**: Check PICP and ECE
3. **Try active learning**: Follow [Active Learning Workflow](#active-learning-workflow)
4. **Deploy with GO/NO-GO gates**: See [GO_NO_GO_POLICY.md](GO_NO_GO_POLICY.md)

---

## Support

- **Documentation**: `docs/` directory
- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions

---

**Last Updated**: January 2025  
**Version**: 2.0

