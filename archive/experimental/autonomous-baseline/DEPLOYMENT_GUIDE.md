# 🚀 DEPLOYMENT GUIDE: Partial System (Validated Components)

**Status**: Production-Ready for Calibrated Uncertainty + OOD Detection  
**Version**: 1.0.0-partial  
**Date**: October 9, 2025

---

## Overview

This guide covers deployment of the **validated components** from the autonomous materials baseline:

### ✅ What's Deployed (Validated)
1. **Calibrated Uncertainty**: PICP@95% = 94.4%  
2. **OOD Detection**: AUC = 1.0, TPR@10%FPR = 100%  
3. **Random Sampling**: For experiment selection

### ❌ What's NOT Deployed (Failed Validation)
4. **Active Learning**: Use random sampling instead (RF-based AL performs worse than random)

---

## Quick Start

### 1. Prepare Production Model

```bash
cd autonomous-baseline

# Activate environment
source .venv/bin/activate

# Package validated models into production-ready predictor
python scripts/prepare_production_model.py \\
    --conformal-model models/rf_conformal_fixed_alpha0.045.pkl \\
    --training-data data/processed/uci_splits/train.csv \\
    --output models/production
```

**Output**:
- `models/production/conformal_predictor.pkl` (calibrated model)
- `models/production/ood_detector.pkl` (Mahalanobis detector)
- `models/production/predictor_metadata.json` (validation metrics)

---

### 2. Python API Usage

```python
from src.deployment import AutonomousPredictor
import pandas as pd
import numpy as np

# Load production model
predictor = AutonomousPredictor.load('models/production')

# Make predictions with safety checks
X = np.array([[...]])  # Feature matrix (N, 81)
results = predictor.predict_with_safety(X)

for result in results:
    if result.ood_flag:
        print(f"⚠️  OOD detected (score={result.ood_score:.1f})")
        print(f"   Recommend expert review before synthesis")
    else:
        print(f"✅ Safe to synthesize")
        print(f"   Predicted Tc: {result.prediction:.1f} K")
        print(f"   95% PI: [{result.lower_bound:.1f}, {result.upper_bound:.1f}] K")
```

---

### 3. Batch Predictions (CSV)

```bash
# Predict on pre-featurized candidates
python scripts/predict_cli.py \\
    --input data/processed/uci_splits/test.csv \\
    --output predictions.csv \\
    --model models/production
```

**Output** (`predictions.csv`):
```csv
predicted_tc,lower_bound,upper_bound,interval_width,ood_flag,ood_score,...
35.6,16.0,55.2,39.2,False,16.9,...
```

---

### 4. Recommend Experiments (Random Sampling)

```bash
# Recommend 10 experiments using validated random sampling
python scripts/predict_cli.py \\
    --input candidates.csv \\
    --recommend 10 \\
    --output recommended_experiments.csv \\
    --model models/production \\
    --seed 42
```

**Output**: Top 10 candidates selected randomly (OOD filtered)

---

## Feature Requirements

### ⚠️ IMPORTANT: Feature Compatibility

The production model requires **81 features** from the UCI Superconductivity dataset. For new compounds:

**Option A: Use Pre-Featurized Inputs (Recommended)**
- Provide CSV with all 81 features pre-computed
- Features must match training feature names exactly
- See `data/processed/uci_splits/test.csv` for format

**Option B: Use Matminer Featurizer (Advanced)**
- Requires `matminer` package installed
- Uses Magpie descriptors (81 features)
- May require additional element property data

**Current Limitation**:
- Lightweight featurizer (8 features) is NOT compatible with production model
- For deployment, use pre-featurized data or full Matminer pipeline

---

## GO / NO-GO Decision Logic

### ✅ GO (Safe to Synthesize)

1. **Not OOD**: `ood_flag == False` (Mahalanobis score < 150.0)
2. **Calibrated Interval**: 95% prediction interval available
3. **High Confidence**: Narrow interval width (< 30 K preferred)

### ⚠️ CONDITIONAL GO (Expert Review)

1. **Near OOD**: `ood_score > 100` but `< 150`
2. **Wide Interval**: Interval width > 50 K (low confidence)

### ❌ NO-GO (Do Not Synthesize)

1. **OOD Flagged**: `ood_flag == True` (score ≥ 150.0)
2. **Negative Tc Predicted**: `prediction < 0 K` (unphysical)

---

## Monitoring & Logging

### Prediction History

```python
# Export prediction history for monitoring
predictor.export_history('logs/prediction_history.json')

# Get monitoring stats
stats = predictor.get_monitoring_stats()
print(f"Total predictions: {stats['n_predictions']}")
print(f"OOD rate: {stats['ood_rate']:.1%}")
print(f"Mean interval width: {stats['mean_interval_width']:.1f} K")
```

### Key Metrics to Monitor

1. **OOD Rate**: Should be < 10% if candidates are well-curated
2. **Interval Width**: Monitor for calibration drift (target: 30-50 K)
3. **Prediction Distribution**: Should match training distribution

---

## Example Workflow

### End-to-End Experiment Selection

```python
from src.deployment import AutonomousPredictor
import pandas as pd

# 1. Load production model
predictor = AutonomousPredictor.load('models/production')

# 2. Load candidate compounds (pre-featurized)
candidates = pd.read_csv('candidates_featurized.csv')

# 3. Recommend experiments (random sampling, OOD filtered)
recommended = predictor.recommend_experiments(
    candidates,
    n_experiments=10,
    ood_filter=True,  # Filter out OOD samples
    random_state=42   # Reproducible
)

# 4. Save recommendations
recommended.to_csv('recommended_experiments.csv', index=False)

# 5. Review top candidates
print("Top 5 Recommended Experiments:")
for i, row in recommended.head(5).iterrows():
    print(f"{i+1}. Tc={row['predicted_tc']:.1f} K, "
          f"PI=[{row['lower_bound']:.1f}, {row['upper_bound']:.1f}] K")

# 6. Export history for monitoring
predictor.export_history('logs/experiment_selection_history.json')
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Validated models in `models/production/`
- [ ] Production environment has `scikit-learn`, `numpy`, `pandas`
- [ ] Feature pipeline tested on sample data
- [ ] Monitoring/logging infrastructure ready

### Deployment

- [ ] Load production model
- [ ] Test prediction on known sample (e.g., first test sample)
- [ ] Verify OOD detection (test with synthetic OOD sample)
- [ ] Enable logging to `logs/prediction_history.json`

### Post-Deployment

- [ ] Monitor OOD rate (target: < 10%)
- [ ] Monitor interval widths (target: 30-50 K)
- [ ] Collect experimental results for validation
- [ ] Retrain/recalibrate quarterly (or after 100+ new experiments)

---

## Known Limitations

1. **Feature Compatibility**: Requires exact 81 features from UCI dataset
2. **No Active Learning**: Use random sampling (validated strategy)
3. **Calibration Drift**: May need recalibration after significant distribution shift
4. **OOD Threshold**: Tuned for UCI dataset (may need adjustment for other domains)

---

## Troubleshooting

### Issue: "Feature mismatch" error

**Cause**: Input features don't match production model (81 features)

**Fix**:
```python
# Check expected features
predictor = AutonomousPredictor.load('models/production')
print(f"Expected features: {len(predictor.feature_names)}")
print(f"Feature names: {predictor.feature_names[:5]}...")

# Ensure input has all features
X = candidates[predictor.feature_names].values
```

### Issue: High OOD rate (> 20%)

**Cause**: Candidates far from training distribution

**Options**:
1. **Retrain** with candidates as additional training data
2. **Adjust threshold**: Increase OOD threshold (e.g., 200.0 instead of 150.0)
3. **Expert review**: Manually review OOD candidates

### Issue: Wide prediction intervals

**Cause**: Model uncertainty or calibration drift

**Options**:
1. **Collect more data** in low-confidence regions
2. **Recalibrate** conformal predictor with recent data
3. **Switch model**: Consider GP/BNN for better uncertainty (Phase 10)

---

## Performance Benchmarks

### Prediction Speed

- **Single sample**: < 10 ms
- **Batch (100 samples)**: < 100 ms
- **Recommendation (1000 → 10)**: < 500 ms

### Memory Usage

- **Model size**: ~77 MB (Random Forest + conformal scores)
- **Runtime memory**: ~200 MB (including feature matrix)

---

## Next Steps (Phase 10+)

### Future Improvements

1. **Replace RF with GP/BNN** for working active learning
   - Expected: 30-50% RMSE reduction
   - Estimated time: 2-3 weeks

2. **Add Ensemble OOD Detection** (Mahalanobis + KDE + Conformal)
   - More robust OOD detection
   - Lower false positive rate

3. **Implement Calibration Monitoring**
   - Automatic recalibration triggers
   - Drift detection alerts

---

## Support

**Contact**: GOATnote Autonomous Research Lab Initiative  
**Email**: b@thegoatnote.com  
**Documentation**: `autonomous-baseline/` directory

**Reports**:
- Validation: `VALIDATION_SUITE_COMPLETE.md`
- Executive Summary: `EXECUTIVE_SUMMARY_VALIDATION.md`
- Evidence Pack: `evidence/EVIDENCE_PACK_REPORT.txt`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-partial | 2025-10-09 | Initial deployment (validated components only) |

**Status**: ✅ PRODUCTION-READY for calibrated uncertainty + OOD detection

