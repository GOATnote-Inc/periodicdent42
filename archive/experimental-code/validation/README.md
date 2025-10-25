# Validation Studies

Rigorous scientific validation of claims with honest reporting.

## Experiment Selection Validation

**File**: `validate_selection_strategy.py`

### Purpose

Rigorously test the claim: *"Shannon entropy selection reduces experiments needed by 10x"*

### Methodology

**Controlled Active Learning Benchmark**:

1. **Dataset**: UCI Superconductor Database (21,263 samples)
2. **Split**:
   - Initial training: 100 samples
   - Candidate pool: 20,163 samples  
   - Test set: 1,000 samples (held out)

3. **Strategies Compared**:
   - Shannon entropy selection (our method)
   - Random selection (baseline)
   - Uncertainty sampling (standard active learning)
   - Diversity sampling (coverage)

4. **Metrics**:
   - Model RMSE on held-out test set
   - Information gain (Shannon entropy)
   - Reduction factor: How many random experiments = N entropy experiments?

### Running the Validation

```bash
# Full validation (100 iterations, ~10 minutes)
python experiments/validate_selection_strategy.py \
  --dataset data/superconductors/raw/train.csv \
  --iterations 100 \
  --batch-size 10 \
  --output experiments/validation_results

# Quick test (20 iterations, ~2 minutes)
python experiments/validate_selection_strategy.py \
  --dataset data/superconductors/raw/train.csv \
  --iterations 20 \
  --batch-size 10 \
  --output experiments/validation_quick
```

### Output

- `VALIDATION_REPORT.md` - Comprehensive report with honest assessment
- `validation_results.png` - Publication-quality plots (4 panels)
- `validation_data.json` - Raw results (for further analysis)
- `validation_results.pkl` - Python pickle (full results)

### Expected Results

Based on preliminary testing:

- **Random baseline**: 1.0x (by definition)
- **Uncertainty sampling**: 2-3x reduction
- **Diversity sampling**: 2-3x reduction  
- **Shannon entropy**: 4-6x reduction (estimated)

**Honest Assessment**: We expect 4-6x reduction, not 10x. If results show less, we report that honestly.

### Why Honest Reporting Matters

1. **Trust**: Hiring managers value scientific integrity
2. **Credibility**: Overstating results damages reputation
3. **Value**: Even 3-5x reduction is significant for expensive experiments
4. **Transparency**: Shows you can assess your own work critically

### Key Features

**Statistical Rigor**:
- Controlled splits (fixed random seed)
- Multiple strategies compared
- Held-out test set
- Multiple evaluation metrics

**Publication Quality**:
- 4-panel figure (RMSE, information gain, reduction factors, R²)
- Clear methodology section
- Honest interpretation
- Recommendations based on results

**Reproducibility**:
- Fixed random seed
- Saved intermediate results
- Documented parameters
- Raw data available for inspection

---

## Philosophy

> "Scientists don't trust perfection. They trust honesty."

If validation shows 5x reduction instead of 10x:
- ❌ Don't hide it
- ❌ Don't cherry-pick metrics
- ✅ Report it honestly
- ✅ Explain what works and what doesn't
- ✅ Show you can assess your work critically

**This is what separates junior from senior engineers.**

