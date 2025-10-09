# Autonomous Materials Baseline v2.0 - Technical Overview

## Executive Summary

This project provides a **deployment-ready baseline** for superconductor critical temperature (Tc) prediction designed specifically for **autonomous robotic laboratories**. Unlike traditional ML benchmarks, this baseline prioritizes **calibrated uncertainty**, **physics grounding**, and **risk-aware active learning** to enable safe, budget-efficient materials discovery.

**Deployment Status**: Phase 1 Complete (Leakage-Safe Data Foundation)

---

## Problem Statement

### Challenge

Autonomous laboratories require ML models that can:

1. **Quantify prediction uncertainty** reliably (not just point predictions)
2. **Avoid data leakage** that inflates reported performance
3. **Detect out-of-distribution inputs** to prevent unsafe recommendations
4. **Optimize limited experimental budgets** via active learning
5. **Provide physics-interpretable explanations** for regulatory compliance

Traditional ML benchmarks fail on 3-5 of these requirements.

### Solution

We build a **composition-only Tc predictor** with:

- ✅ **Conformal prediction** for calibrated 95% confidence intervals
- ✅ **Family-wise data splitting** with near-duplicate detection
- ✅ **Multi-modal OOD detection** (Mahalanobis + KDE + conformal risk)
- ✅ **Diversity-aware active learning** with cost/risk gates
- ✅ **Physics-grounded features** mapped to BCS theory

---

## Design Principles

### 1. Honest Uncertainty

**Problem**: Most ML models underestimate uncertainty, leading to overconfident decisions.

**Solution**:
- **Epistemic uncertainty** via ensembles, MC dropout, or GP
- **Aleatoric uncertainty** via parametric models (NGBoost)
- **Conformal prediction** for distribution-free coverage guarantees
- **Calibration metrics** (ECE, PICP) as deployment gates

**Success Criterion**: PICP@95% ∈ [0.94, 0.96] AND ECE ≤ 0.05

---

### 2. Leakage Prevention

**Problem**: Test set contamination inflates metrics by 10-50% in materials datasets.

**Solution**:
- **Formula-level splitting** (no shared formulas across train/test)
- **Family-level awareness** (element set tracking)
- **Near-duplicate detection** (cosine similarity > 0.995 fails CI)
- **Automated checks** in test suite

**Success Criterion**: Zero formula overlap, zero near-duplicates (cosine > 0.995)

---

### 3. OOD Robustness

**Problem**: Models extrapolate wildly on out-of-distribution chemistry.

**Solution**:
- **Mahalanobis distance** in feature space (95th percentile threshold)
- **KDE density scoring** (5th percentile flagged as OOD)
- **Conformal nonconformity** (risk-controlled predictions)
- **No-Go gates** in active learning controller

**Success Criterion**: ≥90% synthetic OOD detection at ≤10% FPR

---

### 4. Budget-Efficient Active Learning

**Problem**: Random exploration wastes 30-50% of experimental budget.

**Solution**:
- **Acquisition functions** (UCB, EI, MaxVar, EIG-proxy)
- **Diversity selection** (k-Medoids, DPP) to avoid mode collapse
- **Cost-aware scoring** (synthesis difficulty penalty)
- **Risk gates** (reject high-variance candidates)

**Success Criterion**: ≥30% RMSE reduction in ≤20 acquisitions vs random

---

### 5. Physics Interpretability

**Problem**: Black-box predictions fail regulatory review in materials.

**Solution**:
- **BCS-grounded features** (atomic mass → Debye frequency → Tc)
- **SHAP values** for per-prediction explanations
- **PDP/ICE curves** for global trend validation
- **Sanity checks** on isotope effect, EN spread, valence correlation

**Success Criterion**: Top 5 features align with BCS intuition

---

## System Architecture

### Data Flow

```
Raw Data (CSV)
    ↓
Feature Engineering (matminer/fallback)
    ↓
Leakage-Safe Splitting (family-wise, near-dup check)
    ↓
Data Contracts (schema + checksums)
    ↓
Model Training (RF/MLP/NGBoost + uncertainty)
    ↓
Calibration (conformal prediction)
    ↓
OOD Detection (Mahalanobis + KDE + conformal)
    ↓
Active Learning Loop (acquisition + diversity + risk gates)
    ↓
Evidence Pack (metrics, plots, manifests, model cards)
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Data Splits** | Leakage-safe train/val/test | Stratified + family blocking |
| **Contracts** | Schema validation + checksums | Pydantic + SHA-256 |
| **Features** | Composition → ML-ready features | matminer (Magpie) or lightweight |
| **Models** | Uncertainty-aware predictors | RF+QRF, MLP+MC, NGBoost, GP |
| **Calibration** | 95% PI coverage | Split/Mondrian conformal |
| **OOD** | Novelty detection | Mahalanobis + KDE + conformal |
| **AL Controller** | Budget/risk/cost gates | UCB/EI + k-Medoids/DPP |
| **Reporting** | Audit-ready evidence | JSON metrics + PNG plots + manifests |

---

## Uncertainty Models

### RF + Quantile Regression Forest (QRF)

**Epistemic Uncertainty**: Bootstrap aggregation variance  
**Aleatoric Uncertainty**: Quantile intervals (2.5%, 97.5%)

**Pros**: Fast, robust, no hyperparameter tuning  
**Cons**: Underestimates tail uncertainty

**Use Case**: First-pass baseline, production deployment (low latency)

---

### MLP + MC Dropout

**Epistemic Uncertainty**: Monte Carlo sampling (T=50) with dropout (p=0.2)  
**Aleatoric Uncertainty**: Optional heteroscedastic output layer

**Pros**: Expressive, scales to large data  
**Cons**: Requires tuning, slower inference (T forward passes)

**Use Case**: High-accuracy regime, complex feature interactions

---

### NGBoost

**Epistemic Uncertainty**: None (single model)  
**Aleatoric Uncertainty**: Parametric Normal(μ, σ) distribution

**Pros**: Best for heteroscedastic noise, fast training  
**Cons**: No epistemic uncertainty (combine with ensemble for both)

**Use Case**: Noisy experimental data, known aleatoric variance

---

### Gaussian Process Regression (GP)

**Epistemic Uncertainty**: Full posterior covariance  
**Aleatoric Uncertainty**: Noise parameter σ_n

**Pros**: Gold-standard uncertainty, automatic relevance determination (ARD)  
**Cons**: O(N³) scaling, requires dimensionality reduction (PCA) for >1000 samples

**Use Case**: Small datasets (<500 samples), high-stakes decisions

---

## Active Learning Strategy

### Workflow

```
1. Initialize with labeled seed (N=50, stratified by family)
2. Train model on labeled set
3. FOR each round (budget limit):
   a. Score unlabeled pool with acquisition function
   b. Apply diversity selection (k-Medoids/DPP)
   c. Apply risk gates (OOD, σ² cap, cost)
   d. Select top-K candidates
   e. Query oracle (robotic synthesis + measurement)
   f. Add to labeled set
   g. Retrain model
4. Generate evidence pack
```

### Acquisition Functions

| Function | Formula | Best For |
|----------|---------|----------|
| **UCB** | μ + β·σ | Balanced exploration/exploitation |
| **EI** | E[max(0, y_best - y)] | Optimization (maximize Tc) |
| **MaxVar** | σ² | Pure exploration (fill gaps) |
| **EIG-Proxy** | H(y\|D) - H(y\|D∪x) | Information maximization |

**Default**: UCB with β=2.0 (well-calibrated for most tasks)

---

### Diversity Selection

**Problem**: Acquisition functions can select chemically similar candidates (mode collapse).

**Solution**: After scoring, apply diversity filter:

1. **k-Medoids (PAM)**: Greedy selection of K representative candidates in feature space
2. **DPP**: Probabilistic sampling favoring diverse, high-scoring sets

**Objective**: Maximize `acquisition_score - λ·diversity_penalty`

**Default**: k-Medoids with λ=0.3

---

### Risk Gates

Before querying, check:

1. ✅ **Budget**: Total queries < limit
2. ✅ **OOD**: Candidate not flagged by Mahalanobis/KDE/conformal
3. ✅ **Uncertainty**: σ² < risk threshold
4. ✅ **Cost**: synthesis_cost < budget (if cost model available)

**No-Go** if ANY gate fails.

---

## Calibration & Conformal Prediction

### Split Conformal

1. Fit model on train set
2. Compute nonconformity scores on calibration set: `s_i = |y_i - ŷ_i|`
3. Find (1-α)-quantile of scores: `q = quantile(s, 0.95)`
4. Prediction interval: `[ŷ - q, ŷ + q]`

**Guarantee**: ≥95% coverage on test set (distribution-free)

---

### Mondrian Conformal

Same as split conformal, but **per chemical family**:

1. Partition calibration set by family
2. Compute family-specific quantiles: `q_family`
3. Prediction interval: `[ŷ - q_family, ŷ + q_family]`

**Advantage**: Localized coverage for heterogeneous chemistry

---

### Calibration Metrics

- **PICP (Prediction Interval Coverage Probability)**: Fraction of test targets inside PI
  - Target: 0.95 ± 0.01 for 95% PI
- **ECE (Expected Calibration Error)**: Binned calibration curve error
  - Target: ≤ 0.05
- **PI Width**: Median and 90th percentile interval width
  - Minimize subject to PICP ≥ 0.94

---

## OOD Detection

### Mahalanobis Distance

```
d(x) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
```

Fit μ, Σ on train features. Flag if `d(x) > τ` (95th percentile).

**Pros**: Fast, handles correlated features  
**Cons**: Assumes Gaussian, fails on multimodal data

---

### KDE Density

Fit kernel density estimator on train features. Flag if `p(x) < τ` (5th percentile).

**Pros**: Non-parametric, handles multimodal  
**Cons**: Curse of dimensionality (apply PCA first if D > 20)

---

### Conformal Nonconformity

Use conformal score `s = |y - ŷ|` as OOD proxy. Flag if `s > τ_high`.

**Pros**: Prediction-aware (not just feature-based)  
**Cons**: Requires calibration set

---

### Ensemble OOD Decision

```
OOD = (mahalanobis > τ_M) OR (kde_density < τ_K) OR (conformal > τ_C)
```

**Tuning**: Set thresholds to achieve target FPR (e.g., 10%) on validation set.

---

## Evidence Pack

Each run generates a **reproducible evidence pack**:

### Metrics (`metrics.json`)

```json
{
  "model": "rf_qrf",
  "seed": 42,
  "rmse_test": 12.34,
  "mae_test": 9.87,
  "r2_test": 0.85,
  "ece": 0.04,
  "picp_95": 0.95,
  "pi_width_median": 15.2,
  "al_rmse_reduction_pct": 34.2,
  "al_info_gain_bits_per_query": 2.1,
  "ood_tpr_90": 0.92,
  "ood_fpr": 0.08
}
```

### Plots

- `calibration_reliability.png` - Reliability diagram (predicted prob vs observed)
- `calibration_pi_coverage.png` - PI coverage vs width tradeoff
- `al_curves_rmse.png` - RMSE over AL rounds (vs random baseline)
- `al_info_gain.png` - Information gain per round
- `al_diversity.png` - Chemical space coverage over rounds
- `ood_scatter.png` - Mahalanobis vs KDE density with thresholds
- `importances.png` - Feature importances (Gini or permutation)
- `shap_summary.png` - SHAP values for top 10 features
- `pdp_top5.png` - PDP curves for top 5 features

### Manifests

- `manifest.json` - SHA-256 checksums of all artifacts
- `model_card.md` - Model details, hyperparameters, limitations, thresholds
- `GO_NO_GO_POLICY.md` - Deployment decision rules with filled thresholds

---

## Success Metrics

### Calibration (Gate 1)

- ✅ PICP@95% ∈ [0.94, 0.96]
- ✅ ECE ≤ 0.05
- ✅ PI width < 20 K (median)

### Active Learning (Gate 2)

- ✅ RMSE reduction ≥30% after 20 acquisitions (vs random)
- ✅ Information gain ≥1.5 bits/query (average)
- ✅ Chemical space coverage ≥80% unique families explored

### OOD Detection (Gate 3)

- ✅ TPR ≥90% on synthetic OOD probes
- ✅ FPR ≤10% on validation set

### Leakage Prevention (Gate 4)

- ✅ Zero formula overlap across train/val/test
- ✅ Zero near-duplicates (cosine > 0.995)

### Reproducibility (Gate 5)

- ✅ Fixed seed = bit-identical results
- ✅ Evidence manifest checksums match
- ✅ All tests pass in CI

---

## Limitations & Future Work

### Current Limitations

1. **Composition-only features**: Ignores crystal structure, synthesis conditions
2. **BCS superconductors only**: Does not cover high-Tc cuprates, iron-based, etc.
3. **Single-objective**: Optimizes Tc only (not stability, cost, scalability)
4. **Static dataset**: No online learning or concept drift handling
5. **No multi-fidelity**: Treats all measurements equally (no uncertainty on y)

### Planned Improvements

- [ ] **Structure-aware features**: Integrate crystal graphs (e.g., MEGNet, CGCNN)
- [ ] **Multi-task learning**: Joint prediction of Tc, gap, stability
- [ ] **Cost models**: Explicit synthesis difficulty scoring
- [ ] **Online AL**: Batch updates without full retraining
- [ ] **Multi-fidelity**: Combine DFT, simulation, and experimental data

---

## Contact & Contribution

**Maintainer**: GOATnote Autonomous Research Lab Initiative  
**Email**: b@thegoatnote.com

See `CONTRIBUTING.md` for development guidelines.

---

**Last Updated**: 2024-10-09  
**Status**: Phase 1 Complete (Data Foundation)  
**Deployment Ready**: False (Pending Phases 2-7)

