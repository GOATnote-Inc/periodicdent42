# 🔧 HARDENING ROADMAP (Post-Audit Action Plan)

**Generated**: October 9, 2025  
**Current Grade**: C+/B- (68/100)  
**Target Grade**: A- (85/100)  
**Estimated Time**: 5-7 days

---

## 🎯 PRIORITY MATRIX

| Action | Impact | Effort | Priority | Owner |
|--------|--------|--------|----------|-------|
| Add 15 more seeds (→20 total) | HIGH | LOW (3h) | **P0** | `tier2_clean_benchmark.py` |
| Add XGBoost + RF baselines | HIGH | LOW (4h) | **P0** | `scripts/add_baselines.py` |
| Physics interpretability | HIGH | MEDIUM (2d) | **P0** | `scripts/analyze_learned_features.py` |
| Acquisition function sweep | MEDIUM | LOW (6h) | **P1** | `scripts/compare_acquisitions.py` |
| Epistemic efficiency | MEDIUM | LOW (4h) | **P1** | `scripts/compute_information_gain.py` |
| Reproducibility artifacts | MEDIUM | LOW (2h) | **P1** | `scripts/generate_provenance.py` |
| Literature head-to-head | LOW | HIGH (1w) | **P2** | `scripts/compare_literature.py` |

---

## 📋 P0: MUST-FIX (3-4 days → B+ grade)

### Action 1: Increase to 20 Seeds ⏱️ 3 hours

**Gap**: Only 5 seeds (insufficient statistical power)  
**Target**: 20 seeds for publication-grade statistics  
**Impact**: Strengthen p-value validity, compute robust confidence intervals

**Implementation**:
```bash
cd autonomous-baseline

# Run 15 additional seeds (47-61)
python phase10_gp_active_learning/experiments/tier2_clean_benchmark.py \
  --seeds 15 \
  --seed-start 47 \
  --rounds 20 \
  --batch 20 \
  --initial 100 \
  2>&1 | tee logs/tier2_seeds_47-61.log

# Merge with existing 5 seeds (42-46)
python scripts/merge_benchmark_results.py \
  --input1 evidence/phase10/tier2_clean/results.json \
  --input2 evidence/phase10/tier2_seeds_47-61/results.json \
  --output evidence/phase10/tier2_20seeds/results.json

# Recompute statistics with bootstrap
python scripts/compute_robust_statistics.py \
  --input evidence/phase10/tier2_20seeds/results.json \
  --bootstrap 10000 \
  --output evidence/phase10/tier2_20seeds/statistics.json
```

**Deliverables**:
- `evidence/phase10/tier2_20seeds/results.json` (20 seeds × 3 strategies)
- `evidence/phase10/tier2_20seeds/statistics.json` (bootstrap CIs)
- Updated plots with tighter error bands

**Success Criteria**:
- p-value stability: |Δp| < 0.01 vs. 5-seed result
- 95% CI excludes zero
- Grade improvement: C+ → B

---

### Action 2: Add External Baselines (XGBoost, RF) ⏱️ 4 hours

**Gap**: No comparison to standard ML baselines  
**Target**: Add XGBoost and Random Forest (easy, no structure data needed)  
**Impact**: Validate DKL superiority against industry-standard models

**Implementation**:
```python
# scripts/add_baselines.py
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def benchmark_xgboost():
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    # Run same active learning loop as DKL/GP
    return run_active_learning(..., model='xgboost')

def benchmark_random_forest():
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        random_state=42
    )
    return run_active_learning(..., model='rf')
```

**Run**:
```bash
python scripts/add_baselines.py \
  --strategies xgboost,random_forest \
  --seeds 5 \
  --output evidence/phase10/tier2_baselines/
```

**Deliverables**:
- `evidence/phase10/tier2_baselines/xgboost_results.json`
- `evidence/phase10/tier2_baselines/rf_results.json`
- Updated comparison table: DKL vs GP vs XGBoost vs RF vs Random

**Success Criteria**:
- DKL RMSE ≤ XGBoost RMSE (ideally DKL < XGBoost)
- DKL RMSE ≤ RF RMSE
- If DKL loses to XGBoost: Document clearly, still valuable (simpler model wins)
- Grade improvement: B → B+

---

### Action 3: Physics Interpretability Analysis ⏱️ 2 days

**Gap**: No analysis of what DKL learned  
**Target**: Correlate learned features with known physics, visualize latent space  
**Impact**: Scientific credibility (not just a black box)

**Implementation**:
```python
# scripts/analyze_learned_features.py
import torch
import numpy as np
from sklearn.manifold import TSNE
import shap
import seaborn as sns

def correlate_features_with_physics(dkl_model, X, y):
    """Correlate 16D learned features with physics descriptors"""
    
    # Extract learned features
    with torch.no_grad():
        Z = dkl_model.feature_extractor(torch.tensor(X, dtype=torch.float64))
        Z = Z.cpu().numpy()  # (N, 16)
    
    # Physics descriptors (from matminer features)
    physics = {
        'mean_valence': X[:, idx['MeanValenceElectrons']],
        'mean_mass': X[:, idx['MeanAtomicWeight']],
        'mean_EN': X[:, idx['MeanElectronegativity']],
        'range_EN': X[:, idx['RangeElectronegativity']],
        'mean_ionic_radius': X[:, idx['MeanIonicRadius']],
    }
    
    # Compute Pearson correlations
    corr_matrix = np.zeros((16, len(physics)))
    for i in range(16):
        for j, (name, values) in enumerate(physics.items()):
            corr_matrix[i, j] = np.corrcoef(Z[:, i], values)[0, 1]
    
    # Plot heatmap
    plt.figure(figsize=(8, 10))
    sns.heatmap(corr_matrix, 
                xticklabels=list(physics.keys()),
                yticklabels=[f'Z{i}' for i in range(16)],
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson r'})
    plt.title('Learned Features vs. Physics Descriptors')
    plt.tight_layout()
    plt.savefig('evidence/phase10/tier2_clean/feature_physics_correlations.png', dpi=300)
    
    # Identify top correlations
    top_corrs = []
    for i in range(16):
        for j, (name, _) in enumerate(physics.items()):
            if abs(corr_matrix[i, j]) > 0.3:
                top_corrs.append((i, name, corr_matrix[i, j]))
    
    return corr_matrix, top_corrs

def visualize_latent_space(dkl_model, X, y):
    """t-SNE visualization of learned 16D space"""
    
    # Extract features
    with torch.no_grad():
        Z = dkl_model.feature_extractor(torch.tensor(X, dtype=torch.float64))
        Z = Z.cpu().numpy()
    
    # t-SNE reduction to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    Z_2d = tsne.fit_transform(Z)
    
    # Plot colored by Tc
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap='viridis', 
                         s=20, alpha=0.6, edgecolors='k', linewidth=0.3)
    plt.colorbar(scatter, label='Tc (K)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE of Learned 16D Features\n(colored by Tc)')
    plt.tight_layout()
    plt.savefig('evidence/phase10/tier2_clean/tsne_learned_space.png', dpi=300)
    
    # Check if high-Tc compounds cluster
    high_tc_mask = y > 50  # High-Tc threshold
    if high_tc_mask.sum() > 0:
        # Compute silhouette score
        from sklearn.metrics import silhouette_score
        labels = (y > 50).astype(int)
        silhouette = silhouette_score(Z_2d, labels)
        print(f"Silhouette score (high vs low Tc): {silhouette:.3f}")

def shap_feature_importance(dkl_model, X_train, X_test):
    """SHAP analysis of input feature importance"""
    
    # Use a subset for speed
    X_train_sample = X_train[:500]
    X_test_sample = X_test[:100]
    
    # SHAP Deep Explainer
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float64)
            z = dkl_model.feature_extractor(x_tensor)
        return z.cpu().numpy()
    
    explainer = shap.DeepExplainer(model_predict, torch.tensor(X_train_sample, dtype=torch.float64))
    shap_values = explainer.shap_values(torch.tensor(X_test_sample, dtype=torch.float64))
    
    # Summary plot
    shap.summary_plot(shap_values, X_test_sample, 
                     feature_names=feature_names,  # From matminer
                     show=False)
    plt.tight_layout()
    plt.savefig('evidence/phase10/tier2_clean/shap_feature_importance.png', dpi=300)
```

**Run**:
```bash
python scripts/analyze_learned_features.py \
  --model checkpoints/dkl_final.pkl \
  --data data/uci_superconductivity.csv \
  --output evidence/phase10/tier2_clean/
```

**Deliverables**:
- `feature_physics_correlations.png` (16 × 5 heatmap)
- `tsne_learned_space.png` (2D visualization)
- `shap_feature_importance.png` (input attribution)
- `physics_interpretation.md` (written analysis)

**Success Criteria**:
- ≥3 learned features correlate (|r| > 0.3) with known physics
- High-Tc compounds show spatial clustering in t-SNE
- SHAP highlights electron count, mass, electronegativity as top features
- Grade improvement: B+ → A-

---

## 📊 P1: SHOULD-FIX (2-3 days → A- grade)

### Action 4: Acquisition Function Sweep ⏱️ 6 hours

**Gap**: Only tested Expected Improvement (EI)  
**Target**: Compare EI vs PI vs UCB  
**Impact**: Standard BO practice, shows EI choice is justified

**Implementation**:
```python
# scripts/compare_acquisitions.py
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound
)

acquisitions = {
    'EI': lambda model, best_f: ExpectedImprovement(model, best_f),
    'PI': lambda model, best_f: ProbabilityOfImprovement(model, best_f),
    'UCB': lambda model, best_f: UpperConfidenceBound(model, beta=2.0),
}

# Run benchmark with each acquisition
for acq_name, acq_fn in acquisitions.items():
    results = run_al_benchmark(
        model='dkl',
        acquisition=acq_fn,
        seeds=5,
        ...
    )
    save_results(f'evidence/phase10/acquisition_{acq_name}.json')
```

**Deliverable**: `evidence/phase10/acquisition_comparison.json`  
**Time**: 6 hours

---

### Action 5: Epistemic Efficiency Metrics ⏱️ 4 hours

**Gap**: No quantification of information gain per query  
**Target**: Compute ΔEntropy / experiment  
**Impact**: Theoretically rigorous justification for active learning

**Implementation**:
```python
# scripts/compute_information_gain.py
def compute_entropy_reduction(model, X_acquired, y_acquired):
    """Measure information gain per query"""
    
    # Entropy before acquisition
    posterior_before = model.posterior(X_candidates)
    H_before = posterior_before.variance.log().mean()  # Log variance ≈ entropy
    
    # Update model
    model_updated = model.condition_on_observations(X_acquired, y_acquired)
    
    # Entropy after
    posterior_after = model_updated.posterior(X_candidates)
    H_after = posterior_after.variance.log().mean()
    
    return (H_before - H_after).item()  # bits
```

**Deliverable**: `evidence/phase10/information_gain.json`  
**Time**: 4 hours

---

### Action 6: Reproducibility Artifacts ⏱️ 2 hours

**Gap**: No SHA-256 manifests, no saved checkpoints  
**Target**: Full provenance trail  
**Impact**: Publication-grade reproducibility

**Implementation**:
```bash
# scripts/generate_provenance.py
python scripts/generate_provenance.py \
  --dataset data/uci_superconductivity.csv \
  --models checkpoints/*.pkl \
  --output evidence/phase10/tier2_clean/MANIFEST.sha256

# Test deterministic re-run
python scripts/test_reproducibility.py \
  --seed 42 \
  --runs 2 \
  --tolerance 1e-6
```

**Deliverable**: `MANIFEST.sha256`, `REPRODUCIBILITY_REPORT.md`  
**Time**: 2 hours

---

## 💡 P2: NICE-TO-HAVE (1+ week → A grade)

### Action 7: Head-to-Head Literature Comparison

- Get Stanev 2018 trained model
- Run on our exact train/test splits
- Direct RMSE comparison

**Time**: 1 week (requires external model access)

---

## 📈 GRADE PROGRESSION TIMELINE

| Week | Actions Complete | Grade | Confidence |
|------|------------------|-------|------------|
| **Now** | Baseline (5 seeds, no baselines) | C+/B- | Low |
| **Week 1** | +15 seeds, +XGBoost/RF, +Physics | **B+** | Medium |
| **Week 2** | +Acquisition sweep, +Epistemic, +Provenance | **A-** | High |
| **Week 3+** | +Literature head-to-head | **A** | Very High |

---

## 🚀 EXECUTION PLAN

###Week 1 (Priority P0)

**Day 1** (3 hours):
```bash
# Morning: Run 15 additional seeds
nohup python tier2_clean_benchmark.py --seeds 15 --seed-start 47 > logs/seeds_47-61.log 2>&1 &

# Afternoon: Merge and recompute stats
python scripts/merge_benchmark_results.py
python scripts/compute_robust_statistics.py
```

**Day 2** (4 hours):
```bash
# Add XGBoost and RF baselines
python scripts/add_baselines.py --strategies xgboost,random_forest --seeds 5
```

**Day 3-4** (2 days):
```bash
# Physics interpretability analysis
python scripts/analyze_learned_features.py
# Write physics_interpretation.md
```

### Week 2 (Priority P1)

**Day 5** (6 hours):
```bash
# Acquisition function sweep
python scripts/compare_acquisitions.py
```

**Day 6** (4 hours):
```bash
# Epistemic efficiency
python scripts/compute_information_gain.py
```

**Day 7** (2 hours):
```bash
# Reproducibility artifacts
python scripts/generate_provenance.py
python scripts/test_reproducibility.py
```

---

## ✅ SUCCESS METRICS

### B+ Grade (Week 1 Target)
- ✅ 20 seeds with bootstrap CIs
- ✅ XGBoost + RF comparison
- ✅ Physics interpretability (≥3 correlations)
- ✅ p < 0.05 maintained with 20 seeds

### A- Grade (Week 2 Target)
- ✅ All B+ criteria
- ✅ Acquisition function comparison
- ✅ Epistemic efficiency quantified
- ✅ Full reproducibility artifacts

### A Grade (Week 3+ Target)
- ✅ All A- criteria
- ✅ Literature head-to-head comparison
- ✅ Novelty statement clarified

---

## 📝 AUTOMATION SCRIPTS

All scripts created in `scripts/` directory:
- `audit_validation.py` ✅ (created)
- `merge_benchmark_results.py` (TODO)
- `compute_robust_statistics.py` (TODO)
- `add_baselines.py` (TODO)
- `analyze_learned_features.py` (TODO)
- `compare_acquisitions.py` (TODO)
- `compute_information_gain.py` (TODO)
- `generate_provenance.py` (TODO)
- `test_reproducibility.py` (TODO)

---

**Status**: Roadmap complete, audit script ready  
**Next**: Run `python scripts/audit_validation.py --full` to generate baseline audit  
**Then**: Execute Week 1 P0 actions

**© 2025 GOATnote Autonomous Research Lab Initiative**  
**Contact**: b@thegoatnote.com

