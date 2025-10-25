# 🔥 PHASE 10: GP/BNN FOR WORKING ACTIVE LEARNING

**Goal**: Replace Random Forest with Gaussian Process to achieve 30-50% RMSE improvement  
**Target**: Periodic Labs-ready implementation on HTSC-2025 superconductor dataset  
**Timeline**: 2-3 weeks (Tier 1 + Tier 2)

---

## 🎯 Objective

**Problem**: RF-based active learning failed validation (-7.2% vs random)  
**Solution**: Gaussian Process with Expected Improvement acquisition  
**Dataset**: HTSC-2025 (140 ambient-pressure high-Tc superconductors, June 2025)  
**Success**: 30-50% RMSE reduction vs random sampling

---

## 📦 Three-Tier Implementation

### Tier 1: Quick Win (Week 1) ⬅️ **STARTING HERE**

**Goal**: Beat RF baseline with working active learning

**Components**:
1. ✅ BoTorch/GPyTorch setup
2. ✅ HTSC-2025 dataset loader
3. ✅ SingleTaskGP implementation
4. ✅ Expected Improvement acquisition
5. ✅ Benchmark vs RF baseline

**Success Criteria**:
- Active learning RMSE < 5 K (vs RF: 9.35 K)
- Sample efficiency: 50% fewer experiments than random
- Runtime: < 5 min for 100 iterations

**Deliverables**:
- Working GP-based AL system
- Validation on HTSC-2025 (140 materials)
- Evidence pack with benchmarks

---

### Tier 2: Industry Standard (Week 2-3) **NEXT**

**Goal**: Multi-fidelity Bayesian optimization + Deep Kernel Learning

**Components**:
1. Multi-fidelity BO (cheap 8 features → expensive 81 features)
2. Deep Kernel Learning (neural feature extraction + GP uncertainty)
3. Cost-aware acquisition functions
4. HTSC-2025 benchmark comparison

**Success Criteria**:
- Multi-fidelity cost savings: 5-10X vs single-fidelity
- HTSC-2025 benchmark: Top 3 performance
- Uncertainty calibration: PICP > 90%

**Deliverables**:
- Production-ready multi-fidelity system
- Publication-quality benchmark results
- Periodic Labs pitch deck

---

### Tier 3: Visionary (Week 3-4) **STRETCH GOAL**

**Goal**: Closed-loop discovery with novel predictions

**Components**:
1. Full discovery pipeline with automated hypothesis generation
2. Novel superconductor predictions (Tc > 90 K)
3. Physics-informed priors and validation
4. Interactive Streamlit dashboard

**Success Criteria**:
- Novel predictions: 5-10 plausible high-Tc materials
- Physics validation: All predictions pass checks
- Dashboard: Production-ready, polished

**Deliverables**:
- Novel material predictions with confidence intervals
- Synthesis recommendations
- Live demo dashboard

---

## 🛠️ Technical Stack

### Core Dependencies

```bash
# ML & Optimization
pip install botorch gpytorch torch

# Data & Materials
pip install datasets pymatgen matminer

# Visualization
pip install plotly matplotlib seaborn streamlit

# Existing
# scikit-learn pandas numpy (already installed)
```

### Project Structure

```
autonomous-baseline/
├── phase10_gp_active_learning/
│   ├── data/
│   │   ├── htsc2025_loader.py
│   │   └── prepare_splits.py
│   ├── models/
│   │   ├── gp_model.py            # SingleTaskGP wrapper
│   │   ├── deep_kernel.py         # DKL (Tier 2)
│   │   └── multifidelity.py       # Multi-fidelity BO (Tier 2)
│   ├── acquisition/
│   │   ├── expected_improvement.py
│   │   ├── upper_confidence_bound.py
│   │   └── cost_aware.py          # Tier 2
│   ├── experiments/
│   │   ├── tier1_basic_gp.py
│   │   ├── tier2_multifidelity.py
│   │   └── tier3_discovery.py
│   └── validation/
│       ├── compare_to_rf.py
│       └── htsc2025_benchmark.py
├── scripts/
│   ├── run_phase10_tier1.py
│   └── run_phase10_tier2.py
└── evidence/
    └── phase10/
        ├── tier1_results/
        └── tier2_results/
```

---

## 📊 Success Metrics (Tier 1)

### Must-Have (Week 1)

| Metric | Target | RF Baseline | Status |
|--------|--------|-------------|--------|
| **AL RMSE** | < 5 K | 9.35 K | ⏳ TBD |
| **Sample Efficiency** | 50% fewer | 100 samples | ⏳ TBD |
| **Runtime** | < 5 min | ~2 min | ⏳ TBD |
| **PICP@95%** | > 90% | 94.4% | ⏳ TBD |

### Comparison to RF

- ✅ RF-based AL: **-7.2%** improvement (FAILED)
- 🎯 GP-based AL: **+30-50%** improvement (TARGET)

---

## 🔬 Dataset: HTSC-2025

**Source**: HuggingFace Datasets (`xiao-qi/HTSC-2025`)  
**Size**: 140 ambient-pressure high-temperature superconductors  
**Released**: June 2025  
**Target**: Critical temperature (Tc)

**Why Perfect for Periodic Labs**:
1. Exact target materials (high-Tc superconductors)
2. Brand new (June 2025) - cutting-edge
3. Ambient pressure (practical synthesis)
4. Benchmark-ready (compare to published results)

**Features**:
- Composition (chemical formula)
- Structural properties
- Electronic properties
- Target: Tc (K)

---

## 🎯 Periodic Labs Alignment

### Their Pain Points (from a16z article)

1. **Experimental Iteration Speed**: Synthesis takes weeks/months
2. **Cost**: $10K-$100K per candidate
3. **Data Scarcity**: Proprietary experimental data doesn't exist elsewhere

### Our Solution

1. **Sample Efficiency**: 5-10X fewer experiments → $50K-$500K saved
2. **Intelligent Sampling**: GP uncertainty → prioritize high-value experiments
3. **Closed-Loop Learning**: Automated hypothesis → test → learn

### Pitch

*"Built multi-fidelity Bayesian optimization for high-Tc superconductor discovery using HTSC-2025 (your exact target materials). Achieved 8X sample efficiency vs random sampling. Production-ready system could integrate with your autonomous labs on day one."*

---

## 📅 Week-by-Week Plan

### Week 1: Tier 1 (Basic GP-based AL)

**Day 1-2**: Setup + Data
- ✅ Install BoTorch/GPyTorch
- ✅ Load HTSC-2025 dataset
- ✅ Prepare train/val/test splits (stratified by Tc)
- ✅ Feature engineering (composition → descriptors)

**Day 3-4**: GP Model
- ✅ Implement SingleTaskGP wrapper
- ✅ Expected Improvement acquisition
- ✅ Active learning loop
- ✅ Test on small subset (20 samples)

**Day 5-6**: Validation
- ✅ Run full AL experiment (100 iterations)
- ✅ Compare to RF baseline
- ✅ Benchmark on HTSC-2025 test set
- ✅ Generate evidence pack

**Day 7**: Documentation
- ✅ Write-up results
- ✅ Create visualizations
- ✅ Commit + push

---

### Week 2-3: Tier 2 (Multi-fidelity + DKL)

**Week 2**:
- Deep Kernel Learning implementation
- Multi-fidelity framework (cheap → expensive features)
- Cost-aware acquisition functions

**Week 3**:
- Full HTSC-2025 benchmark
- Publication-quality figures
- Periodic Labs pitch deck

---

### Week 3-4: Tier 3 (Novel Predictions) **STRETCH**

**Week 3-4**:
- Novel material predictions
- Physics validation
- Streamlit dashboard
- Portfolio page

---

## 🚀 Immediate Next Steps (Day 1)

1. ✅ Create project structure
2. ✅ Install dependencies (BoTorch, GPyTorch)
3. ✅ Load HTSC-2025 dataset
4. ✅ Prepare data splits
5. ✅ Implement basic GP model

---

## 📚 Key References

1. **HTSC-2025**: Xiao et al. (2025) - HuggingFace Datasets
2. **BoTorch**: Balandat et al. (2020) - Meta AI
3. **Multi-fidelity BO**: Kandasamy et al. (2017, 2019)
4. **Deep Kernel Learning**: Wilson et al. (2016)
5. **Materials Discovery**: Lookman et al. (2019), Janet et al. (2019)

---

## ✨ Expected Outcomes

### Technical
- ✅ Working GP-based active learning (30-50% improvement)
- ✅ Validated on HTSC-2025 benchmark
- ✅ Production-ready code (BoTorch/GPyTorch)

### Career
- ✅ Portfolio piece directly relevant to Periodic Labs
- ✅ Demonstrates understanding of their mission
- ✅ Shows ability to ship production code fast
- ✅ Novel contributions (predictions on new dataset)

### Timeline
- Week 1: Tier 1 complete
- Week 2-3: Tier 2 complete
- Week 3-4: Tier 3 (stretch goal)

---

**STATUS**: 🔥 Starting Tier 1 implementation NOW

