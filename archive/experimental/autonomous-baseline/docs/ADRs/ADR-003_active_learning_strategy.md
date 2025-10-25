# ADR-003: Diversity-Aware Active Learning with Risk Gates

**Status**: Accepted  
**Date**: 2024-10-09  
**Deciders**: Research Team, Lab Operations

---

## Context

Autonomous laboratories operate under **constrained budgets**:
- Synthesis cost: $500-5000/compound
- Characterization time: 1-7 days/sample
- Total budget: 100-500 experiments/year

**Random exploration** wastes 30-50% of budget on uninformative experiments. We need **active learning** to intelligently select next experiments.

However, naive AL has failure modes:
- **Mode collapse**: Repeatedly sampling similar compounds
- **OOD exploitation**: Selecting unrealistic candidates
- **High-risk queries**: Sampling high-uncertainty candidates that may fail synthesis

---

## Decision

**We will use diversity-aware active learning with multi-gate risk control**:

1. **Acquisition function** (UCB, EI, MaxVar, EIG-proxy) scores candidates
2. **Diversity selection** (k-Medoids or DPP) ensures chemical space coverage
3. **Risk gates** (OOD, σ² cap, cost) filter unsafe candidates
4. **Budget controller** enforces spending limits

---

## Rationale

### Why Diversity-Aware AL?

Standard AL (e.g., uncertainty sampling) suffers from **mode collapse**:

Example:
```
Round 1: Select compound A (high σ)
Round 2: Select compound A' (similar to A, still high σ)
Round 3: Select compound A'' (still similar, still high σ)
...
```

**Problem**: Explores local neighborhood instead of global chemical space.

**Solution**: After scoring by acquisition function, apply **diversity filter** to select K representatives from top-M candidates.

---

### Acquisition Functions

| Function | Formula | Exploration | Exploitation | Use Case |
|----------|---------|-------------|--------------|----------|
| **UCB** | μ + β·σ | High | High | **Default** (balanced) |
| **EI** | E[max(0, y_best - y)] | Medium | High | Optimization (maximize Tc) |
| **MaxVar** | σ² | Very High | Low | Fill gaps (pure exploration) |
| **EIG-Proxy** | H(y\|D) - H(y\|D∪x) | High | Medium | Information maximization |

**Default**: UCB with β=2.0 (well-calibrated for most tasks per literature)

**Tuning**: Increase β for more exploration, decrease for more exploitation.

---

### Diversity Selection Methods

#### k-Medoids (PAM)

```python
# 1. Score all unlabeled candidates
scores = acquisition_fn(X_pool)

# 2. Select top-M candidates (M >> K)
top_M_idx = np.argsort(scores)[-M:]

# 3. Run k-Medoids on features to select K representatives
medoids = kmedoids(X_pool[top_M_idx], n_clusters=K)
selected = top_M_idx[medoids]
```

**Pros**: Fast (O(KM) per iteration), interpretable (medoids are actual candidates)  
**Cons**: Greedy, may miss global optimum

---

#### DPP (Determinantal Point Process)

```python
# 1. Score all unlabeled candidates
scores = acquisition_fn(X_pool)

# 2. Construct kernel matrix
L_ij = score_i · score_j · exp(-||x_i - x_j||² / 2σ²)

# 3. Sample K diverse points from DPP(L)
selected = dpp_sample(L, K)
```

**Pros**: Probabilistic, globally optimal (MAP), theoretically grounded  
**Cons**: Slower (O(N³)), requires tuning σ

**Default**: k-Medoids (faster, sufficient for most cases)

---

### Risk Gates

Before querying selected candidates, apply filters:

#### Gate 1: OOD Detection

```python
if mahalanobis_distance(x) > τ_M or kde_density(x) < τ_K:
    REJECT  # Out of distribution
```

**Purpose**: Prevent extrapolation to unrealistic chemistry

---

#### Gate 2: Uncertainty Cap

```python
if σ²(x) > σ²_max:
    REJECT  # Too uncertain (likely to fail synthesis)
```

**Purpose**: Avoid high-risk candidates

**Default**: σ²_max = 90th percentile of calibration set variance

---

#### Gate 3: Cost Filter (Optional)

```python
if cost(x) > budget_remaining / queries_remaining:
    REJECT  # Too expensive
```

**Purpose**: Stay within budget

**Requirement**: Need cost model (e.g., synthesis difficulty predictor)

---

#### Gate 4: Budget Controller

```python
if total_queries >= budget_total:
    STOP  # Budget exhausted
```

---

### Conditional GO (Exploration)

Allow ≤K OOD queries per round if:
- High information gain (top 5% of EIG-proxy)
- Expert approval (human-in-loop for OOD)

**Purpose**: Balance safety with exploration

---

## Implementation Workflow

```python
# Initialize
labeled_set = seed_labeled_data (N=50, stratified)
unlabeled_pool = remaining_train_data

for round in range(max_rounds):
    # 1. Train model on labeled set
    model.fit(labeled_set)
    
    # 2. Score unlabeled pool
    scores = acquisition_fn(model, unlabeled_pool)
    
    # 3. Select top-M candidates
    top_M = np.argsort(scores)[-M:]
    
    # 4. Apply diversity selection
    if diversity_method == "kmedoids":
        selected = kmedoids(unlabeled_pool[top_M], K)
    elif diversity_method == "dpp":
        selected = dpp_sample(unlabeled_pool[top_M], K)
    
    # 5. Apply risk gates
    filtered = []
    for x in selected:
        if passes_all_gates(x):
            filtered.append(x)
    
    # 6. Query oracle (autonomous lab)
    y_new = autonomous_lab.query(filtered)
    
    # 7. Update labeled set
    labeled_set.add(filtered, y_new)
    unlabeled_pool.remove(filtered)
    
    # 8. Log metrics
    log_rmse(model, test_set)
    log_info_gain(...)
    log_diversity(...)
```

---

## Validation Metrics

### 1. RMSE Reduction vs Random

```
Improvement = (RMSE_random - RMSE_AL) / RMSE_random
```

**Target**: ≥30% after 20 acquisitions (K=10 per round)

**Baseline**: Random sampling from same pool

---

### 2. Information Gain per Query

```
IG = H(y | D) - H(y | D ∪ {x, y})
```

**Proxy**: -log(σ²) or mutual information approximation

**Target**: ≥1.5 bits/query (average)

---

### 3. Chemical Space Coverage

```
Coverage = unique_families_explored / total_families
```

**Target**: ≥80% after 100 queries

---

### 4. Regret (Optimization Tasks)

```
Regret = max(y_pool) - max(y_selected)
```

**Target**: Minimize (but not primary goal for baseline)

---

## Ablation Studies

Test components individually:

1. **Acquisition only** (no diversity): UCB without k-Medoids → expect mode collapse
2. **Diversity only** (no acquisition): Random + k-Medoids → suboptimal RMSE
3. **No OOD gates**: Allow OOD queries → expect failures
4. **No uncertainty cap**: Allow high-σ² queries → expect synthesis failures

**Result**: Combined system should outperform all ablations.

---

## Consequences

### Positive

- ✅ **Budget efficiency**: 30-50% fewer wasted experiments
- ✅ **Coverage**: Explores diverse chemistry (80%+ families)
- ✅ **Safety**: OOD/risk gates prevent failures
- ✅ **Interpretable**: k-Medoids selections are actual compounds (not interpolations)

### Negative

- ⚠️ **Computational cost**: k-Medoids + gates add 5-10s per round (negligible vs synthesis time)
- ⚠️ **Hyperparameter sensitivity**: β, M, σ²_max require tuning
- ⚠️ **Cold start**: Requires good seed set (N≥50, stratified)

### Mitigation Strategies

1. **Hyperparameter search**: Grid search β, M on validation set
2. **Online tuning**: Adjust β based on exploration/exploitation balance
3. **Warm start**: Use literature data or cheap DFT for seed set

---

## Deployment Gates

### Gate 1: AL Validation (5 Seeds)

Run AL pipeline with 5 random seeds:
- ✅ Mean RMSE reduction ≥30% after 20 rounds (vs random baseline)
- ✅ Std dev RMSE reduction ≤10% (robust)

### Gate 2: Diversity Validation

- ✅ Coverage ≥70% unique families after 50 queries
- ✅ No single family >20% of selections (mode collapse check)

### Gate 3: Safety Validation

- ✅ Zero OOD queries (unless conditional GO approved)
- ✅ Zero high-σ² queries (σ² > σ²_max)

---

## Related ADRs

- ADR-001: Composition-First Features (defines chemical space)
- ADR-002: Uncertainty Calibration (provides σ for acquisition)

---

## References

- Settles, B. (2009). *Active Learning Literature Survey*. Computer Sciences Technical Report 1648.
- Kulesza, A., & Taskar, B. (2012). *Determinantal Point Processes for Machine Learning*. Foundations and Trends in ML, 5(2-3), 123-286.
- Lookman, T., et al. (2019). *Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design*. npj Computational Materials, 5, 21.
- Ren, F., et al. (2018). *Accelerated discovery of metallic glasses through iteration of machine learning and high-throughput experiments*. Science Advances, 4, eaaq1566.

---

**Supersedes**: None  
**Superseded by**: None  
**Status**: Active

