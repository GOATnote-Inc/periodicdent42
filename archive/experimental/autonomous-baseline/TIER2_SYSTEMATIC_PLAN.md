# Tier 2 Systematic Execution Plan

**Objective**: Address remaining critical review points with scientific rigor  
**Timeline**: 5 days (Oct 10-14, 2025)  
**Target Grade**: A- (from current B+)

---

## Priority Queue (Ordered by Scientific Impact)

### P0: Sharpness Analysis (Critical Flaw #1)
**Issue**: Calibration coverage alone is insufficient - need interval width analysis

**Tasks**:
1. Implement sharpness metrics (avg width | covered, avg width | not covered)
2. Compute conditional coverage by Tc quantile (low/mid/high)
3. Analyze coverage breakdown by iteration number
4. Statistical test: Does sharpness differ between CEI and EI?
5. Generate sharpness plots with confidence intervals

**Deliverables**:
- `scripts/analyze_sharpness.py` (200 lines)
- `experiments/novelty/noise_sensitivity/sharpness_analysis.json`
- `experiments/novelty/noise_sensitivity/sharpness_plots/` (3 figures)
- Updated NOVELTY_FINDING.md with sharpness metrics

**Success Criteria**:
- Report avg PI width ± SD for both methods
- Conditional coverage stratified by 3+ categories
- Statistical comparison (paired t-test on PI widths)

**Time Estimate**: 6 hours

---

### P1: DKL Ablation Study (Critical Flaw #4)
**Issue**: "DKL beats GP" lacks depth - need to isolate feature learning value

**Tasks**:
1. Implement PCA+GP baseline (16D via PCA, not learning)
2. Implement random projection+GP baseline (16D random features)
3. Re-run benchmark: DKL vs PCA+GP vs Random+GP vs GP-raw
4. Report RMSE, training time, inference time for each
5. Ablate latent dimensionality (4D, 8D, 16D, 32D)

**Deliverables**:
- `scripts/dkl_ablation.py` (300 lines)
- `experiments/ablations/dkl_vs_baselines_results.json`
- `experiments/ablations/latent_dim_sweep.json`
- Updated NOVELTY_FINDING.md with ablation results

**Success Criteria**:
- ≥3 baselines compared with statistical tests
- Wall-clock time reported for each method
- Latent dim sweep shows optimal dimensionality
- Quantify feature learning contribution: DKL - PCA+GP

**Time Estimate**: 8 hours

---

### P2: Computational Profiling (Critical Flaw #9)
**Issue**: "40% cost reduction" claim is vague - need detailed breakdown

**Tasks**:
1. Profile time breakdown: GP inference, EI scoring, conformal calibration
2. Measure scaling with pool size (n=100, 500, 1000, 5000)
3. Separate CPU operations from actual bottlenecks
4. Report hardware specs and reproducibility
5. Compare Filter-CEI cost savings (if P2 cost = dominant, savings minimal)

**Deliverables**:
- `scripts/profile_computational_cost.py` (200 lines)
- `experiments/profiling/cost_breakdown.json`
- `experiments/profiling/scaling_curves.png`
- Cost table in NOVELTY_FINDING.md

**Success Criteria**:
- Per-operation breakdown (% of total time)
- Scaling analysis (O(n²) vs O(n) validation)
- Hardware specs documented
- Identify actual bottleneck (GP vs scoring vs conformal)

**Time Estimate**: 4 hours

---

### P3: Filter-CEI Pareto Frontier (Critical Flaw #5)
**Issue**: "Keep top 20%" lacks justification - need sensitivity analysis

**Tasks**:
1. Implement Filter-CEI with variable keep_fraction (5%, 10%, 20%, 30%, 50%)
2. Run on 3 noise levels (σ=0, 10, 20 K)
3. Compute Pareto frontier: RMSE vs cost (candidates evaluated)
4. Identify optimal keep_fraction per noise level
5. Formalize as Algorithm box for paper

**Deliverables**:
- `experiments/novelty/filter_cei_pareto.py` (250 lines)
- `experiments/novelty/filter_cei/pareto_frontier.json`
- `experiments/novelty/filter_cei/pareto_plot.png`
- Algorithm 2 box in paper

**Success Criteria**:
- 5 keep_fraction values tested
- Pareto curve shows trade-off clearly
- Optimal point identified with justification
- Cost reduction quantified (e.g., 80% cost → 5% RMSE increase)

**Time Estimate**: 5 hours

---

### P4: Time-to-Discovery Metrics (Critical Flaw #6)
**Issue**: "Iterations until Tc > 80K" is arbitrary - need established metrics

**Tasks**:
1. Implement simple regret: f(x_best) - f(x_optimal)
2. Implement cumulative regret: Σ[f(x_opt) - f(x_t)]
3. Report both metrics for all methods
4. Compare to "iterations to threshold" for domain relevance
5. Cite optimization literature for metric choice

**Deliverables**:
- `scripts/compute_regret_metrics.py` (150 lines)
- Updated results with simple/cumulative regret
- Correlation analysis: regret vs iterations-to-threshold

**Success Criteria**:
- Report simple + cumulative regret (not just final RMSE)
- Show correlation with domain metric
- Cite ≥2 optimization papers for metric justification

**Time Estimate**: 3 hours

---

## Execution Order (5 Days)

### Day 1 (Friday, Oct 10): Sharpness Analysis
**Hours**: 6  
**Output**: Sharpness metrics, conditional coverage, plots

```bash
# Morning (3h): Implement sharpness analysis
python scripts/analyze_sharpness.py

# Afternoon (3h): Generate plots, update docs
python scripts/plot_sharpness.py
```

**Checkpoint**: Sharpness results in NOVELTY_FINDING.md

---

### Day 2 (Saturday, Oct 11): DKL Ablation - Part 1
**Hours**: 6  
**Output**: PCA+GP and Random+GP baselines

```bash
# Morning (3h): Implement baselines
# Afternoon (3h): Run benchmark (3 methods × 5 seeds)
python scripts/dkl_ablation.py --baselines pca random
```

**Checkpoint**: PCA+GP and Random+GP results

---

### Day 3 (Sunday, Oct 12): DKL Ablation - Part 2
**Hours**: 4  
**Output**: Latent dim sweep, comparison table

```bash
# Morning (2h): Latent dim sweep
python scripts/dkl_ablation.py --latent-dims 4,8,16,32

# Afternoon (2h): Analysis and documentation
python scripts/analyze_ablation_results.py
```

**Checkpoint**: Complete DKL ablation section in paper

---

### Day 4 (Monday, Oct 13): Profiling + Filter-CEI
**Hours**: 6  
**Output**: Cost breakdown, Pareto frontier

```bash
# Morning (3h): Computational profiling
python scripts/profile_computational_cost.py

# Afternoon (3h): Filter-CEI Pareto analysis
python experiments/novelty/filter_cei_pareto.py
```

**Checkpoint**: Cost table + Pareto plot in paper

---

### Day 5 (Tuesday, Oct 14): Metrics + Integration
**Hours**: 4  
**Output**: Regret metrics, final documentation

```bash
# Morning (2h): Implement regret metrics
python scripts/compute_regret_metrics.py

# Afternoon (2h): Update all docs, check consistency
python scripts/verify_tier2_complete.py
```

**Checkpoint**: All Tier 2 items complete, documentation updated

---

## Success Metrics (Objective)

### Quantitative
- [ ] Sharpness: Report avg PI width ± SD for CEI vs EI (paired test)
- [ ] DKL: Quantify feature learning gain (DKL - PCA, in RMSE)
- [ ] Profiling: Identify bottleneck operation (% of total time)
- [ ] Filter-CEI: Report Pareto optimal point (cost vs RMSE)
- [ ] Metrics: Report simple + cumulative regret (not just RMSE)

### Qualitative
- [ ] All claims backed by statistical tests (not just plots)
- [ ] Hardware specs documented for reproducibility
- [ ] Figures have proper error bars + sample sizes
- [ ] No unsubstantiated claims ("40% faster" without breakdown)
- [ ] Cite ≥3 papers for methods (PCA, regret metrics, Pareto optimization)

---

## Anti-Patterns to Avoid

### ❌ What NOT to Do
1. **Overhyping**: "Revolutionary!", "Game-changer!", "Impossible to ignore!"
2. **Vague claims**: "Much faster", "Significantly better" (without numbers)
3. **Cherry-picking**: Only report favorable results
4. **P-hacking**: Run many tests, report only p < 0.05
5. **Missing baselines**: Compare to strawman, not strong baselines

### ✅ What TO Do
1. **Measured language**: "X achieves Y ± Z (n=N, p=P)"
2. **Precise claims**: "23% reduction (CI: [18%, 28%], n=10)"
3. **Report all results**: Include null findings, failures
4. **Pre-register**: State hypotheses before experiments
5. **Strong baselines**: Compare to best available methods

---

## Documentation Standards

### Figure Captions (Must Include)
- Sample size (n=X seeds)
- Error representation (± SD or 95% CI)
- Statistical test result (p=X, method)
- Hardware/software context

### Tables (Must Include)
- Mean ± SD (or 95% CI)
- Sample size per cell
- Statistical test (p-value, method)
- Practical significance threshold

### Claims (Must Include)
- Quantitative value with uncertainty
- Sample size
- Statistical test (if comparing)
- Reproducibility info (seed, script)

---

## Tier 2 Completion Criteria

### Code Quality
- [ ] All scripts have `--seed` argument for reproducibility
- [ ] All results saved as JSON with metadata (git SHA, timestamp)
- [ ] All plots have 300 DPI, proper captions, error bars
- [ ] All long-running jobs have progress bars

### Statistical Rigor
- [ ] All comparisons use paired tests (not independent t-tests)
- [ ] All p-values accompanied by effect size (Cohen's d or similar)
- [ ] All null results checked with equivalence testing (TOST)
- [ ] All claims have 95% CI or SD reported

### Reproducibility
- [ ] All scripts run from repository root with no manual setup
- [ ] All dependencies in requirements.lock (exact versions)
- [ ] All random operations seeded deterministically
- [ ] All results have SHA-256 checksums

### Documentation
- [ ] All methods cited (≥1 paper per method)
- [ ] All practical thresholds justified (domain literature)
- [ ] All figures have complete captions (ready for paper)
- [ ] All code has docstrings (purpose, inputs, outputs)

---

## Risk Mitigation

### Risk 1: DKL ablation shows PCA+GP ≈ DKL
**Probability**: Medium  
**Impact**: Undermines DKL contribution claim  
**Mitigation**: Report honestly, emphasize interpretability (49 physics correlations)

### Risk 2: Profiling shows conformal calibration is bottleneck
**Probability**: Low  
**Impact**: Weakens Filter-CEI cost argument  
**Mitigation**: Report honestly, reframe as "negligible overhead for safety"

### Risk 3: Filter-CEI Pareto frontier shows no clear optimal point
**Probability**: Medium  
**Impact**: Can't recommend specific keep_fraction  
**Mitigation**: Report curve, let users choose based on risk tolerance

### Risk 4: Time constraints (5 days ambitious for 26 hours work)
**Probability**: High  
**Impact**: May not complete all P0-P4  
**Mitigation**: Prioritize P0-P1 (sharpness, DKL), defer P2-P4 if needed

---

## Daily Checkpoints

### End of Day 1
**Expected**: Sharpness analysis complete  
**Verification**: `experiments/novelty/noise_sensitivity/sharpness_analysis.json` exists  
**Decision**: Proceed to Day 2 if complete, else debug/extend Day 1

### End of Day 2
**Expected**: PCA+GP baseline complete  
**Verification**: `experiments/ablations/dkl_vs_baselines_results.json` exists  
**Decision**: Proceed if DKL shows clear advantage; if not, run more seeds

### End of Day 3
**Expected**: DKL ablation complete  
**Verification**: Updated NOVELTY_FINDING.md with ablation table  
**Decision**: Proceed to profiling if documentation complete

### End of Day 4
**Expected**: Profiling + Filter-CEI complete  
**Verification**: Cost table + Pareto plot exist  
**Decision**: Proceed to metrics if both complete

### End of Day 5
**Expected**: All Tier 2 complete  
**Verification**: Run verification script, check all checkboxes  
**Decision**: If <80% complete, extend 1-2 days; else proceed to paper draft

---

## Output Specifications

### JSON Results Format
```json
{
  "metadata": {
    "script": "analyze_sharpness.py",
    "git_sha": "154e94f",
    "timestamp": "2025-10-10T09:00:00Z",
    "seed": 42,
    "runtime_seconds": 123.4
  },
  "results": {
    "cei_avg_pi_width_K": 45.2,
    "cei_avg_pi_width_sd_K": 3.1,
    "ei_avg_pi_width_K": null,
    "comparison": {
      "test": "paired_t_test",
      "p_value": 0.03,
      "cohens_d": 0.34
    }
  }
}
```

### Plot Specifications
- Format: PNG, 300 DPI, 8×6 inches
- Title: Method + sample size (e.g., "Sharpness Analysis (n=10 seeds)")
- Axes: Bold labels with units
- Error bars: ± SD or 95% CI (state in caption)
- Legend: Upper right or as appropriate
- Annotations: Key findings (e.g., "p=0.03, Cohen's d=0.34")

---

## Tier 2 TODO Checklist

### P0: Sharpness Analysis
- [ ] Implement sharpness metrics computation
- [ ] Compute conditional coverage (3+ categories)
- [ ] Generate sharpness plots (3 figures)
- [ ] Statistical comparison (CEI vs EI)
- [ ] Update NOVELTY_FINDING.md with results

### P1: DKL Ablation
- [ ] Implement PCA+GP baseline
- [ ] Implement random projection+GP baseline
- [ ] Run 4-way comparison (DKL, PCA, Random, GP-raw)
- [ ] Latent dim sweep (4D, 8D, 16D, 32D)
- [ ] Report wall-clock time per method
- [ ] Quantify feature learning contribution
- [ ] Update NOVELTY_FINDING.md with ablation

### P2: Computational Profiling
- [ ] Profile operation-level breakdown
- [ ] Measure scaling (n=100 to 5000)
- [ ] Document hardware specs
- [ ] Generate cost table + scaling plot
- [ ] Update NOVELTY_FINDING.md with profiling

### P3: Filter-CEI Pareto
- [ ] Implement variable keep_fraction
- [ ] Run on 3 noise levels × 5 fractions
- [ ] Compute Pareto frontier
- [ ] Generate Pareto plot
- [ ] Write Algorithm 2 box for paper

### P4: Regret Metrics
- [ ] Implement simple regret computation
- [ ] Implement cumulative regret computation
- [ ] Recompute for all existing results
- [ ] Correlation with domain metric
- [ ] Update NOVELTY_FINDING.md with metrics

---

**Status**: Ready to begin systematic execution  
**Next**: Day 1 (Sharpness Analysis) - 6 hours  
**Expected Grade After Tier 2**: A-

