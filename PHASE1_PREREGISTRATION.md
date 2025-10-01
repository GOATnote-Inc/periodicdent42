# Phase 1 Validation - Pre-Registration Template

**IMPORTANT**: Fill this out BEFORE running any Phase 1 experiments.  
**Purpose**: Prevent p-hacking, improve reproducibility, increase credibility.  
**Platform**: Register on OSF (https://osf.io) before experiment start.

---

## Study Information

### Title
Validation of Reinforcement Learning Advantages in High-Noise Black-Box Optimization

### Authors
- Periodic Labs Team
- [Add names here]

### Date Created
**To be filled**: [Date when registered on OSF]

### Research Question
Does Reinforcement Learning (specifically PPO+ICM) show consistent advantages over 
Bayesian Optimization in high-noise optimization problems across multiple benchmark 
functions?

---

## Hypotheses

### Primary Hypothesis (H1)
**Statement**: PPO+ICM will achieve better optimization performance than standard 
Bayesian Optimization (GP-UCB) at noise level σ≥2.0 on at least 3 out of 5 benchmark 
functions.

**Direction**: One-tailed (RL > BO)

**Success Criterion**: 
- Statistical significance: p < 0.01 (Bonferroni corrected for 5 functions)
- Effect size: Cohen's d > 0.5 (medium effect)
- Robustness: RL better in ≥60% of the 5 benchmark functions

**Falsification Criterion**:
If RL does not show significant advantage on ≥3 functions, we will:
1. Document null result
2. Investigate why preliminary finding didn't replicate
3. Consider alternative explanations

### Secondary Hypotheses (H2-H4)

**H2**: The advantage of RL over BO increases with noise level  
- Test at σ = 0.0, 0.5, 1.0, 1.5, 2.0, 3.0
- Expect RL/BO performance ratio to increase with σ

**H3**: RL outperforms robust BO variants at high noise  
- Compare to heteroscedastic GP and robust BO
- Expect RL advantage even against advanced baselines

**H4**: The noise threshold where RL becomes advantageous is σ ≈ 1.5-2.0  
- Identify crossover point where RL starts outperforming BO
- Validate tentative threshold from preliminary work

### Null Hypothesis (H0)
RL does not consistently outperform BO across benchmark functions at high noise levels.

**If H0 is supported**: We will document this as a null result and publish findings.

---

## Methods

### Benchmark Functions

1. **Branin** (2D)
   - Domain: x₁ ∈ [-5, 10], x₂ ∈ [0, 15]
   - Global optimum: f(x*) = 0.397887
   - Characteristics: 3 global optima, smooth

2. **Ackley** (2D, 5D, 10D)
   - Domain: xᵢ ∈ [-32.768, 32.768]
   - Global optimum: f(x*) = 0
   - Characteristics: Many local optima, difficult for BO

3. **Rastrigin** (2D, 5D, 10D)
   - Domain: xᵢ ∈ [-5.12, 5.12]
   - Global optimum: f(x*) = 0
   - Characteristics: Highly multimodal, regular structure

4. **Rosenbrock** (2D, 5D, 10D)
   - Domain: xᵢ ∈ [-5, 10]
   - Global optimum: f(x*) = 0
   - Characteristics: Long narrow valley, difficult for gradient-free

5. **Hartmann6** (6D)
   - Domain: xᵢ ∈ [0, 1]
   - Global optimum: f(x*) = -3.32237
   - Characteristics: 6 local optima, pharmaceutical relevance

**Rationale**: Mix of dimensions (2D, 5D, 6D, 10D) and characteristics (smooth, multimodal, narrow valleys)

### Noise Model
**Type**: Additive Gaussian noise  
**Levels**: σ ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 3.0}  
**Implementation**: y_observed = f(x) + N(0, σ²)  

**Justification**: 
- σ=0.0: Clean baseline
- σ=0.5-1.0: Typical lab noise
- σ=1.5-2.0: High noise (preliminary RL advantage)
- σ=3.0: Extreme noise

**Note**: We acknowledge this is simplified. Future work should test heteroscedastic 
and non-Gaussian noise.

### Optimization Methods

#### 1. Standard Bayesian Optimization (GP-UCB)
**Implementation**: GPyOpt or BoTorch  
**Hyperparameters**:
- Kernel: RBF + White noise kernel
- Acquisition: Upper Confidence Bound (β=2.0)
- Initialization: 5 random samples
- Max iterations: 100

#### 2. Robust Bayesian Optimization
**Implementation**: BoTorch with robust acquisition  
**Hyperparameters**: [To be specified based on implementation]

#### 3. Heteroscedastic GP
**Implementation**: GPy with HeteroscedasticGaussian likelihood  
**Hyperparameters**: [To be specified based on implementation]

#### 4. PPO+ICM (our RL method)
**Implementation**: src/reasoning/ppo_agent.py  
**Hyperparameters**:
- Learning rate: 3e-4
- PPO clip: 0.2
- ICM weight: 0.1
- Discount factor: 0.99
- [Full hyperparameters to be documented]

#### 5. Random Search
**Implementation**: Uniform random sampling  
**Purpose**: Baseline comparison

**Hyperparameter Commitment**: All hyperparameters will be fixed BEFORE running 
experiments. No tuning based on results.

### Sample Size

**Trials per condition**: n = 30  
**Total conditions**: 5 functions × 6 noise levels × 5 methods = 150 conditions  
**Total experiments**: 150 × 30 = 4,500 experiments  

**Power Analysis**:
- Expected effect size: d = 0.7 (medium-large)
- Desired power: 0.8
- Alpha (corrected): 0.01
- Required n per group: ~26

**Justification**: n=30 provides 80% power to detect medium-large effects at corrected 
alpha level.

### Experimental Protocol

**Randomization**:
1. Random seed generation for all experiments (fixed, logged)
2. Counterbalanced order of conditions
3. Independent random seeds for each trial

**Compute Environment**:
- Hardware: [To be specified]
- Software versions: Python 3.12, PyTorch [version], GPyOpt [version]
- Random seed: [To be logged]

**Data Collection**:
- Raw data saved for every experiment
- Checkpoints every 10% completion
- Metadata: timestamp, seeds, hyperparameters

**Stopping Rules**:
- Max iterations: 100 per experiment
- Early stopping: None (run all 100 iterations)
- Convergence threshold: Not used (to avoid bias)

---

## Analysis Plan

### Primary Outcome Measure
**Best value found** after 100 iterations: y_best = min(y₁, ..., y₁₀₀)

**Secondary Outcome Measures**:
1. Sample efficiency: Iterations to reach 90% of best value
2. Convergence rate: AUC of performance curve
3. Robustness: Standard deviation across 30 trials
4. Failure rate: Proportion of trials that fail to improve

### Statistical Tests

#### Primary Analysis (H1)
**Test**: Paired t-test (RL vs BO) on best value found  
**Correction**: Bonferroni for 5 functions (α = 0.01/5 = 0.002 per function)  
**Effect size**: Cohen's d with 95% confidence interval  

**Success criterion**: 
- p < 0.002 on ≥3 out of 5 functions
- d > 0.5 (medium effect)

#### Secondary Analyses (H2-H4)
**H2 (noise effect)**: Linear regression of RL advantage vs noise level  
**H3 (robust BO)**: Compare RL to robust BO and heteroscedastic GP  
**H4 (threshold)**: Identify crossover point using spline interpolation  

#### Exploratory Analyses
**Allowed**: 
- Function-specific patterns
- Dimensionality effects
- Convergence dynamics

**Not allowed**:
- Changing success criteria based on results
- Selective reporting of functions
- P-hacking or multiple testing without correction

### Visualization Plan
1. Performance curves (mean ± 95% CI) for each function × noise level
2. Heatmap of RL vs BO advantage across conditions
3. Effect size plots with confidence intervals
4. Distribution of best values (violin plots)

### Handling Missing Data
**Protocol**: 
- If experiment crashes: Re-run with same random seed
- If method fails to converge: Record as failure, include in analysis
- Missing data threshold: If >5% missing, investigate and report

### Outlier Handling
**Protocol**: 
- No outlier removal (all data included)
- If extreme values suspected: Run sensitivity analysis with/without
- Document any anomalies

---

## Deviations from Pre-Registration

**To be updated**: Any deviations from this plan will be documented here with justification.

**Allowed deviations** (without compromising validity):
- Hyperparameter bugs (fix and document)
- Implementation errors (fix and re-run)
- Compute failures (restart)

**Not allowed deviations**:
- Changing success criteria
- Dropping functions based on results
- Adding methods that weren't pre-registered

---

## Timeline

**Pre-registration**: [Date - BEFORE experiments start]  
**Experiment start**: [Date]  
**Expected completion**: [Date + 2-3 weeks]  
**Analysis completion**: [Date + 4 weeks]  
**Report completion**: [Date + 6 weeks]  

**Checkpoints**:
- 25% complete: Review preliminary patterns (DO NOT change plan)
- 50% complete: Check for technical issues
- 75% complete: Prepare analysis pipeline
- 100% complete: Run full analysis

---

## Data Sharing

**Commitment**: All raw data and code will be shared publicly after completion.

**Platform**: 
- Data: OSF repository
- Code: GitHub (https://github.com/GOATnote-Inc/periodicdent42)

**Format**: 
- Raw results: JSON
- Processed results: CSV
- Code: Python scripts + environment.yml

**Embargo**: None (immediate release after paper/report completion)

---

## Conflicts of Interest

**Financial**: None  
**Intellectual**: We developed the RL method being tested  

**Mitigation**: 
- Pre-registration commits us to report negative results
- Statistical tests are objective
- All data/code will be public for independent verification

---

## Expected Outcomes & Interpretation

### Scenario 1: H1 Confirmed (RL advantage replicates)
**Interpretation**: RL shows consistent advantage at high noise  
**Next steps**: Proceed to Phase 2 (mechanism studies)  
**Caution**: Still simulation-based, need hardware validation  

### Scenario 2: H1 Partially Confirmed (RL advantage on 1-2 functions)
**Interpretation**: RL advantage may be function-specific  
**Next steps**: Investigate why some functions show advantage, others don't  
**Revision**: Narrow hypothesis to specific function classes  

### Scenario 3: H1 Not Confirmed (No consistent RL advantage)
**Interpretation**: Preliminary finding was likely a statistical fluke  
**Next steps**: Document null result, investigate alternative explanations  
**Publication**: Write "Null Result" paper  
**Pivot**: Focus on other customer pain points  

### Scenario 4: BO Outperforms RL
**Interpretation**: RL hypothesis is wrong  
**Next steps**: Document, investigate why preliminary result was misleading  
**Learning**: Small sample sizes can be very misleading  

---

## Registration

**Pre-registration platform**: Open Science Framework (https://osf.io)  
**Registration date**: [TO BE FILLED BEFORE EXPERIMENTS]  
**OSF project URL**: [TO BE FILLED]  
**Registration DOI**: [TO BE FILLED]  

**Confirmation**: By registering, we commit to:
1. Running experiments exactly as specified
2. Reporting all results (positive and negative)
3. Not changing success criteria based on results
4. Sharing all data and code publicly

---

## Signatures

**Principal Investigator**: [Name] _____________ Date: _______  
**Team Lead**: [Name] _____________ Date: _______  
**Data Analyst**: [Name] _____________ Date: _______  

---

## Notes for Future Us

**Why we're doing this**:
- We had ONE positive result (n=10, Branin, σ=2.0)
- User correctly called out premature "breakthrough" claims
- We need rigorous validation before making any claims
- Pre-registration prevents p-hacking and increases credibility

**What success looks like**:
- RL shows advantage on ≥3/5 functions at high noise
- Effect sizes are substantial (d > 0.5)
- Results are robust across trials

**What failure looks like**:
- RL doesn't replicate
- We document this honestly
- We learn from it and move on

**Most important**: **Scientific integrity > positive results**

---

**Status**: TEMPLATE - Must be filled out and registered on OSF before experiments  
**Last Updated**: October 1, 2025  
**Version**: 1.0

---

*"Pre-registration is insurance against self-deception."*

