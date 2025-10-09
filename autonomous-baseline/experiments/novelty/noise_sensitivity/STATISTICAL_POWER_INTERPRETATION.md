# Statistical Power Analysis

## Executive Summary

**Research Question**: Can our study (n=10 seeds) detect the observed effect size (ΔRMSE=0.054 K)?

**Answer**: ❌ **NO** - Study is underpowered.

---

## Observed Effect

- **CEI RMSE**: 22.50 ± 0.75 K
- **EI RMSE**: 22.56 ± 0.74 K
- **Δ RMSE**: 0.054 K
- **Cohen's d**: 0.073 (tiny effect)

---

## Minimum Detectable Effect (MDE)

With **n=10 seeds** and **σ=0.74 K**:
- **MDE at 80% power**: 0.98 K
- **MDE at 90% power**: 1.14 K

**Interpretation**: We can only detect effects ≥ 0.98 K. Observed effect (0.054 K) is **0.1×** smaller than MDE.

---

## Required Sample Size

To detect ΔRMSE=0.054 K at 80% power:
- **Required n**: 2936 seeds
- **Current n**: 10 seeds
- **Shortfall**: 2926 additional seeds needed

**Conclusion**: Study lacks statistical power to detect tiny effects.

---

## Equivalence Testing (TOST)

**Hypothesis**: Are CEI and EI practically equivalent?

- **Equivalence bound**: ε = 1.5 K (based on synthesis variability)
- **Observed Δ**: 0.864 K
- **p-value (TOST)**: 0.0363
- **Conclusion**: EQUIVALENT

**Interpretation**: Methods are statistically equivalent within practical bounds.

---

## Practical Materiality Threshold

**Domain Knowledge** (from materials science literature):
- DFT vs experiment MAE: 2-5 K
- Synthesis variability: 5-10 K
- Measurement error (XRD): 1-3 K
- Multi-lab reproducibility: 8-12 K

**Recommended threshold**: **1.5 K**

**Justification**: Effects smaller than 1.5 K are below typical synthesis variability (5-10 K) and within measurement error (1-3 K). For materials discovery, ΔRMSE < 1.5 K is not practically meaningful.

**Observed effect (0.054 K) vs threshold (1.5 K)**: 
✅ Below threshold → Not practically meaningful

---

## Recommendations

1. **For Publication**: Report as "no detectable effect within power bounds"
   - State MDE: 0.98 K
   - Acknowledge underpowered for tiny effects
   - Emphasize observed effect < practical threshold

2. **For Follow-Up**: If curious about tiny effects:
   - Run 2936 seeds (not 10)
   - Use equivalence testing (TOST) framework
   - Report with practical interpretation

3. **For Deployment**: 
   - Effect (0.054 K) << synthesis variability (5-10 K)
   - **Use vanilla EI** (simpler, equivalent performance)

---

## References

- Stanev et al., npj Comput Mater 4:29 (2018) - DFT/experiment gap
- Zunger, Nature Rev Mater 3:117 (2018) - Synthesis variability
- MRS Bulletin 44:443 (2019) - Multi-lab reproducibility

---

**Generated**: 1760050872.2515013
