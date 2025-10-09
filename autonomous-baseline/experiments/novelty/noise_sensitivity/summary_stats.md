# Noise Sensitivity Results Summary

| σ (K) | Method | RMSE (K) | Regret (K) | Coverage@90 | p-value | Significant? |
|-------|--------|----------|------------|-------------|---------|--------------|
| 0 | Vanilla EI | 22.56 | 82.11 | N/A | — | — |
| 0 | Conformal-EI (Locally Adaptive) | 22.50 | 80.25 | 0.900 | 0.3600 | ❌ |
| 2 | Vanilla EI | 22.46 | 88.32 | N/A | — | — |
| 2 | Conformal-EI (Locally Adaptive) | 22.51 | 87.38 | 0.900 | 0.3911 | ❌ |
| 5 | Vanilla EI | 23.01 | 95.38 | N/A | — | — |
| 5 | Conformal-EI (Locally Adaptive) | 22.96 | 95.62 | 0.900 | 0.5191 | ❌ |
| 10 | Vanilla EI | 24.90 | 100.67 | N/A | — | — |
| 10 | Conformal-EI (Locally Adaptive) | 24.87 | 99.13 | 0.900 | 0.1098 | ❌ |
| 20 | Vanilla EI | 31.32 | 114.21 | N/A | — | — |
| 20 | Conformal-EI (Locally Adaptive) | 31.31 | 111.96 | 0.901 | 0.1871 | ❌ |
| 50 | Vanilla EI | 58.34 | 217.86 | N/A | — | — |
| 50 | Conformal-EI (Locally Adaptive) | 58.21 | 220.29 | 0.900 | 0.5862 | ❌ |

**Legend**: ✅ Significant (p < 0.05) | ❌ Not significant

**σ_critical = NOT FOUND** (no significant difference at any tested noise level)