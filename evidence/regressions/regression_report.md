# Regression Detection Report

**Timestamp:** 2025-10-08T02:18:42.421526+00:00
**Git SHA:** abc1008
**CI Run ID:** run_008

## ❌ Status: REGRESSION DETECTED

## Regressions

| Metric | Baseline μ±σ (EWMA) | Current | Δ | z | PH | Status |
|--------|---------------------|---------|---|---|----|--------|
| coverage | 0.8700±0.0120 (0.8708) | 0.7000 | -0.1700 | -14.22 |  | ❌ Fail |
| ece | 0.1200±0.0120 (0.1192) | 0.2500 | +0.1300 | +10.88 |  | ❌ Fail |
| brier | 0.1000±0.0120 (0.0992) | 0.1800 | +0.0800 | +6.69 |  | ❌ Fail |
| entropy_delta_mean | 0.0500±0.0120 (0.0492) | 0.1200 | +0.0700 | +5.86 |  | ❌ Fail |

## Suggested Actions

1. **Check dataset drift**: Verify `data_contracts.yaml` checksums match
2. **Check seed sensitivity**: Re-run with `--seed 42` for reproducibility
3. **Review recent changes**: Check files modified in this commit
4. **Consider waiver**: If intentional, add to `GOVERNANCE_CHANGE_ACCEPT.yml`
