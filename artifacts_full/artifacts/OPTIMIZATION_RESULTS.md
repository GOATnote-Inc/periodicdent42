# Fixed-Shape Performance Analysis (S=512)

**Statistical Comparison with Bootstrap Confidence Intervals**

---

## Configuration

- Batch Size (B): 32
- Attention Heads (H): 8
- Sequence Length (S): 512
- Head Dimension (D): 64
- Precision: FP16 (TF32 disabled)
- Sample Size: N=100

## PyTorch SDPA (FlashAttention-2)

**Median**: 0.3195 ms
**95% CI**: [0.3185, 0.3210] ms
**Mean**: 0.3287 ms
**Std Dev**: 0.0292 ms

## PyTorch SDPA (No Custom Kernel Speedup)

**Median**: 0.3195 ms
**95% CI**: [0.3185, 0.3210] ms
**Mean**: 0.3287 ms
**Std Dev**: 0.0292 ms

## Statistical Comparison

➡️ **No significant difference** (+0.0%)

### Effect Sizes
- **Hedges' g**: 0.000 (negligible)
- **Cliff's Delta**: 0.000 (negligible)

### Significance Testing
- **CIs Overlap**: True
- **Mann-Whitney U p-value**: 1.0000
- **Statistically Significant**: False

## 📝 Publication-Ready Statement

Baseline achieved 0.3195 ms (95% CI: [0.3185, 0.3210]) vs. candidate 0.3195 ms (95% CI: [0.3185, 0.3210]), representing 1.000× speedup (+0.0%). Effect size (Hedges' g = 0.000) indicates negligible effect. Difference is not statistically significant (p=1.0000).

## Interpretation

**No meaningful performance difference detected**. 
The 0.0% difference is within measurement noise. 
Small effect size (Hedges' g = 0.00) confirms this. 
**Baseline configuration is already optimal** for this workload.

## 🔬 Reproducibility

- **Environment**: Locked (TF32 disabled, deterministic algorithms enabled)
- **Bootstrap Method**: 10,000 resamples with seed=42
- **Raw Data**: Available in `.npy` files for reanalysis
- **Statistical Tests**: Non-parametric (Mann-Whitney U, Cliff's Delta)
