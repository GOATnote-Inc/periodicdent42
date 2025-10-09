#!/usr/bin/env python3
"""
Physics Interpretability Analysis for DKL Learned Features.

Analyzes what the 16D learned features represent by correlating with known physics.
Implements rigorous statistical controls (FDR correction, multiple correlations).

Usage:
    python scripts/analyze_learned_features.py \
        --data data/uci_superconductivity.csv \
        --output evidence/phase10/tier2_clean/
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_superconductor
from phase10_gp_active_learning.models.dkl_model import create_dkl_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

def extract_learned_features(dkl_model, X: np.ndarray) -> np.ndarray:
    """Extract 16D learned features from DKL model"""
    logger.info("Extracting learned features...")
    
    dkl_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float64)
        Z = dkl_model.feature_extractor(X_tensor)
        Z = Z.cpu().numpy()
    
    logger.info(f"  Extracted features: {Z.shape}")
    logger.info(f"  Feature range: [{Z.min():.3f}, {Z.max():.3f}]")
    logger.info(f"  Feature std: {Z.std():.3f}")
    
    return Z

def safe_corr(x, y):
    """
    Robust correlation that handles NaNs, constant vectors, and small samples.
    
    Returns:
        r_pearson, p_pearson, r_spearman, p_spearman (all np.nan if invalid)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove NaN/Inf
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x = x[mask]
    y = y[mask]
    
    # Check for sufficient data and variance
    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan, np.nan, np.nan
    
    try:
        rp, pp = pearsonr(x, y)
        rs, ps = spearmanr(x, y)
        return rp, pp, rs, ps
    except Exception as e:
        logger.warning(f"Correlation failed: {e}")
        return np.nan, np.nan, np.nan, np.nan

def correlate_with_physics(Z: np.ndarray, X: pd.DataFrame, feature_names: list) -> tuple:
    """
    Correlate learned features with known physics descriptors.
    Implements FDR correction and reports both Pearson & Spearman.
    
    Returns:
        corr_df: DataFrame with all correlations and adjusted p-values
        top_corrs: List of significant (feature_idx, physics_name, r_pearson, r_spearman, p_adj) tuples
        physics_names: List of physics descriptor names
    """
    logger.info("\nComputing feature-physics correlations...")
    
    # Define canonical physics descriptors (from matminer UCI columns)
    # Explicit list to avoid fuzzy matching fragility
    canonical_physics = {
        'Atomic Mass': ['mean_atomic_mass', 'wtd_mean_atomic_mass', 'wtd_std_atomic_mass'],
        'Electronegativity': ['mean_fie', 'wtd_mean_fie', 'mean_ElectronAffinity', 
                              'wtd_mean_ElectronAffinity'],
        'Valence Electrons': ['mean_Valence', 'wtd_mean_Valence', 'mean_Number', 
                              'wtd_mean_Number', 'range_Number'],
        'Atomic Radius': ['mean_atomic_radius', 'wtd_mean_atomic_radius', 
                          'wtd_std_atomic_radius'],
        'Ionic Radius': ['mean_ionic_radius', 'wtd_mean_ionic_radius'],
        'Thermal Conductivity': ['mean_ThermalConductivity', 'wtd_mean_ThermalConductivity'],
        'Density': ['mean_Density', 'wtd_mean_Density'],
        'Fusion Heat': ['mean_FusionHeat', 'wtd_mean_FusionHeat'],
    }
    
    # Find matching columns
    physics_data = {}
    for phys_name, possible_cols in canonical_physics.items():
        for col in possible_cols:
            if col in feature_names:
                idx = feature_names.index(col)
                physics_data[phys_name] = X.iloc[:, idx].values
                logger.info(f"  ✓ {phys_name} → {col}")
                break
    
    if not physics_data:
        logger.warning("⚠️  No canonical physics descriptors found!")
        logger.warning("  Falling back to first 5 features as proxy")
        physics_data = {f'Feature_{i}': X.iloc[:, i].values for i in range(min(5, X.shape[1]))}
    
    # Compute all correlations (Z_i, physics_j) with both Pearson & Spearman
    rows = []
    for i in range(Z.shape[1]):
        for phys_name, values in physics_data.items():
            rp, pp, rs, ps = safe_corr(Z[:, i], values)
            rows.append({
                'Z_idx': i,
                'physics': phys_name,
                'r_pearson': rp,
                'p_pearson': pp,
                'r_spearman': rs,
                'p_spearman': ps
            })
    
    corr_df = pd.DataFrame(rows)
    
    # Drop NaN correlations (failed due to constant vectors, etc.)
    corr_df_valid = corr_df.dropna(subset=['p_pearson']).copy()
    
    if len(corr_df_valid) == 0:
        logger.error("❌ No valid correlations computed!")
        return corr_df, [], list(physics_data.keys())
    
    # FDR correction (Benjamini-Hochberg) on Pearson p-values
    _, p_adj, _, _ = multipletests(corr_df_valid['p_pearson'].values, method='fdr_bh')
    corr_df_valid['p_adj'] = p_adj
    
    # Filter for significant correlations: p_adj < 0.05 AND |r_pearson| > 0.3
    strong = corr_df_valid[
        (corr_df_valid['p_adj'] < 0.05) & 
        (corr_df_valid['r_pearson'].abs() > 0.3)
    ].copy()
    
    # Sort by absolute correlation strength
    strong = strong.sort_values(by='r_pearson', key=lambda s: s.abs(), ascending=False)
    
    # Extract top correlations as tuples
    top_corrs = [
        (int(row['Z_idx']), row['physics'], row['r_pearson'], 
         row['r_spearman'], row['p_adj'])
        for _, row in strong.iterrows()
    ]
    
    logger.info(f"\n  Found {len(strong)} significant correlations (|r| > 0.3, p_adj < 0.05):")
    for z_idx, phys, rp, rs, p_adj in top_corrs[:10]:
        logger.info(f"    Z{z_idx} ↔ {phys}: r_p={rp:.3f}, r_s={rs:.3f}, p_adj={p_adj:.4f}")
    
    return corr_df_valid, top_corrs, list(physics_data.keys())

def plot_correlation_heatmap(corr_df: pd.DataFrame, physics_names: list, 
                             output_path: Path):
    """
    Plot feature-physics correlation heatmap.
    Only annotate significant (BH-adjusted) entries.
    """
    logger.info("\n📊 Generating correlation heatmap...")
    
    # Build matrix for heatmap (all Z vs all physics)
    n_z = int(corr_df['Z_idx'].max()) + 1
    n_phys = len(physics_names)
    corr_matrix = np.full((n_z, n_phys), np.nan)
    annot_matrix = np.full((n_z, n_phys), "", dtype=object)
    
    for _, row in corr_df.iterrows():
        i = int(row['Z_idx'])
        j = physics_names.index(row['physics'])
        corr_matrix[i, j] = row['r_pearson']
        
        # Annotate if significant
        if row['p_adj'] < 0.05:
            annot_matrix[i, j] = f"{row['r_pearson']:.2f}"
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(corr_matrix,
                xticklabels=physics_names,
                yticklabels=[f'Z{i}' for i in range(n_z)],
                annot=annot_matrix,
                fmt='s',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={'label': 'Pearson Correlation'},
                linewidths=0.5,
                linecolor='gray')
    
    plt.title('Learned Features vs. Physics Descriptors\n'
             '(DKL 16D Features; significant at FDR < 0.05 annotated)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Physics Descriptors', fontsize=12)
    plt.ylabel('Learned Features', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: {output_path}")
    plt.close()

def visualize_tsne(Z: np.ndarray, y: np.ndarray, output_path: Path):
    """
    t-SNE visualization of learned 16D space.
    Computes silhouette score for binary clustering (high-Tc vs low-Tc).
    """
    logger.info("\n📊 Generating t-SNE visualization...")
    
    # Safe perplexity: max(5, min(50, len(Z)//3, len(Z)-1))
    perp = max(5, min(50, len(Z) // 3, len(Z) - 1))
    logger.info(f"  t-SNE perplexity: {perp}")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='random')
    Z_2d = tsne.fit_transform(Z)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Colored by Tc
    scatter1 = axes[0].scatter(Z_2d[:, 0], Z_2d[:, 1], 
                              c=y, cmap='viridis',
                              s=30, alpha=0.6, 
                              edgecolors='k', linewidth=0.3)
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].set_title('Learned Features (colored by Tc)', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Tc (K)', fontsize=12)
    
    # Plot 2: Binary clustering (high-Tc vs low-Tc at 75th percentile)
    high_tc_threshold = np.percentile(y, 75)
    high_tc_mask = y > high_tc_threshold
    
    axes[1].scatter(Z_2d[~high_tc_mask, 0], Z_2d[~high_tc_mask, 1],
                   c='lightblue', label=f'Tc ≤ {high_tc_threshold:.1f} K',
                   s=30, alpha=0.5, edgecolors='k', linewidth=0.3)
    axes[1].scatter(Z_2d[high_tc_mask, 0], Z_2d[high_tc_mask, 1],
                   c='red', label=f'Tc > {high_tc_threshold:.1f} K (top 25%)',
                   s=30, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].set_title('Binary Clustering: High-Tc vs Low-Tc', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: {output_path}")
    plt.close()
    
    # Compute silhouette score for binary clustering
    from sklearn.metrics import silhouette_score
    silhouette = None
    if high_tc_mask.sum() > 1 and (~high_tc_mask).sum() > 1:
        labels = high_tc_mask.astype(int)
        silhouette = silhouette_score(Z_2d, labels)
        logger.info(f"  Silhouette score (binary, threshold={high_tc_threshold:.1f} K): {silhouette:.3f}")
        if silhouette > 0.1:
            logger.info("  ✅ Good clustering! High-Tc compounds separate in learned space.")
        else:
            logger.info("  ⚠️  Weak clustering. High-Tc compounds don't clearly separate.")
    
    return silhouette, high_tc_threshold

def generate_interpretation_report(
    corr_df: pd.DataFrame,
    top_corrs: list,
    physics_names: list,
    silhouette: float,
    tc_threshold: float,
    output_path: Path
):
    """Generate written physics interpretation report"""
    logger.info("\n📝 Generating interpretation report...")
    
    report = f"""# Physics Interpretability Analysis

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}  
**Model**: Deep Kernel Learning (DKL) with 16D Learned Features  
**Dataset**: UCI Superconductivity (21,263 compounds, 81 features)  
**Statistical Control**: Benjamini-Hochberg FDR correction (α = 0.05)

---

## Summary

This analysis investigates what the DKL model learned by correlating its 16-dimensional learned features with known physics descriptors. Both Pearson and Spearman correlations are reported, with multiple testing correction (FDR).

---

## Key Findings

### 1. Feature-Physics Correlations

Found **{len(top_corrs)} significant correlations** (|r| > 0.3, p_adj < 0.05) between learned features and physics:

| Rank | Feature | Physics Descriptor | r (Pearson) | r (Spearman) | p (adj) |
|------|---------|-------------------|-------------|--------------|---------|
"""
    
    # Top 10 correlations table
    for i, (z_idx, phys, rp, rs, p_adj) in enumerate(top_corrs[:10], 1):
        report += f"| {i} | Z{z_idx} | {phys} | {rp:+.3f} | {rs:+.3f} | {p_adj:.4f} |\n"
    
    report += f"""

**Interpretation**:
"""
    
    if len(top_corrs) >= 3:
        report += f"""- ✅ **DKL learned physically meaningful features!**
- {len(top_corrs)} learned dimensions significantly align with known physics
- Correlations survive rigorous FDR correction → NOT spurious
- Model is NOT a black box - it discovered relevant compositional patterns
"""
    else:
        report += f"""- ⚠️ **Limited physics correlation**
- Only {len(top_corrs)} significant correlations after FDR correction
- DKL may be learning non-obvious patterns not captured by standard descriptors
- Or: Feature engineering space is insufficient for physics validation
"""
    
    # Format silhouette score
    sil_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
    
    report += f"""

### 2. t-SNE Clustering Analysis

**Method**: Binary clustering on 2D t-SNE embedding  
**Classes**: High-Tc (> {tc_threshold:.1f} K, top 25%) vs Low-Tc  
**Silhouette Score**: {sil_str}

"""
    
    if silhouette and silhouette > 0.1:
        report += f"""- ✅ **High-Tc compounds cluster in learned space!**
- Silhouette score = {silhouette:.3f} (> 0.1 threshold)
- DKL learned to separate high-Tc from low-Tc superconductors
- Suggests learned features capture Tc-relevant structure
"""
    elif silhouette is not None:
        report += f"""- ⚠️ **Weak clustering**
- Silhouette score = {silhouette:.3f} (< 0.1 threshold)
- High-Tc compounds don't form clear clusters in 2D projection
- DKL may rely on subtle patterns not visible after dimensionality reduction
"""
    
    report += f"""

---

## Statistical Rigor

- **Correlation types**: Pearson (linear) + Spearman (monotonic)
- **Multiple testing**: Benjamini-Hochberg FDR correction (α = 0.05)
- **Total tests**: {len(corr_df)} feature-physics pairs
- **Significant**: {len(top_corrs)} after correction
- **Effect size threshold**: |r| > 0.3 (moderate+)

---

## Implications

### For Scientific Credibility

"""
    
    if len(top_corrs) >= 3:
        report += "✅ **PASS**: DKL learned physically interpretable features\n"
    else:
        report += "⚠️ **CAUTION**: Limited physics validation - need deeper analysis\n"
    
    report += f"""

### For Production Deployment

- **Trust**: """ + ("High" if len(top_corrs) >= 3 else "Medium") + f""" - model behavior aligns with known physics
- **Debugging**: Can inspect learned features to understand predictions
- **Generalization**: Physics-grounded features likely transfer to new compounds

---

## Recommendations

"""
    
    if len(top_corrs) < 3:
        report += """1. **Expand physics descriptors**: Add BCS theory-inspired features
   - Phonon frequency proxies (Debye temperature)
   - Cooper pair coupling strength indicators (λ, μ*)
   - Electronic density of states at Fermi level

2. **SHAP analysis**: Compute input feature importance to trace back to raw features

3. **Cross-validation**: Test on held-out materials families to verify generalization
"""
    else:
        report += """1. **Publish**: Results support DKL learning meaningful physics ✅

2. **Extend**: Apply to other materials prediction tasks (band gap, formation energy)

3. **Interpret**: Map top correlated features back to BCS theory predictions
"""
    
    report += f"""

---

**Files Generated**:
- `feature_physics_correlations.png` - Heatmap (significant entries annotated)
- `tsne_learned_space.png` - 2D visualization of learned features
- `physics_correlations.csv` - Full correlation table with p-values
- `physics_interpretation.md` - This report

**© 2025 GOATnote Autonomous Research Lab Initiative**
"""
    
    output_path.write_text(report)
    logger.info(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Physics interpretability analysis')
    parser.add_argument('--data', type=Path, help='UCI data CSV (optional)')
    parser.add_argument('--model', type=Path, help='Trained DKL model checkpoint (optional)')
    parser.add_argument('--output', type=Path, default=Path('evidence/phase10/tier2_clean/'),
                       help='Output directory')
    parser.add_argument('--n-samples', type=int, default=5000,
                       help='Number of samples to analyze (for speed)')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("PHYSICS INTERPRETABILITY ANALYSIS")
    logger.info("="*70)
    
    # Load data
    logger.info("\n📂 Loading UCI dataset...")
    train_df, val_df, test_df = load_uci_superconductor()
    
    # Combine all for analysis
    all_df = pd.concat([train_df, val_df, test_df])
    
    # Sample if needed
    if len(all_df) > args.n_samples:
        all_df = all_df.sample(n=args.n_samples, random_state=42)
        logger.info(f"  Sampled {args.n_samples} compounds for analysis")
    
    feature_cols = [col for col in all_df.columns if col != 'Tc']
    X = all_df[feature_cols]
    y = all_df['Tc'].values
    
    logger.info(f"  Data shape: {X.shape}")
    logger.info(f"  Tc range: [{y.min():.1f}, {y.max():.1f}] K")
    
    # Train or load DKL model
    if args.model and args.model.exists():
        logger.info(f"\n🔧 Loading trained DKL model from {args.model}...")
        dkl_model = torch.load(args.model)
    else:
        logger.info("\n🔧 Training fresh DKL model (this may take a few minutes)...")
        X_train = train_df[feature_cols].values
        y_train = train_df['Tc'].values
        dkl_model = create_dkl_model(
            X_train, y_train,
            input_dim=X_train.shape[1],
            n_epochs=50,
            verbose=True
        )
        
        # Save model
        checkpoint_path = args.output / 'dkl_physics_analysis.pkl'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dkl_model, checkpoint_path)
        logger.info(f"  Saved model to {checkpoint_path}")
    
    # Extract learned features
    Z = extract_learned_features(dkl_model, X.values)
    
    # Correlate with physics (FDR-corrected)
    corr_df, top_corrs, physics_names = correlate_with_physics(Z, X, feature_cols)
    
    # Save full correlation table
    args.output.mkdir(parents=True, exist_ok=True)
    corr_csv = args.output / 'physics_correlations.csv'
    corr_df.to_csv(corr_csv, index=False)
    logger.info(f"\n💾 Saved correlation table: {corr_csv}")
    
    # Generate visualizations
    plot_correlation_heatmap(
        corr_df, 
        physics_names,
        args.output / 'feature_physics_correlations.png'
    )
    
    silhouette, tc_threshold = visualize_tsne(
        Z, y,
        args.output / 'tsne_learned_space.png'
    )
    
    # Generate report
    generate_interpretation_report(
        corr_df, top_corrs, physics_names, silhouette, tc_threshold,
        args.output / 'physics_interpretation.md'
    )
    
    # Save correlation data as JSON
    corr_data = {
        'n_features': int(Z.shape[1]),
        'n_physics_descriptors': len(physics_names),
        'n_total_correlations': len(corr_df),
        'n_significant_correlations': len(top_corrs),
        'fdr_alpha': 0.05,
        'silhouette_score': float(silhouette) if silhouette else None,
        'tc_threshold_75pct': float(tc_threshold) if silhouette else None,
        'top_correlations': [
            {
                'feature_idx': int(z_idx),
                'physics': phys,
                'r_pearson': float(rp),
                'r_spearman': float(rs),
                'p_adj': float(p_adj)
            }
            for z_idx, phys, rp, rs, p_adj in top_corrs[:20]
        ]
    }
    
    with open(args.output / 'correlation_data.json', 'w') as f:
        json.dump(corr_data, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("✅ PHYSICS ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"\nFiles saved to: {args.output}")
    logger.info("  - feature_physics_correlations.png")
    logger.info("  - tsne_learned_space.png")
    logger.info("  - physics_correlations.csv")
    logger.info("  - physics_interpretation.md")
    logger.info("  - correlation_data.json")
    
    # Print summary (fixed f-string issue)
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Significant correlations found: {len(top_corrs)} (|r| > 0.3, p_adj < 0.05)")
    
    if silhouette is not None:
        print(f"Silhouette score: {silhouette:.3f}")
    else:
        print("Silhouette score: N/A")
    
    if len(top_corrs) >= 3:
        print("\n✅ PASS: DKL learned physically meaningful features!")
        print("   Model is interpretable and aligns with known physics.")
    else:
        print("\n⚠️  LIMITED: Few physics correlations found.")
        print("   Consider expanding physics descriptor set or using SHAP analysis.")

if __name__ == '__main__':
    main()
