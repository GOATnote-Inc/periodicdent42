#!/usr/bin/env python3
"""
Compute paired statistics with 95% CI for DKL vs baselines.

Usage:
    python scripts/paired_stats.py \
        --ref evidence/phase10/tier2_20seeds/results.json \
        --out evidence/phase10/tier2_20seeds/paired_report.md
"""

import argparse
import json
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def compute_paired_stats(dkl_rmse: list, baseline_rmse: list, baseline_name: str) -> dict:
    """
    Compute paired t-test statistics.
    
    Returns dict with:
        - p_value: Paired t-test p-value
        - ci_95: 95% confidence interval for difference
        - mean_diff: Mean difference (DKL - baseline)
        - cohen_d: Effect size
        - significant: Whether p < 0.05
    """
    dkl = np.array(dkl_rmse)
    baseline = np.array(baseline_rmse)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(dkl, baseline)
    
    # Mean difference
    diff = dkl - baseline
    mean_diff = diff.mean()
    
    # 95% CI for difference
    ci_95 = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=stats.sem(diff))
    
    # Effect size (Cohen's d)
    cohen_d = mean_diff / diff.std()
    
    # Improvement percentage
    improvement_pct = (baseline.mean() - dkl.mean()) / baseline.mean() * 100
    
    return {
        'baseline_name': baseline_name,
        'p_value': float(p_value),
        'ci_95_lower': float(ci_95[0]),
        'ci_95_upper': float(ci_95[1]),
        'mean_diff': float(mean_diff),
        'cohen_d': float(cohen_d),
        'improvement_pct': float(improvement_pct),
        'significant': bool(p_value < 0.05),
        'dkl_mean': float(dkl.mean()),
        'dkl_std': float(dkl.std()),
        'baseline_mean': float(baseline.mean()),
        'baseline_std': float(baseline.std())
    }

def generate_report(results_file: Path, output_file: Path):
    """Generate paired statistics report"""
    
    # Load results
    with open(results_file) as f:
        data = json.load(f)
    
    n_seeds = data['n_seeds']
    
    # Extract RMSE histories
    dkl_rmse = [h[-1] for h in data['results']['dkl']['rmse_histories']]
    
    # Initialize report
    report = f"""# Paired Statistics Report

**Date**: {data.get('timestamp', 'N/A')}  
**Seeds**: {n_seeds}  
**Dataset**: UCI Superconductivity (21,263 compounds)

---

## Summary

"""
    
    # Compute paired stats for each baseline
    all_stats = {}
    
    if 'gp' in data['results']:
        gp_rmse = [h[-1] for h in data['results']['gp']['rmse_histories']]
        all_stats['GP'] = compute_paired_stats(dkl_rmse, gp_rmse, 'GP')
    
    if 'random' in data['results']:
        random_rmse = [h[-1] for h in data['results']['random']['rmse_histories']]
        all_stats['Random'] = compute_paired_stats(dkl_rmse, random_rmse, 'Random')
    
    # Add summary table
    report += "| Comparison | ΔRMSE (K) | 95% CI | p-value | Significant | Improvement |\n"
    report += "|------------|-----------|--------|---------|-------------|-------------|\n"
    
    for baseline, stats_dict in all_stats.items():
        sig_mark = "✅" if stats_dict['significant'] else "❌"
        report += f"| DKL vs {baseline} | {stats_dict['mean_diff']:.2f} | "
        report += f"[{stats_dict['ci_95_lower']:.2f}, {stats_dict['ci_95_upper']:.2f}] | "
        report += f"{stats_dict['p_value']:.4f} | {sig_mark} | "
        report += f"{stats_dict['improvement_pct']:.1f}% |\n"
    
    report += "\n---\n\n"
    
    # Detailed statistics
    report += "## Detailed Statistics\n\n"
    
    for baseline, stats_dict in all_stats.items():
        report += f"### DKL vs {baseline}\n\n"
        report += f"**Paired t-test**: t={stats_dict['mean_diff']/stats.sem(np.array(dkl_rmse) - np.array(data['results'][baseline.lower()]['rmse_histories'][0][-1])):.3f}, "
        report += f"p={stats_dict['p_value']:.4f}\n\n"
        
        report += f"**Mean Difference**: {stats_dict['mean_diff']:.2f} K\n\n"
        report += f"**95% Confidence Interval**: [{stats_dict['ci_95_lower']:.2f}, {stats_dict['ci_95_upper']:.2f}] K\n\n"
        report += f"**Effect Size (Cohen's d)**: {stats_dict['cohen_d']:.2f}\n\n"
        
        # Interpretation
        if abs(stats_dict['cohen_d']) >= 0.8:
            effect_interp = "Large"
        elif abs(stats_dict['cohen_d']) >= 0.5:
            effect_interp = "Medium"
        else:
            effect_interp = "Small"
        
        report += f"**Effect Interpretation**: {effect_interp}\n\n"
        
        # Significance interpretation
        if stats_dict['significant']:
            if stats_dict['ci_95_upper'] < 0:
                report += f"✅ **Significant improvement**: DKL is {stats_dict['improvement_pct']:.1f}% better than {baseline} (p<0.05)\n\n"
            else:
                report += f"✅ **Statistically significant difference** (p<0.05), but CI includes zero\n\n"
        else:
            report += f"❌ **Not statistically significant** (p≥0.05)\n\n"
        
        # Raw statistics
        report += f"**DKL**: {stats_dict['dkl_mean']:.2f} ± {stats_dict['dkl_std']:.2f} K\n\n"
        report += f"**{baseline}**: {stats_dict['baseline_mean']:.2f} ± {stats_dict['baseline_std']:.2f} K\n\n"
        
        report += "---\n\n"
    
    # Acceptance criteria
    report += "## Acceptance Criteria (2025 Standards)\n\n"
    report += "| Criterion | Target | Status |\n"
    report += "|-----------|--------|--------|\n"
    report += f"| Seeds | ≥ 20 | {'✅' if n_seeds >= 20 else '❌'} ({n_seeds}) |\n"
    
    any_significant = any(s['significant'] for s in all_stats.values())
    report += f"| p-value | < 0.05 | {'✅' if any_significant else '❌'} |\n"
    
    gp_excludes_zero = all_stats.get('GP', {}).get('ci_95_upper', 1) < 0
    report += f"| 95% CI excludes zero | Yes | {'✅' if gp_excludes_zero else '❌'} |\n"
    
    # Recommendation
    report += "\n---\n\n## Recommendation\n\n"
    
    if n_seeds >= 20 and any_significant and gp_excludes_zero:
        report += "✅ **PASS**: Results meet 2025 reproducibility standards.\n"
        report += "- Statistical power sufficient (n≥20)\n"
        report += "- Significant improvement demonstrated (p<0.05)\n"
        report += "- Effect size quantified with non-overlapping CI\n\n"
        report += "**Ready for publication and production deployment.**\n"
    elif n_seeds < 20:
        report += "⚠️ **INSUFFICIENT POWER**: Need at least 20 seeds for robust claims.\n"
    elif not any_significant:
        report += "❌ **NOT SIGNIFICANT**: No statistically significant improvement demonstrated.\n"
    elif not gp_excludes_zero:
        report += "⚠️ **BORDERLINE**: Significant p-value but CI includes zero.\n"
        report += "Consider increasing sample size or using bootstrap CI.\n"
    
    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report)
    
    print(f"✅ Report saved to: {output_file}")
    
    # Generate plot
    plot_file = output_file.with_suffix('.png')
    generate_plot(data, all_stats, plot_file)
    
    # Save JSON for programmatic access
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"✅ Statistics saved to: {json_file}")
    print(f"✅ Plot saved to: {plot_file}")
    
    return all_stats

def generate_plot(data, all_stats, output_file):
    """Generate visualization of paired differences"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Box plot of RMSE distributions
    ax1 = axes[0]
    
    dkl_rmse = [h[-1] for h in data['results']['dkl']['rmse_histories']]
    
    plot_data = [dkl_rmse]
    labels = ['DKL']
    
    if 'gp' in data['results']:
        gp_rmse = [h[-1] for h in data['results']['gp']['rmse_histories']]
        plot_data.append(gp_rmse)
        labels.append('GP')
    
    if 'random' in data['results']:
        random_rmse = [h[-1] for h in data['results']['random']['rmse_histories']]
        plot_data.append(random_rmse)
        labels.append('Random')
    
    bp = ax1.boxplot(plot_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#4CAF50', '#2196F3', '#FF9800']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('RMSE (K)', fontsize=12)
    ax1.set_title(f'RMSE Distribution ({data["n_seeds"]} seeds)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Paired differences with CI
    ax2 = axes[1]
    
    baselines = list(all_stats.keys())
    means = [all_stats[b]['mean_diff'] for b in baselines]
    ci_lower = [all_stats[b]['ci_95_lower'] for b in baselines]
    ci_upper = [all_stats[b]['ci_95_upper'] for b in baselines]
    
    y_pos = np.arange(len(baselines))
    
    colors = ['green' if all_stats[b]['significant'] else 'gray' for b in baselines]
    
    ax2.barh(y_pos, means, color=colors, alpha=0.7, edgecolor='black')
    
    # Error bars for CI
    errors = [[m - l for m, l in zip(means, ci_lower)],
              [u - m for m, u in zip(means, ci_upper)]]
    ax2.errorbar(means, y_pos, xerr=errors, fmt='none', ecolor='black', capsize=5, linewidth=2)
    
    # Zero line
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'DKL vs {b}' for b in baselines])
    ax2.set_xlabel('ΔRMSE (K) [DKL - Baseline]', fontsize=12)
    ax2.set_title('Paired Differences with 95% CI', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compute paired statistics')
    parser.add_argument('--ref', type=Path, required=True, help='Results JSON file')
    parser.add_argument('--out', type=Path, required=True, help='Output markdown file')
    args = parser.parse_args()
    
    if not args.ref.exists():
        print(f"❌ Results file not found: {args.ref}")
        return 1
    
    print(f"📊 Computing paired statistics from: {args.ref}")
    stats = generate_report(args.ref, args.out)
    
    print("\n" + "="*70)
    print("✅ PAIRED STATISTICS COMPLETE")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    exit(main())

