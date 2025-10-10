"""
Analyze DKL Ablation Results

Processes ablation results to create publication-ready figures and statistical analysis.

Usage:
    python scripts/analyze_dkl_ablation.py --results experiments/ablations/dkl_ablation_results_real.json

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
License: MIT
"""

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analyze_ablation")

# Publication-quality plot settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

COLORS = {
    'dkl': '#2E86AB',
    'pca_gp': '#A23B72',
    'random_gp': '#F18F01',
    'gp_raw': '#C73E1D'
}

LABELS = {
    'dkl': 'DKL (learned 16D)',
    'pca_gp': 'PCA+GP (16D)',
    'random_gp': 'Random+GP (16D)',
    'gp_raw': 'GP (raw 81D)'
}


def plot_rmse_comparison(results: Dict, output_dir: Path):
    """Plot RMSE comparison across methods"""
    logger.info("Generating RMSE comparison plot...")
    
    methods = ['dkl', 'pca_gp', 'random_gp', 'gp_raw']
    rmse_means = []
    rmse_stds = []
    
    for method in methods:
        stats = results[method]['statistics']
        rmse_means.append(stats['rmse_mean'])
        rmse_stds.append(stats['rmse_std'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, rmse_means, yerr=rmse_stds, capsize=5,
                   color=[COLORS[m] for m in methods],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel('Final RMSE (K)', fontweight='bold')
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_title(f'DKL Ablation: Feature Learning Contribution (n={results["dkl"]["statistics"]["n_seeds"]} seeds)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Add significance markers
    dkl_mean = rmse_means[0]
    for i, method in enumerate(methods[1:], start=1):
        comparison = results[method]['comparison_vs_dkl']
        if comparison['significant']:
            # Add star for significant difference
            y_pos = max(rmse_means[i] + rmse_stds[i], dkl_mean + rmse_stds[0]) + 1.5
            ax.plot([0, i], [y_pos, y_pos], 'k-', linewidth=1)
            ax.text((0 + i) / 2, y_pos + 0.5, f"p={comparison['p_value']:.3f}", 
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'dkl_ablation_rmse_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved: {output_path}")
    plt.close()


def plot_training_time(results: Dict, output_dir: Path):
    """Plot training time comparison"""
    logger.info("Generating training time plot...")
    
    methods = ['dkl', 'pca_gp', 'random_gp', 'gp_raw']
    time_means = []
    time_stds = []
    
    for method in methods:
        stats = results[method]['statistics']
        time_means.append(stats['time_mean'])
        time_stds.append(stats['time_std'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, time_means, yerr=time_stds, capsize=5,
                   color=[COLORS[m] for m in methods],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel('Total Time (seconds)', fontweight='bold')
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_title(f'Computational Cost Comparison (n={results["dkl"]["statistics"]["n_seeds"]} seeds)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = output_dir / 'dkl_ablation_time_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved: {output_path}")
    plt.close()


def generate_summary_table(all_results: Dict, output_dir: Path):
    """Generate markdown summary table"""
    logger.info("Generating summary table...")
    
    results = all_results['results']
    metadata = all_results['metadata']
    methods = ['dkl', 'pca_gp', 'random_gp', 'gp_raw']
    
    table = []
    table.append("# DKL Ablation Study Results\n")
    table.append(f"**Seeds**: {metadata['n_seeds']}\n")
    table.append(f"**Rounds**: {metadata['n_rounds']}\n")
    table.append(f"**Latent Dim**: {metadata['latent_dim']}D\n\n")
    
    table.append("## Performance Comparison\n\n")
    table.append("| Method | RMSE (K) | Δ vs DKL | p-value | Significant? | Time (s) |\n")
    table.append("|--------|----------|----------|---------|--------------|----------|\n")
    
    dkl_stats = results['dkl']['statistics']
    table.append(f"| {LABELS['dkl']} | {dkl_stats['rmse_mean']:.2f} ± {dkl_stats['rmse_std']:.2f} | baseline | - | - | {dkl_stats['time_mean']:.1f} ± {dkl_stats['time_std']:.1f} |\n")
    
    for method in methods[1:]:
        stats = results[method]['statistics']
        comp = results[method]['comparison_vs_dkl']
        sig = "✅" if comp['significant'] else "❌"
        table.append(f"| {LABELS[method]} | {stats['rmse_mean']:.2f} ± {stats['rmse_std']:.2f} | {comp['delta_rmse']:+.2f} | {comp['p_value']:.4f} | {sig} | {stats['time_mean']:.1f} ± {stats['time_std']:.1f} |\n")
    
    table.append("\n## Interpretation\n\n")
    
    # Determine outcome
    pca_comp = results['pca_gp']['comparison_vs_dkl']
    
    if not pca_comp['significant'] and abs(pca_comp['delta_rmse']) < 1.0:
        table.append("**Finding**: DKL and PCA+GP are **statistically equivalent** (p={:.3f}, Δ={:.2f} K).\n\n".format(
            pca_comp['p_value'], pca_comp['delta_rmse']))
        table.append("**Implication**: Feature learning does not provide measurable advantage over PCA dimensionality reduction in this regime.\n\n")
        table.append("**Honest Assessment**: The claimed \"DKL beats GP\" advantage is primarily from dimensionality reduction (16D vs 81D), not neural network feature learning.\n\n")
    elif pca_comp['significant'] and pca_comp['delta_rmse'] < 0:
        table.append("**Finding**: DKL **significantly outperforms** PCA+GP (p={:.4f}, Δ={:.2f} K).\n\n".format(
            pca_comp['p_value'], pca_comp['delta_rmse']))
        table.append("**Implication**: Feature learning via neural networks provides measurable benefit beyond linear dimensionality reduction.\n\n")
    else:
        table.append("**Finding**: PCA+GP performs better than DKL (p={:.4f}, Δ={:.2f} K).\n\n".format(
            pca_comp['p_value'], pca_comp['delta_rmse']))
        table.append("**Implication**: Linear PCA is more effective than neural network feature learning in this dataset.\n\n")
    
    table.append("## References\n\n")
    table.append("- Wilson et al. (2016): \"Deep Kernel Learning\"\n")
    table.append("- Tipping & Bishop (1999): \"Probabilistic Principal Component Analysis\"\n")
    
    output_path = output_dir / 'DKL_ABLATION_RESULTS.md'
    with open(output_path, 'w') as f:
        f.writelines(table)
    
    logger.info(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DKL ablation results")
    parser.add_argument(
        '--results',
        type=Path,
        default=Path('experiments/ablations/dkl_ablation_results_real.json'),
        help='Path to ablation results JSON'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('experiments/ablations'),
        help='Output directory for plots and tables'
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("DKL ABLATION ANALYSIS")
    logger.info("=" * 80)
    
    if not args.results.exists():
        logger.error(f"Results file not found: {args.results}")
        logger.error("Please run tier2_dkl_ablation_real.py first")
        return
    
    # Load results
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Generate outputs
    args.output.mkdir(parents=True, exist_ok=True)
    
    plot_rmse_comparison(results['results'], args.output)
    plot_training_time(results['results'], args.output)
    generate_summary_table(results, args.output)
    
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    # Print key findings
    logger.info("")
    logger.info("KEY FINDINGS:")
    pca_comp = results['results']['pca_gp']['comparison_vs_dkl']
    dkl_rmse = results['results']['dkl']['statistics']['rmse_mean']
    pca_rmse = results['results']['pca_gp']['statistics']['rmse_mean']
    
    logger.info(f"  DKL: {dkl_rmse:.2f} K")
    logger.info(f"  PCA+GP: {pca_rmse:.2f} K")
    logger.info(f"  Difference: {pca_comp['delta_rmse']:.2f} K (p={pca_comp['p_value']:.4f})")
    
    if pca_comp['significant']:
        if pca_comp['delta_rmse'] < 0:
            logger.info("  → DKL significantly outperforms PCA")
        else:
            logger.info("  → PCA significantly outperforms DKL")
    else:
        logger.info("  → Methods are statistically equivalent")
    
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

