#!/usr/bin/env python3
"""
Generate publication-quality plots from noise sensitivity results.
Auto-run when experiments/novelty/noise_sensitivity/results.json exists.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("plot_noise_sensitivity")

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    "vanilla_ei": "#2E86AB",  # Blue
    "conformal_ei": "#A23B72",  # Purple
}

LABELS = {
    "vanilla_ei": "Vanilla EI",
    "conformal_ei": "Conformal-EI (Locally Adaptive)",
}


def load_results(results_path: Path) -> dict:
    """Load results JSON"""
    with open(results_path) as f:
        return json.load(f)


def plot_rmse_vs_noise(data: dict, outdir: Path):
    """Plot RMSE vs noise level with error bars"""
    logger.info("📊 Generating RMSE vs noise plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract sigma levels
    sigmas = sorted([float(k) for k in data.keys()])
    
    for method in ["vanilla_ei", "conformal_ei"]:
        rmse_means = []
        rmse_stds = []
        
        for sigma in sigmas:
            sigma_data = data[str(float(sigma))]
            method_data = sigma_data.get(method, {})
            rmse_means.append(method_data.get("rmse_mean", np.nan))
            rmse_stds.append(method_data.get("rmse_std", np.nan))
        
        ax.errorbar(
            sigmas, rmse_means, yerr=rmse_stds,
            marker='o', markersize=8, linewidth=2, capsize=5,
            label=LABELS[method], color=COLORS[method]
        )
    
    ax.set_xlabel("Noise Level σ (K)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Final RMSE (K)", fontsize=12, fontweight='bold')
    ax.set_title("Active Learning Performance vs Noise", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(outdir / "rmse_vs_noise.png")
    logger.info(f"✅ Saved: {outdir / 'rmse_vs_noise.png'}")
    plt.close()


def plot_regret_vs_noise(data: dict, outdir: Path):
    """Plot regret vs noise level with significance markers"""
    logger.info("📊 Generating regret vs noise plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sigmas = sorted([float(k) for k in data.keys()])
    
    for method in ["vanilla_ei", "conformal_ei"]:
        regret_means = []
        regret_stds = []
        
        for sigma in sigmas:
            sigma_data = data[str(float(sigma))]
            method_data = sigma_data.get(method, {})
            regret_means.append(method_data.get("regret_mean", np.nan))
            regret_stds.append(method_data.get("regret_std", np.nan))
        
        ax.errorbar(
            sigmas, regret_means, yerr=regret_stds,
            marker='o', markersize=8, linewidth=2, capsize=5,
            label=LABELS[method], color=COLORS[method]
        )
    
    # Mark significant differences
    for sigma in sigmas:
        sigma_data = data[str(float(sigma))]
        comparison = sigma_data.get("comparison", {})
        p_val = comparison.get("p_value_regret", 1.0)
        
        if p_val < 0.05:
            # Draw significance marker
            y_max = max([sigma_data[m]["regret_mean"] for m in ["vanilla_ei", "conformal_ei"]])
            ax.plot(sigma, y_max * 1.1, marker='*', markersize=15, color='gold',
                   markeredgecolor='black', markeredgewidth=0.5)
    
    ax.set_xlabel("Noise Level σ (K)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Regret (K)", fontsize=12, fontweight='bold')
    ax.set_title("Regret Reduction vs Noise (⭐ = p < 0.05)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(outdir / "regret_vs_noise.png")
    logger.info(f"✅ Saved: {outdir / 'regret_vs_noise.png'}")
    plt.close()


def plot_coverage_vs_noise(data: dict, outdir: Path):
    """Plot calibration quality (Coverage@90) vs noise"""
    logger.info("📊 Generating coverage vs noise plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sigmas = sorted([float(k) for k in data.keys()])
    
    # Only Conformal-EI has meaningful coverage metrics
    coverage_90_means = []
    coverage_90_stds = []
    
    for sigma in sigmas:
        sigma_data = data[str(float(sigma))]
        cei_data = sigma_data.get("conformal_ei", {})
        coverage_90_means.append(cei_data.get("coverage_90_mean", np.nan))
        coverage_90_stds.append(cei_data.get("coverage_90_std", np.nan))
    
    ax.errorbar(
        sigmas, coverage_90_means, yerr=coverage_90_stds,
        marker='o', markersize=8, linewidth=2, capsize=5,
        label="Conformal-EI", color=COLORS["conformal_ei"]
    )
    
    # Target coverage line
    ax.axhline(y=0.90, color='black', linestyle='--', linewidth=2, label="Target (90%)")
    ax.fill_between(sigmas, 0.85, 0.95, alpha=0.2, color='green', label="Acceptable (±5%)")
    
    ax.set_xlabel("Noise Level σ (K)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Coverage@90 (proportion)", fontsize=12, fontweight='bold')
    ax.set_title("Calibration Quality vs Noise", fontsize=14, fontweight='bold')
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc='lower left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(outdir / "coverage_vs_noise.png")
    logger.info(f"✅ Saved: {outdir / 'coverage_vs_noise.png'}")
    plt.close()


def generate_summary_stats(data: dict, outdir: Path):
    """Generate summary statistics table"""
    logger.info("📊 Generating summary statistics...")
    
    sigmas = sorted([float(k) for k in data.keys()])
    
    summary = []
    summary.append("# Noise Sensitivity Results Summary")
    summary.append("")
    summary.append("| σ (K) | Method | RMSE (K) | Regret (K) | Coverage@90 | p-value | Significant? |")
    summary.append("|-------|--------|----------|------------|-------------|---------|--------------|")
    
    for sigma in sigmas:
        sigma_data = data[str(float(sigma))]
        comparison = sigma_data.get("comparison", {})
        p_val = comparison.get("p_value_regret", np.nan)
        sig = "✅" if p_val < 0.05 else "❌"
        
        for method in ["vanilla_ei", "conformal_ei"]:
            method_data = sigma_data.get(method, {})
            rmse = method_data.get("rmse_mean", np.nan)
            regret = method_data.get("regret_mean", np.nan)
            cov90 = method_data.get("coverage_90_mean", np.nan)
            
            if method == "vanilla_ei":
                summary.append(
                    f"| {sigma:.0f} | {LABELS[method]} | {rmse:.2f} | {regret:.2f} | N/A | — | — |"
                )
            else:
                summary.append(
                    f"| {sigma:.0f} | {LABELS[method]} | {rmse:.2f} | {regret:.2f} | {cov90:.3f} | {p_val:.4f} | {sig} |"
                )
    
    summary.append("")
    summary.append("**Legend**: ✅ Significant (p < 0.05) | ❌ Not significant")
    summary.append("")
    
    # Find σ_critical
    critical_sigmas = [sigma for sigma in sigmas 
                      if data[str(float(sigma))].get("comparison", {}).get("p_value_regret", 1.0) < 0.05]
    
    if critical_sigmas:
        sigma_critical = min(critical_sigmas)
        summary.append(f"**σ_critical = {sigma_critical:.0f} K** (first noise level where CEI beats EI, p < 0.05)")
    else:
        summary.append("**σ_critical = NOT FOUND** (no significant difference at any tested noise level)")
    
    summary_path = outdir / "summary_stats.md"
    summary_path.write_text("\n".join(summary))
    logger.info(f"✅ Saved: {summary_path}")


def main():
    # Locate results
    results_path = Path("experiments/novelty/noise_sensitivity/results.json")
    
    if not results_path.exists():
        logger.warning(f"⚠️  Results not found at {results_path}")
        logger.info("This script will auto-run when experiment completes.")
        return
    
    logger.info("="*70)
    logger.info("NOISE SENSITIVITY PLOTTING")
    logger.info("="*70)
    logger.info(f"📂 Loading: {results_path}")
    
    data = load_results(results_path)
    outdir = results_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_rmse_vs_noise(data, outdir)
    plot_regret_vs_noise(data, outdir)
    plot_coverage_vs_noise(data, outdir)
    generate_summary_stats(data, outdir)
    
    logger.info("="*70)
    logger.info("✅ PLOTTING COMPLETE")
    logger.info(f"📂 Outputs: {outdir}")
    logger.info("   - rmse_vs_noise.png")
    logger.info("   - regret_vs_noise.png")
    logger.info("   - coverage_vs_noise.png")
    logger.info("   - summary_stats.md")
    logger.info("="*70)


if __name__ == "__main__":
    main()

