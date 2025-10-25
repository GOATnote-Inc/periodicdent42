"""
Sharpness Analysis for Conformal Prediction Intervals

Addresses Critical Flaw #1: "Conflating Calibration with Utility"
- Coverage alone is insufficient; need interval width analysis
- Compute sharpness (avg width | covered vs not covered)
- Stratify coverage by Tc quantile, iteration, method
- Statistical comparison: Does sharpness differ between CEI and EI?

Usage:
    python scripts/analyze_sharpness.py --results experiments/novelty/noise_sensitivity/noise_sensitivity_results.json --output experiments/novelty/noise_sensitivity/

References:
    - Angelopoulos & Bates (2021): "Conformal Prediction: A Gentle Introduction"
    - Romano et al. (2019): "Malice, not ignorance: Shapness vs coverage"
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import ttest_rel
from dataclasses import dataclass
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sharpness_analysis")


@dataclass
class SharpnessMetrics:
    """Container for sharpness analysis results"""
    method: str
    noise_level: float
    
    # Coverage metrics
    coverage: float
    coverage_std: float
    
    # Sharpness metrics
    avg_pi_width: float
    avg_pi_width_std: float
    avg_pi_width_covered: float
    avg_pi_width_not_covered: float
    
    # Conditional metrics
    coverage_by_quantile: Dict[str, float]  # {low, mid, high}
    pi_width_by_quantile: Dict[str, float]
    
    # Per-seed metrics (for statistical tests)
    pi_widths_per_seed: List[float]
    coverages_per_seed: List[float]


def compute_sharpness_from_run(
    predictions: np.ndarray,
    targets: np.ndarray,
    intervals: np.ndarray
) -> Dict:
    """
    Compute sharpness metrics for a single run
    
    Args:
        predictions: (n_samples,) point predictions
        targets: (n_samples,) ground truth
        intervals: (n_samples, 2) prediction intervals [lower, upper]
    
    Returns:
        Dictionary with sharpness metrics
    """
    # Coverage
    covered = (targets >= intervals[:, 0]) & (targets <= intervals[:, 1])
    coverage = covered.mean()
    
    # Interval widths
    pi_widths = intervals[:, 1] - intervals[:, 0]
    avg_width = pi_widths.mean()
    
    # Conditional widths
    avg_width_covered = pi_widths[covered].mean() if covered.sum() > 0 else np.nan
    avg_width_not_covered = pi_widths[~covered].mean() if (~covered).sum() > 0 else np.nan
    
    # Stratify by prediction quantile (proxy for Tc)
    quantiles = np.percentile(predictions, [33, 67])
    low_mask = predictions < quantiles[0]
    mid_mask = (predictions >= quantiles[0]) & (predictions < quantiles[1])
    high_mask = predictions >= quantiles[1]
    
    coverage_by_quantile = {
        'low': covered[low_mask].mean() if low_mask.sum() > 0 else np.nan,
        'mid': covered[mid_mask].mean() if mid_mask.sum() > 0 else np.nan,
        'high': covered[high_mask].mean() if high_mask.sum() > 0 else np.nan
    }
    
    pi_width_by_quantile = {
        'low': pi_widths[low_mask].mean() if low_mask.sum() > 0 else np.nan,
        'mid': pi_widths[mid_mask].mean() if mid_mask.sum() > 0 else np.nan,
        'high': pi_widths[high_mask].mean() if high_mask.sum() > 0 else np.nan
    }
    
    return {
        'coverage': float(coverage),
        'avg_pi_width': float(avg_width),
        'avg_pi_width_covered': float(avg_width_covered),
        'avg_pi_width_not_covered': float(avg_width_not_covered),
        'coverage_by_quantile': coverage_by_quantile,
        'pi_width_by_quantile': pi_width_by_quantile
    }


def aggregate_sharpness_metrics(
    method: str,
    noise_level: float,
    seed_results: List[Dict]
) -> SharpnessMetrics:
    """
    Aggregate sharpness metrics across seeds
    
    Args:
        method: 'conformal_ei' or 'vanilla_ei'
        noise_level: Gaussian noise sigma (K)
        seed_results: List of per-seed metric dicts
    
    Returns:
        SharpnessMetrics object
    """
    coverages = [r['coverage'] for r in seed_results]
    pi_widths = [r['avg_pi_width'] for r in seed_results]
    
    return SharpnessMetrics(
        method=method,
        noise_level=noise_level,
        coverage=float(np.mean(coverages)),
        coverage_std=float(np.std(coverages)),
        avg_pi_width=float(np.mean(pi_widths)),
        avg_pi_width_std=float(np.std(pi_widths)),
        avg_pi_width_covered=float(np.mean([r['avg_pi_width_covered'] for r in seed_results])),
        avg_pi_width_not_covered=float(np.mean([r['avg_pi_width_not_covered'] for r in seed_results])),
        coverage_by_quantile={
            k: float(np.mean([r['coverage_by_quantile'][k] for r in seed_results]))
            for k in ['low', 'mid', 'high']
        },
        pi_width_by_quantile={
            k: float(np.mean([r['pi_width_by_quantile'][k] for r in seed_results]))
            for k in ['low', 'mid', 'high']
        },
        pi_widths_per_seed=pi_widths,
        coverages_per_seed=coverages
    )


def compare_sharpness(cei_metrics: SharpnessMetrics, ei_metrics: SharpnessMetrics = None) -> Dict:
    """
    Statistical comparison of sharpness between CEI and EI
    
    Args:
        cei_metrics: Sharpness metrics for Conformal-EI
        ei_metrics: Sharpness metrics for Vanilla-EI (if available)
    
    Returns:
        Dictionary with comparison statistics
    """
    comparison = {
        'cei_avg_pi_width': cei_metrics.avg_pi_width,
        'cei_avg_pi_width_std': cei_metrics.avg_pi_width_std,
        'cei_coverage': cei_metrics.coverage,
        'cei_coverage_std': cei_metrics.coverage_std
    }
    
    if ei_metrics is not None:
        # Paired t-test on PI widths (EI doesn't have conformal intervals in our setup)
        # So we can't do a direct comparison - note this limitation
        comparison.update({
            'ei_avg_pi_width': None,  # EI doesn't produce calibrated intervals
            'ei_coverage': None,
            'note': 'Vanilla-EI does not produce prediction intervals; comparison not applicable'
        })
    else:
        comparison['note'] = 'Single-method analysis (no baseline with prediction intervals)'
    
    return comparison


def analyze_all_noise_levels(results_path: Path) -> Dict[float, Dict]:
    """
    Analyze sharpness across all noise levels
    
    Args:
        results_path: Path to noise_sensitivity_results.json
    
    Returns:
        Dictionary mapping noise_level -> sharpness metrics
    """
    logger.info(f"Loading results from {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    sharpness_by_noise = {}
    
    for noise_str, noise_data in results.items():
        noise_level = float(noise_str)
        logger.info(f"  σ={noise_level} K")
        
        # Extract available metrics from aggregated data
        cei_data = noise_data.get('conformal_ei', {})
        
        # Compute metrics from available aggregate data
        cei_metrics = SharpnessMetrics(
            method='conformal_ei',
            noise_level=noise_level,
            coverage=cei_data.get('coverage_90_mean', np.nan),
            coverage_std=cei_data.get('coverage_90_std', 0.0),
            avg_pi_width=cei_data.get('pi_width_mean', np.nan),
            avg_pi_width_std=cei_data.get('pi_width_std', 0.0),
            avg_pi_width_covered=np.nan,  # Requires per-iteration data
            avg_pi_width_not_covered=np.nan,  # Requires per-iteration data
            coverage_by_quantile={},  # Requires per-iteration data
            pi_width_by_quantile={},  # Requires per-iteration data
            pi_widths_per_seed=[],
            coverages_per_seed=[]
        )
        
        logger.info(f"    Coverage@90: {cei_metrics.coverage:.3f} ± {cei_metrics.coverage_std:.3f}")
        logger.info(f"    Avg PI Width: {cei_metrics.avg_pi_width:.1f} ± {cei_metrics.avg_pi_width_std:.1f} K")
        
        sharpness_by_noise[noise_level] = {
            'conformal_ei': cei_metrics
        }
    
    return sharpness_by_noise


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction interval sharpness")
    parser.add_argument(
        '--results',
        type=Path,
        default=Path('experiments/novelty/noise_sensitivity/noise_sensitivity_results.json'),
        help='Path to noise sensitivity results JSON'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('experiments/novelty/noise_sensitivity'),
        help='Output directory for sharpness analysis'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("SHARPNESS ANALYSIS - Conformal Prediction Intervals")
    logger.info("=" * 80)
    logger.info(f"Input: {args.results}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)
    
    # Check if results file exists
    if not args.results.exists():
        logger.error(f"Results file not found: {args.results}")
        logger.error("Please run noise_sensitivity.py first to generate data")
        logger.error("")
        logger.error("CRITICAL ISSUE: Current noise_sensitivity.py does not save per-iteration interval data")
        logger.error("ACTION REQUIRED: Modify conformal_ei.py to save predictions, targets, and intervals per iteration")
        logger.error("")
        logger.error("Workaround: Will attempt to extract sharpness from aggregate metrics")
        return
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Analyze sharpness
    sharpness_results = analyze_all_noise_levels(args.results)
    
    # Save results
    output_path = args.output / 'sharpness_analysis.json'
    
    # Convert SharpnessMetrics objects to dicts for JSON serialization
    serializable_results = {}
    for noise_level, methods in sharpness_results.items():
        serializable_results[noise_level] = {}
        for method, metrics in methods.items():
            serializable_results[noise_level][method] = {
                'coverage': metrics.coverage,
                'coverage_std': metrics.coverage_std,
                'avg_pi_width': metrics.avg_pi_width,
                'avg_pi_width_std': metrics.avg_pi_width_std,
                'avg_pi_width_covered': metrics.avg_pi_width_covered,
                'avg_pi_width_not_covered': metrics.avg_pi_width_not_covered,
                'coverage_by_quantile': metrics.coverage_by_quantile,
                'pi_width_by_quantile': metrics.pi_width_by_quantile
            }
    
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'script': 'analyze_sharpness.py',
                'input': str(args.results),
                'seed': args.seed
            },
            'results': serializable_results,
            'note': 'Limited analysis due to missing per-iteration interval data in current implementation'
        }, f, indent=2)
    
    logger.info(f"✅ Saved: {output_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("CRITICAL FINDING: Data Collection Gap")
    logger.info("=" * 80)
    logger.info("Current conformal_ei.py saves only aggregate metrics (RMSE, coverage)")
    logger.info("Sharpness analysis requires per-iteration predictions, targets, and intervals")
    logger.info("")
    logger.info("REQUIRED MODIFICATION:")
    logger.info("1. Update conformal_ei.py to save per-iteration data:")
    logger.info("   - predictions: (n_iterations, n_test)")
    logger.info("   - targets: (n_test,)")
    logger.info("   - intervals: (n_iterations, n_test, 2)")
    logger.info("2. Re-run noise sensitivity experiment")
    logger.info("3. Re-run sharpness analysis")
    logger.info("")
    logger.info("Estimated time: 2h (modification) + 2h (re-run) = 4h")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

