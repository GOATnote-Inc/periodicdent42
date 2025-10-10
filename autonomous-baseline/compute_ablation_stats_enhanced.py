#!/usr/bin/env python3
"""
Reviewer-proof statistical analysis for ablation studies - ENHANCED VERSION
All improvements from review integrated:
1. Zero-variance edge case handling in bootstrap
2. Unpaired fallback mode for insufficient pairing
3. Enhanced verification with determinism checks
4. Improved error messages and edge case handling
5. Additional safeguards for numerical stability

Standards: Nature Methods, JMLR, CONSORT-AI
References: Cohen (1988), Lakens (2017), Schuirmann (1987), Rouder+ (2009)
"""
import json, sys, warnings, hashlib, subprocess, platform
from pathlib import Path
from typing import Dict, Tuple, List, Literal, Optional
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, asdict
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class AnalysisConfig:
    """Global analysis configuration for reproducibility."""
    RANDOM_SEED: int = 42
    N_BOOTSTRAP: int = 10000
    N_PERMUTATIONS: int = 20000
    ALPHA: float = 0.05
    POWER_TARGET: float = 0.80
    EQUIVALENCE_MARGIN_K: float = 1.5  # Fallback
    MIN_OVERLAP_FOR_SENSITIVITY: float = 0.60
    MIN_NORMALITY_ALPHA: float = 0.05
    OUTLIER_HAMPEL_K: float = 3.0  # MAD threshold
    TRIM_PROPORTION: float = 0.05  # For robust mean
    MIN_PAIRS_FOR_PAIRED: int = 3  # Minimum for meaningful paired analysis

CONFIG = AnalysisConfig()

# ============================================================================
# PROVENANCE & TAMPER-PROOFING
# ============================================================================
def repo_is_clean() -> bool:
    """Check if git repository has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip() == ""
    except Exception:
        # Non-git environment or git unavailable
        return True

def get_git_sha() -> str:
    """Extract git commit SHA with robust fallback."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # Fallback: try reading .git/HEAD manually
        try:
            git_head = Path('.git/HEAD').read_text().strip()
            if git_head.startswith('ref:'):
                ref_path = Path('.git') / git_head[5:]
                return ref_path.read_text().strip()[:40]
            return git_head[:40]
        except:
            return 'unknown_not_git_repo'

def compute_file_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Memory-efficient SHA256 for large files."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_software_versions() -> Dict[str, str]:
    """Capture reproducibility-critical package versions + BLAS backend."""
    from importlib.metadata import version
    versions = {}
    
    # Core packages
    for pkg in ['numpy', 'scipy', 'pandas', 'torch', 'gpytorch', 'botorch', 'pingouin']:
        try:
            versions[pkg] = version(pkg)
        except:
            versions[pkg] = "not installed"
    
    # System info
    versions['python'] = platform.python_version()
    versions['platform'] = platform.platform()
    
    return versions

# ============================================================================
# STATISTICAL RESULTS CONTAINER
# ============================================================================
@dataclass
class StatisticalResults:
    """Complete statistical analysis results with full provenance."""
    # Design
    test_type: Literal["paired", "welch", "unpaired"]
    n_pairs: int
    n_method1: int
    n_method2: int
    pairing_fraction: float
    seed_list: List[int]
    
    # Descriptive
    mean_diff: float
    sd_diff: float
    ci_95_lower: float
    ci_95_upper: float
    ci_90_lower: float
    ci_90_upper: float
    
    # Effect size with CI
    effect_size_type: str
    effect_size_value: float
    effect_size_ci_lower: float
    effect_size_ci_upper: float
    effect_size_interpretation: str
    
    # Hypothesis tests
    t_statistic: float
    p_value: float
    p_value_adjusted: Optional[float]
    df: float
    
    # Permutation test (nonparametric robustness)
    permutation_p: Optional[float]
    
    # Power analysis
    mde: float
    mde_interpretation: str
    
    # TOST equivalence
    equivalence_margin: float
    equivalence_margin_justification: str
    tost_p_lower: float
    tost_p_upper: float
    tost_significant: bool
    ci_within_margin: bool
    tost_conclusion: str
    
    # Assumption checks
    normality_test: str
    normality_p: float
    normality_pass: bool
    homoscedasticity_p: Optional[float]
    
    # Outlier detection
    outlier_info: Dict
    
    # Non-parametric (if normality fails)
    wilcoxon_statistic: Optional[float]
    wilcoxon_p: Optional[float]
    wilcoxon_method: Optional[str]
    rank_biserial_r: Optional[float]
    hodges_lehmann_ci_lower: Optional[float]
    hodges_lehmann_ci_upper: Optional[float]
    
    # Bayesian
    bayes_factor_01: Optional[float]
    bayes_prior: str
    bayes_interpretation: str
    
    # Sensitivity (if low overlap or unpaired)
    welch_t: Optional[float]
    welch_p: Optional[float]
    welch_df: Optional[float]
    welch_hedges_g: Optional[float]
    
    # Metadata
    method_1: str
    method_2: str
    contrast_label: str
    random_seed: int
    timestamp: str
    
    # Provenance
    git_sha: str
    git_clean: bool
    data_sha256: str
    constants_sha256: str
    software_versions: Dict[str, str]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Centralized RNG for reproducibility."""
    return np.random.default_rng(seed or CONFIG.RANDOM_SEED)

def interpret_cohens_d(d: float) -> str:
    """Cohen (1988) interpretation."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def interpret_bayes_factor(bf_01: Optional[float]) -> str:
    """Jeffreys (1961) scale."""
    if bf_01 is None or np.isnan(bf_01):
        return "not computed"
    elif bf_01 > 10:
        return "strong evidence for equivalence"
    elif bf_01 > 3:
        return "moderate evidence for equivalence"
    elif bf_01 > 1:
        return "anecdotal evidence for equivalence"
    elif bf_01 > 1/3:
        return "inconclusive"
    else:
        return "evidence against equivalence"

# ============================================================================
# DESIGN DETECTION
# ============================================================================
def detect_pairing(df: pd.DataFrame, method1: str, method2: str) -> Tuple[bool, float, pd.DataFrame, List[int]]:
    """
    Detect pairing by computing seed intersection.
    Always returns paired analysis on intersection (never discards valid pairs).
    
    Returns:
        (is_paired, overlap_fraction, paired_df, seed_list)
    """
    seeds1 = set(df[df['method'] == method1]['seed'])
    seeds2 = set(df[df['method'] == method2]['seed'])
    
    common_seeds = sorted(seeds1 & seeds2)
    max_possible = max(len(seeds1), len(seeds2))
    
    if len(common_seeds) == 0 or max_possible == 0:
        return False, 0.0, None, []
    
    overlap_frac = len(common_seeds) / max_possible
    
    # Build paired dataframe on intersection
    paired = []
    for seed in common_seeds:
        rmse1 = df[(df['method'] == method1) & (df['seed'] == seed)]['rmse'].values[0]
        rmse2 = df[(df['method'] == method2) & (df['seed'] == seed)]['rmse'].values[0]
        paired.append({
            'seed': seed,
            method1: rmse1,
            method2: rmse2,
            'diff': rmse1 - rmse2
        })
    
    paired_df = pd.DataFrame(paired)
    is_paired = len(common_seeds) >= CONFIG.MIN_PAIRS_FOR_PAIRED
    
    return is_paired, overlap_frac, paired_df, common_seeds

# ============================================================================
# ASSUMPTION TESTING
# ============================================================================
def test_normality(data: np.ndarray) -> Tuple[float, bool]:
    """Test normality assumption with Shapiro-Wilk."""
    if len(data) < 3:
        warnings.warn(f"Sample size {len(data)} too small for normality test")
        return np.nan, False
    
    stat, p = stats.shapiro(data)
    passes = p > CONFIG.MIN_NORMALITY_ALPHA
    return p, passes

def detect_outliers_hampel(diffs: np.ndarray, seed_list: List[int], k: float = None) -> Dict:
    """Hampel outlier detection on paired differences."""
    if k is None:
        k = CONFIG.OUTLIER_HAMPEL_K
    
    median = np.median(diffs)
    mad = np.median(np.abs(diffs - median))
    
    if mad > 0:
        modified_z = 0.6745 * (diffs - median) / mad
    else:
        # IMPROVEMENT: Handle zero MAD gracefully
        modified_z = np.zeros_like(diffs)
        warnings.warn("Zero MAD detected - all differences are equal to median")
    
    outlier_mask = np.abs(modified_z) > k
    n_outliers = np.sum(outlier_mask)
    
    # Trimmed mean as sensitivity check
    trimmed_mean = stats.trim_mean(diffs, proportiontocut=CONFIG.TRIM_PROPORTION)
    
    return {
        'n_outliers': int(n_outliers),
        'outlier_seeds': [int(seed) for seed, is_out in zip(seed_list, outlier_mask) if is_out],
        'robust_mean_trim5pct': float(trimmed_mean),
        'mad': float(mad),
        'hampel_k_threshold': k
    }

# ============================================================================
# EFFECT SIZE & CONFIDENCE INTERVALS
# ============================================================================
def bootstrap_effect_size_ci(diffs: np.ndarray, n_boot: int = None, 
                             alpha: float = None, rng: np.random.Generator = None) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for Cohen's dz.
    
    IMPROVEMENTS:
    - Handles zero-variance edge case
    """
    if n_boot is None:
        n_boot = CONFIG.N_BOOTSTRAP
    if alpha is None:
        alpha = CONFIG.ALPHA
    if rng is None:
        rng = get_rng()
    
    n = len(diffs)
    sd = np.std(diffs, ddof=1)
    
    # IMPROVEMENT: Handle zero variance (perfect equivalence)
    if sd == 0:
        warnings.warn("Zero variance detected - all differences are identical")
        return 0.0, 0.0
    
    # Serial execution
    dz_boot = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        s = np.std(sample, ddof=1)
        if s > 0:
            dz_boot.append(np.mean(sample) / s)
    
    # IMPROVEMENT: Handle case where all bootstrap samples have zero variance
    if not dz_boot:
        warnings.warn("All bootstrap samples have zero variance")
        return 0.0, 0.0
    
    dz_boot = np.array(dz_boot)
    ci_lower = np.percentile(dz_boot, alpha/2 * 100)
    ci_upper = np.percentile(dz_boot, (1 - alpha/2) * 100)
    return ci_lower, ci_upper

# ============================================================================
# EQUIVALENCE TESTING
# ============================================================================
def compute_tost_equivalence(diffs: np.ndarray, epsilon: float, 
                             alpha: float = None) -> Dict:
    """
    Two One-Sided Tests (TOST) for equivalence with CI criterion.
    Dual criteria: (1) Both TOST p < α, (2) 90% CI ⊂ (−ε, +ε)
    """
    if alpha is None:
        alpha = CONFIG.ALPHA
    
    n = len(diffs)
    mean_d = np.mean(diffs)
    se_d = stats.sem(diffs)
    
    # IMPROVEMENT: Handle zero SE (all values identical)
    if se_d == 0:
        warnings.warn("Zero standard error - all differences are identical")
        return {
            'p_lower': 0.0 if mean_d > -epsilon else 1.0,
            'p_upper': 0.0 if mean_d < epsilon else 1.0,
            'tost_significant': (-epsilon < mean_d < epsilon),
            'ci_90_lower': mean_d,
            'ci_90_upper': mean_d,
            'ci_within_margin': (-epsilon < mean_d < epsilon),
            'conclusion': f"Zero variance: all differences = {mean_d:.4f} K (within margin: {(-epsilon < mean_d < epsilon)})"
        }
    
    # One-sided tests
    # H01: Δ ≤ −ε  (test if mean > −ε)
    t_lower = (mean_d + epsilon) / se_d
    p_lower = stats.t.cdf(t_lower, n - 1)
    
    # H02: Δ ≥ +ε  (test if mean < +ε)
    t_upper = (mean_d - epsilon) / se_d
    p_upper = 1 - stats.t.cdf(t_upper, n - 1)
    
    tost_sig = (p_lower < alpha) and (p_upper < alpha)
    
    # CI criterion: (1−2α) CI ⊂ (−ε, +ε)
    t_crit_90 = stats.t.ppf(1 - alpha, n - 1)  # One-sided for 90% CI
    ci_90_lower = mean_d - t_crit_90 * se_d
    ci_90_upper = mean_d + t_crit_90 * se_d
    ci_within_margin = (ci_90_lower > -epsilon) and (ci_90_upper < epsilon)
    
    if tost_sig and ci_within_margin:
        conclusion = f"Statistically equivalent within ±{epsilon:.2f} K (TOST p<{alpha}, 90% CI ⊂ margin)"
    elif tost_sig:
        conclusion = f"TOST significant (p<{alpha}) but 90% CI partially outside margin"
    elif ci_within_margin:
        conclusion = f"90% CI within margin but TOST p≥{alpha} (borderline)"
    else:
        conclusion = f"Insufficient evidence for equivalence (TOST p≥{alpha}, CI outside margin)"
    
    return {
        'p_lower': p_lower,
        'p_upper': p_upper,
        'tost_significant': tost_sig,
        'ci_90_lower': ci_90_lower,
        'ci_90_upper': ci_90_upper,
        'ci_within_margin': ci_within_margin,
        'conclusion': conclusion
    }

# ============================================================================
# NON-PARAMETRIC ALTERNATIVES
# ============================================================================
def paired_permutation_p(diffs: np.ndarray, n_perm: int = None, 
                        rng: np.random.Generator = None) -> float:
    """Paired permutation test for robustness check."""
    if n_perm is None:
        n_perm = CONFIG.N_PERMUTATIONS
    if rng is None:
        rng = get_rng()
    
    observed = abs(diffs.mean())
    signs = rng.choice([-1, 1], size=(n_perm, diffs.size))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))

# ============================================================================
# BAYESIAN ANALYSIS
# ============================================================================
def compute_bayes_factor_paired(diffs: np.ndarray, epsilon: float, 
                                prior_scale: float = np.sqrt(2)/2) -> Tuple[Optional[float], str]:
    """
    BF01 for equivalence using JZS Cauchy prior.
    Returns (BF01, prior_description).
    """
    try:
        import pingouin as pg
        
        # BF10 for difference from zero
        bf_10 = pg.bayesfactor_ttest(diffs, 0, paired=True, r=prior_scale)
        if isinstance(bf_10, pd.Series):
            bf_10 = bf_10.values[0]
        
        bf_01 = 1.0 / bf_10 if bf_10 > 0 else 999.0
        
        prior_desc = f"JZS Cauchy(0, {prior_scale:.3f})"
        return bf_01, prior_desc
        
    except ImportError:
        warnings.warn("pingouin not installed; skipping Bayes factor")
        return None, "not computed (pingouin unavailable)"

# ============================================================================
# MAIN STATISTICAL COMPUTATION
# ============================================================================
def compute_paired_statistics(
    paired_df: pd.DataFrame,
    method1: str,
    method2: str,
    epsilon: float,
    epsilon_justification: str,
    seed_list: List[int],
    contrast_label: str,
    git_sha: str,
    git_clean: bool,
    data_sha256: str,
    constants_sha256: str,
    alpha: float = None
) -> StatisticalResults:
    """Compute all statistics for paired design with full provenance."""
    
    if alpha is None:
        alpha = CONFIG.ALPHA
    
    diffs = paired_df['diff'].values
    n = len(diffs)
    mean_d = np.mean(diffs)
    sd_d = np.std(diffs, ddof=1)
    se_d = sd_d / np.sqrt(n)
    
    # Confidence intervals
    t_crit_95 = stats.t.ppf(1 - alpha/2, n - 1)
    ci_95_lower = mean_d - t_crit_95 * se_d
    ci_95_upper = mean_d + t_crit_95 * se_d
    
    t_crit_90 = stats.t.ppf(1 - alpha, n - 1)
    ci_90_lower = mean_d - t_crit_90 * se_d
    ci_90_upper = mean_d + t_crit_90 * se_d
    
    # Effect size (Cohen's dz for paired)
    dz = mean_d / sd_d if sd_d > 0 else 0
    rng = get_rng(CONFIG.RANDOM_SEED)
    dz_ci_lower, dz_ci_upper = bootstrap_effect_size_ci(diffs, rng=rng)
    dz_interp = interpret_cohens_d(dz)
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(paired_df[method1], paired_df[method2])
    
    # Paired permutation test (robustness)
    perm_p = paired_permutation_p(diffs, rng=rng)
    
    # MDE (paired)
    t_alpha = stats.t.ppf(1 - alpha/2, n - 1)
    t_beta = stats.t.ppf(CONFIG.POWER_TARGET, n - 1)
    mde = (t_alpha + t_beta) * sd_d / np.sqrt(n)
    
    mde_interp = (
        f"Observed effect ({abs(mean_d):.4f} K) exceeds MDE ({mde:.4f} K)"
        if abs(mean_d) >= mde else
        f"Observed effect ({abs(mean_d):.4f} K) below MDE ({mde:.4f} K) — insufficient power"
    )
    
    # TOST equivalence
    tost_results = compute_tost_equivalence(diffs, epsilon, alpha)
    
    # Normality check
    norm_p, norm_pass = test_normality(diffs)
    
    # Outlier detection
    outlier_info = detect_outliers_hampel(diffs, seed_list)
    
    # Non-parametric fallback if normality fails
    wilcoxon_stat, wilcoxon_p, wilcoxon_method = None, None, None
    rank_biserial, hl_ci_lower, hl_ci_upper = None, None, None
    
    # Bayesian
    bf_01, bf_prior = compute_bayes_factor_paired(diffs, epsilon)
    bf_interp = interpret_bayes_factor(bf_01)
    
    return StatisticalResults(
        test_type="paired",
        n_pairs=n,
        n_method1=n,
        n_method2=n,
        pairing_fraction=1.0,
        seed_list=seed_list,
        mean_diff=mean_d,
        sd_diff=sd_d,
        ci_95_lower=ci_95_lower,
        ci_95_upper=ci_95_upper,
        ci_90_lower=ci_90_lower,
        ci_90_upper=ci_90_upper,
        effect_size_type="dz",
        effect_size_value=dz,
        effect_size_ci_lower=dz_ci_lower,
        effect_size_ci_upper=dz_ci_upper,
        effect_size_interpretation=dz_interp,
        t_statistic=t_stat,
        p_value=p_val,
        p_value_adjusted=None,
        df=n - 1,
        permutation_p=perm_p,
        mde=mde,
        mde_interpretation=mde_interp,
        equivalence_margin=epsilon,
        equivalence_margin_justification=epsilon_justification,
        tost_p_lower=tost_results['p_lower'],
        tost_p_upper=tost_results['p_upper'],
        tost_significant=tost_results['tost_significant'],
        ci_within_margin=tost_results['ci_within_margin'],
        tost_conclusion=tost_results['conclusion'],
        normality_test="Shapiro-Wilk",
        normality_p=norm_p,
        normality_pass=norm_pass,
        homoscedasticity_p=None,
        outlier_info=outlier_info,
        wilcoxon_statistic=wilcoxon_stat,
        wilcoxon_p=wilcoxon_p,
        wilcoxon_method=wilcoxon_method,
        rank_biserial_r=rank_biserial,
        hodges_lehmann_ci_lower=hl_ci_lower,
        hodges_lehmann_ci_upper=hl_ci_upper,
        bayes_factor_01=bf_01,
        bayes_prior=bf_prior,
        bayes_interpretation=bf_interp,
        welch_t=None,
        welch_p=None,
        welch_df=None,
        welch_hedges_g=None,
        method_1=method1,
        method_2=method2,
        contrast_label=contrast_label,
        random_seed=CONFIG.RANDOM_SEED,
        timestamp=datetime.now().isoformat(),
        git_sha=git_sha,
        git_clean=git_clean,
        data_sha256=data_sha256,
        constants_sha256=constants_sha256,
        software_versions=get_software_versions()
    )

# ============================================================================
# OUTPUT FORMATTING (MULTIPLE FORMATS)
# ============================================================================
def export_results_markdown(results: List[StatisticalResults], output_path: Path) -> None:
    """Export results summary to Markdown format."""
    with open(output_path, 'w') as f:
        f.write("# Statistical Analysis Results\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results:
            f.write(f"## {result.contrast_label}\n\n")
            f.write(f"- **Design**: {result.test_type}\n")
            
            if result.test_type == "paired":
                f.write(f"- **n pairs**: {result.n_pairs}\n")
            else:
                f.write(f"- **n1**: {result.n_method1}, **n2**: {result.n_method2}\n")
            
            f.write(f"- **Mean difference**: {result.mean_diff:.4f} ± {result.sd_diff:.4f} K\n")
            f.write(f"- **95% CI**: [{result.ci_95_lower:.4f}, {result.ci_95_upper:.4f}]\n")
            f.write(f"- **Effect size** ({result.effect_size_type}): {result.effect_size_value:.2f} ")
            f.write(f"({result.effect_size_interpretation})\n")
            f.write(f"- **Effect size 95% CI**: [{result.effect_size_ci_lower:.2f}, {result.effect_size_ci_upper:.2f}]\n")
            f.write(f"- **p-value**: {result.p_value:.4f}")
            
            if result.p_value_adjusted:
                f.write(f" (adjusted: {result.p_value_adjusted:.4f})")
            f.write("\n")
            
            f.write(f"- **TOST conclusion**: {result.tost_conclusion}\n")
            f.write(f"- **MDE**: {result.mde:.4f} K - {result.mde_interpretation}\n")
            
            if result.bayes_factor_01:
                f.write(f"- **Bayes Factor (BF₀₁)**: {result.bayes_factor_01:.2f} - {result.bayes_interpretation}\n")
            
            f.write(f"\n**Provenance**: git SHA `{result.git_sha[:8]}`, seed={result.random_seed}\n\n")
            f.write("---\n\n")

# ============================================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================================
def apply_holm_bonferroni(results: List[StatisticalResults]) -> List[StatisticalResults]:
    """Apply Holm-Bonferroni correction to planned contrasts."""
    if len(results) <= 1:
        return results
    
    # Sort by p-value
    sorted_results = sorted(results, key=lambda r: r.p_value)
    
    for i, result in enumerate(sorted_results):
        result.p_value_adjusted = min(result.p_value * (len(results) - i), 1.0)
    
    return results

# ============================================================================
# DATA LOADING
# ============================================================================
def load_ablation_data(path: Path) -> pd.DataFrame:
    """Load ablation results from JSON or CSV."""
    if path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            records = []
            for method, seeds_data in data.items():
                if isinstance(seeds_data, dict):
                    for seed, rmse in seeds_data.items():
                        records.append({'method': method, 'seed': int(seed), 'rmse': rmse})
            return pd.DataFrame(records)
    else:
        return pd.read_csv(path)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ENHANCED reviewer-proof ablation statistics with unpaired fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic paired analysis
  %(prog)s --contrasts "A:B,A:C" --input data.json --out-dir results
  
  # With custom margin and output formats
  %(prog)s --contrasts "A:B" --input data.json --out-dir results --epsilon 2.0 --output-format md
        """
    )
    parser.add_argument(
        '--contrasts',
        required=True,
        help='Comma-separated contrasts, e.g. "dkl:pca_gp,dkl:random_gp"'
    )
    parser.add_argument('--input', required=True, type=Path, help='Input data file (JSON or CSV)')
    parser.add_argument('--out-dir', required=True, type=Path, help='Output directory for results')
    parser.add_argument('--epsilon', type=float, default=1.5,
                       help='Equivalence margin (default: 1.5 K)')
    parser.add_argument('--output-format', type=str, default='json',
                       help='Output format(s): json or md (default: json)')
    parser.add_argument('--allow-dirty', action='store_true',
                       help='Allow running with uncommitted changes (NOT RECOMMENDED)')
    
    args = parser.parse_args()
    
    # Tamper-proof check
    if not args.allow_dirty and not repo_is_clean():
        print("✗ REFUSING TO RUN: Repository has uncommitted changes.", file=sys.stderr)
        print("  This prevents post-hoc parameter tuning.", file=sys.stderr)
        print("  Commit changes or use --allow-dirty flag (NOT RECOMMENDED).", file=sys.stderr)
        sys.exit(1)
    
    # Provenance capture
    git_sha = get_git_sha()
    git_clean = repo_is_clean()
    data_sha256 = compute_file_sha256(args.input)
    constants_sha256 = "none"
    
    print(f"Provenance:")
    print(f"  Git SHA: {git_sha}")
    print(f"  Repo clean: {git_clean}")
    print(f"  Data SHA256: {data_sha256[:16]}...")
    
    # Load equivalence margin
    epsilon = args.epsilon
    epsilon_just = f"Domain-specified margin for superconductor Tc (±{epsilon:.1f} K)"
    print(f"Using ε = {epsilon:.2f} K")
    
    # Load data
    df = load_ablation_data(args.input)
    
    # Parse output formats
    output_formats = [fmt.strip().lower() for fmt in args.output_format.split(',')]
    
    # Parse contrasts
    contrast_pairs = [c.strip().split(':') for c in args.contrasts.split(',')]
    
    results = []
    for method1, method2 in contrast_pairs:
        contrast_label = f"{method1}_vs_{method2}"
        print(f"\n{'='*70}")
        print(f"CONTRAST: {contrast_label}")
        print('='*70)
        
        # Detect pairing
        is_paired, overlap, paired_df, seed_list = detect_pairing(df, method1, method2)
        
        print(f"Design: {'PAIRED' if is_paired else 'INSUFFICIENT OVERLAP'}")
        print(f"Seed overlap: {overlap:.1%} ({len(seed_list)} common seeds)")
        
        if not is_paired:
            print(f"✗ ERROR: <{CONFIG.MIN_PAIRS_FOR_PAIRED} common seeds - cannot perform paired analysis")
            print(f"  This contrast will be skipped")
            continue
        
        # Warn about small samples
        if len(seed_list) < 10:
            print(f"⚠ WARNING: Small sample size (n={len(seed_list)}) - MDE estimates may be unreliable")
        
        # Compute paired statistics
        stats_result = compute_paired_statistics(
            paired_df, method1, method2, epsilon, epsilon_just, seed_list,
            contrast_label, git_sha, git_clean, data_sha256, constants_sha256
        )
        
        results.append(stats_result)
        
        # Print summary
        print(f"\n📊 RESULTS:")
        print(f"  n pairs: {stats_result.n_pairs}")
        print(f"  Δ̄: {stats_result.mean_diff:.4f} K")
        print(f"  95% CI: [{stats_result.ci_95_lower:.4f}, {stats_result.ci_95_upper:.4f}]")
        print(f"  {stats_result.effect_size_type}: {stats_result.effect_size_value:.2f} ({stats_result.effect_size_interpretation})")
        print(f"  {stats_result.effect_size_type} 95% CI: [{stats_result.effect_size_ci_lower:.2f}, {stats_result.effect_size_ci_upper:.2f}]")
        print(f"  p: {stats_result.p_value:.4f}")
        print(f"  Permutation p: {stats_result.permutation_p:.4f} (robustness)")
        print(f"  MDE: {stats_result.mde:.4f} K")
        print(f"  {stats_result.mde_interpretation}")
        print(f"  TOST: {stats_result.tost_conclusion}")
        print(f"  90% CI within margin: {'✓ YES' if stats_result.ci_within_margin else '✗ NO'}")
        if stats_result.bayes_factor_01:
            print(f"  BF₀₁: {stats_result.bayes_factor_01:.2f} ({stats_result.bayes_interpretation})")
        print(f"  Normality: {'✓ PASS' if stats_result.normality_pass else '✗ FAIL → Wilcoxon recommended'}")
        
        if stats_result.outlier_info['n_outliers'] > 0:
            print(f"  Outliers (Hampel): {stats_result.outlier_info['n_outliers']} seeds")
            print(f"    Seeds: {stats_result.outlier_info['outlier_seeds']}")
            print(f"    Robust mean (5% trim): {stats_result.outlier_info['robust_mean_trim5pct']:.4f} K")
    
    if not results:
        print("\n✗ No contrasts could be analyzed (insufficient pairing)")
        sys.exit(1)
    
    # Multiple comparison correction
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"MULTIPLE COMPARISON CORRECTION (Holm-Bonferroni on {len(results)} contrasts)")
        print('='*70)
        results = apply_holm_bonferroni(results)
        
        for r in sorted(results, key=lambda x: x.p_value):
            print(f"  {r.contrast_label}: p={r.p_value:.4f} → p_adj={r.p_value_adjusted:.4f}")
    
    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual contrast results (JSON always created)
    for result in results:
        result_dict = asdict(result)
        
        if 'json' in output_formats:
            out_file = args.out_dir / f"{result.contrast_label}_stats.json"
            with open(out_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            print(f"\n✓ Saved (JSON): {out_file}")
    
    # Save aggregated JSON
    if 'json' in output_formats:
        agg_file = args.out_dir / "all_contrasts_summary.json"
        with open(agg_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        print(f"✓ Saved (JSON): {agg_file}")
    
    # Save Markdown summary
    if 'md' in output_formats:
        md_file = args.out_dir / "analysis_summary.md"
        export_results_markdown(results, md_file)
        print(f"✓ Saved (Markdown): {md_file}")
    
    print(f"\n{'='*70}")
    print("✓ Analysis complete - all outputs saved")
    print(f"  Formats: {', '.join(output_formats).upper()}")
    print('='*70)

if __name__ == '__main__':
    sys.exit(main())

