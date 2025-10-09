#!/usr/bin/env python3
"""
Lab Self-Audit Validation Script
Automatically executes scientific rigor checklist for Phase 10 Tier 2

Based on NeurIPS/ICML reproducibility standards and materials science best practices.

Usage:
    python scripts/audit_validation.py --full
    python scripts/audit_validation.py --quick  # Fast checks only
"""

import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
import pandas as pd

# Color codes for terminal output
class Colors:
    PASS = '\033[92m'  # Green
    FAIL = '\033[91m'  # Red
    WARN = '\033[93m'  # Yellow
    INFO = '\033[94m'  # Blue
    RESET = '\033[0m'

def check_mark(passed: bool) -> str:
    """Return colored checkmark or X"""
    if passed:
        return f"{Colors.PASS}✅{Colors.RESET}"
    return f"{Colors.FAIL}❌{Colors.RESET}"

def warn_mark() -> str:
    """Return warning symbol"""
    return f"{Colors.WARN}⚠️{Colors.RESET}"

# ============================================================================
# 1. STATISTICAL ROBUSTNESS CHECKS
# ============================================================================

def check_statistical_power(results_path: Path) -> Dict:
    """
    Verify statistical power and recompute p-values.
    
    Checks:
    - Number of seeds ≥ 20 (publication grade)
    - t-test assumptions (normality, equal variance)
    - Bootstrap confidence intervals
    - Effect size (Cohen's d)
    """
    print(f"\n{Colors.INFO}📊 STATISTICAL ROBUSTNESS{Colors.RESET}")
    print("=" * 70)
    
    with open(results_path) as f:
        data = json.load(f)
    
    n_seeds = data['n_seeds']
    dkl_rmse = np.array([h[-1] for h in data['results']['dkl']['rmse_histories']])
    gp_rmse = np.array([h[-1] for h in data['results']['gp']['rmse_histories']])
    random_rmse = np.array([h[-1] for h in data['results']['random']['rmse_histories']])
    
    checks = {}
    
    # Check 1: Number of seeds
    checks['n_seeds'] = {
        'value': n_seeds,
        'target': 20,
        'passed': n_seeds >= 20,
        'criticality': 'HIGH'
    }
    print(f"Seeds: {n_seeds} (target ≥20) {check_mark(n_seeds >= 20)}")
    
    # Check 2: Normality (Shapiro-Wilk)
    _, p_shapiro_dkl = stats.shapiro(dkl_rmse)
    _, p_shapiro_gp = stats.shapiro(gp_rmse)
    normality_ok = p_shapiro_dkl > 0.05 and p_shapiro_gp > 0.05
    checks['normality'] = {
        'dkl_p': float(p_shapiro_dkl),
        'gp_p': float(p_shapiro_gp),
        'passed': normality_ok,
        'criticality': 'MEDIUM'
    }
    print(f"Normality: Shapiro-Wilk p={p_shapiro_dkl:.3f} (DKL), {p_shapiro_gp:.3f} (GP) {check_mark(normality_ok)}")
    
    # Check 3: Recompute t-test
    t_stat, p_value = stats.ttest_ind(dkl_rmse, gp_rmse, equal_var=False)
    reported_p = data['comparisons']['dkl_vs_gp']['p_value']
    p_match = abs(p_value - reported_p) < 0.001
    checks['p_value_verification'] = {
        'computed': float(p_value),
        'reported': float(reported_p),
        'match': p_match,
        'passed': p_match,
        'criticality': 'HIGH'
    }
    print(f"p-value: computed={p_value:.4f}, reported={reported_p:.4f} {check_mark(p_match)}")
    
    # Check 4: Effect size (Cohen's d)
    pooled_std = np.sqrt((dkl_rmse.var() + gp_rmse.var()) / 2)
    cohens_d = (gp_rmse.mean() - dkl_rmse.mean()) / pooled_std
    checks['effect_size'] = {
        'cohens_d': float(cohens_d),
        'interpretation': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
        'passed': abs(cohens_d) > 0.5,  # At least medium effect
        'criticality': 'MEDIUM'
    }
    print(f"Effect size: Cohen's d={cohens_d:.2f} ({checks['effect_size']['interpretation']}) {check_mark(abs(cohens_d) > 0.5)}")
    
    # Check 5: Bootstrap confidence intervals
    n_bootstrap = 10000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        dkl_sample = np.random.choice(dkl_rmse, size=len(dkl_rmse), replace=True)
        gp_sample = np.random.choice(gp_rmse, size=len(gp_rmse), replace=True)
        bootstrap_diffs.append(gp_sample.mean() - dkl_sample.mean())
    
    ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])
    ci_excludes_zero = ci_lower > 0
    checks['confidence_interval'] = {
        'mean_diff': float(gp_rmse.mean() - dkl_rmse.mean()),
        'ci_95': [float(ci_lower), float(ci_upper)],
        'excludes_zero': ci_excludes_zero,
        'passed': ci_excludes_zero,
        'criticality': 'HIGH'
    }
    print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] K {check_mark(ci_excludes_zero)}")
    
    # Check 6: Variance ratio (homoscedasticity)
    var_ratio = dkl_rmse.var() / gp_rmse.var()
    variance_ok = 0.25 < var_ratio < 4.0  # Within 4x difference
    checks['variance_ratio'] = {
        'ratio': float(var_ratio),
        'passed': variance_ok,
        'criticality': 'LOW'
    }
    print(f"Variance ratio: {var_ratio:.2f} (DKL/GP) {check_mark(variance_ok)}")
    
    return checks

# ============================================================================
# 2. REPRODUCIBILITY CHECKS
# ============================================================================

def check_reproducibility(repo_path: Path) -> Dict:
    """
    Verify reproducibility artifacts and determinism.
    
    Checks:
    - SHA-256 checksums for data
    - Model checkpoint existence
    - Seed documentation
    - Deterministic flags in code
    """
    print(f"\n{Colors.INFO}🔒 REPRODUCIBILITY{Colors.RESET}")
    print("=" * 70)
    
    checks = {}
    
    # Check 1: Dataset checksum
    data_file = repo_path / "data" / "uci_superconductivity.csv"
    if data_file.exists():
        sha256 = hashlib.sha256()
        with open(data_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        checksum = sha256.hexdigest()
        checks['dataset_checksum'] = {
            'file': str(data_file),
            'sha256': checksum,
            'passed': True,
            'criticality': 'MEDIUM'
        }
        print(f"Dataset SHA-256: {checksum[:16]}... ✅")
    else:
        checks['dataset_checksum'] = {
            'file': str(data_file),
            'passed': False,
            'criticality': 'MEDIUM'
        }
        print(f"Dataset not found: {data_file} ❌")
    
    # Check 2: Model checkpoints
    checkpoint_dir = repo_path / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pkl"))
        checks['model_checkpoints'] = {
            'count': len(checkpoints),
            'passed': len(checkpoints) > 0,
            'criticality': 'LOW'
        }
        print(f"Model checkpoints: {len(checkpoints)} found {check_mark(len(checkpoints) > 0)}")
    else:
        checks['model_checkpoints'] = {'count': 0, 'passed': False, 'criticality': 'LOW'}
        print(f"Checkpoint directory missing ❌")
    
    # Check 3: Seed documentation in results
    results_file = repo_path / "evidence" / "phase10" / "tier2_clean" / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        n_seeds = results.get('n_seeds', 0)
        checks['seed_documentation'] = {
            'n_seeds': n_seeds,
            'passed': n_seeds >= 5,
            'criticality': 'HIGH'
        }
        print(f"Seeds documented: {n_seeds} {check_mark(n_seeds >= 5)}")
    else:
        checks['seed_documentation'] = {'passed': False, 'criticality': 'HIGH'}
        print(f"Results file missing ❌")
    
    # Check 4: Deterministic flags in code
    model_file = repo_path / "phase10_gp_active_learning" / "models" / "dkl_model.py"
    if model_file.exists():
        code = model_file.read_text()
        has_torch_manual_seed = 'torch.manual_seed' in code
        has_np_random_seed = 'np.random.seed' in code
        deterministic_ok = has_torch_manual_seed and has_np_random_seed
        checks['deterministic_flags'] = {
            'torch_manual_seed': has_torch_manual_seed,
            'np_random_seed': has_np_random_seed,
            'passed': deterministic_ok,
            'criticality': 'HIGH'
        }
        print(f"Deterministic seeds in code: {check_mark(deterministic_ok)}")
    else:
        checks['deterministic_flags'] = {'passed': False, 'criticality': 'HIGH'}
        print(f"Model file missing ❌")
    
    return checks

# ============================================================================
# 3. BASELINE COVERAGE CHECK
# ============================================================================

def check_baseline_coverage(results_path: Path) -> Dict:
    """
    Verify external baseline comparisons.
    
    Expected baselines:
    - XGBoost
    - Random Forest
    - CGCNN (if structure data available)
    - MEGNet (if structure data available)
    """
    print(f"\n{Colors.INFO}📈 BASELINE COVERAGE{Colors.RESET}")
    print("=" * 70)
    
    with open(results_path) as f:
        data = json.load(f)
    
    tested_methods = list(data['results'].keys())
    expected_baselines = ['xgboost', 'random_forest', 'cgcnn', 'megnet']
    
    checks = {}
    for baseline in expected_baselines:
        present = baseline in tested_methods
        checks[baseline] = {
            'tested': present,
            'passed': present,
            'criticality': 'HIGH' if baseline in ['xgboost', 'random_forest'] else 'MEDIUM'
        }
        print(f"{baseline.upper()}: {check_mark(present)}")
    
    n_baselines = sum(1 for c in checks.values() if c['tested'])
    checks['_summary'] = {
        'n_baselines': n_baselines,
        'n_expected': len(expected_baselines),
        'passed': n_baselines >= 2,  # At least 2 external baselines
        'criticality': 'HIGH'
    }
    print(f"\nTotal baselines: {n_baselines}/{len(expected_baselines)} {check_mark(n_baselines >= 2)}")
    
    return checks

# ============================================================================
# 4. PHYSICS INTERPRETABILITY CHECK
# ============================================================================

def check_physics_interpretability(repo_path: Path) -> Dict:
    """
    Verify physical interpretability analysis exists.
    
    Expected artifacts:
    - Feature-physics correlation analysis
    - t-SNE/UMAP visualization
    - SHAP analysis
    """
    print(f"\n{Colors.INFO}🔬 PHYSICS INTERPRETABILITY{Colors.RESET}")
    print("=" * 70)
    
    evidence_dir = repo_path / "evidence" / "phase10" / "tier2_clean"
    
    checks = {}
    
    # Check for correlation analysis
    corr_file = evidence_dir / "feature_physics_correlations.png"
    checks['correlation_analysis'] = {
        'file': str(corr_file),
        'exists': corr_file.exists(),
        'passed': corr_file.exists(),
        'criticality': 'HIGH'
    }
    print(f"Feature-physics correlations: {check_mark(corr_file.exists())}")
    
    # Check for t-SNE visualization
    tsne_file = evidence_dir / "tsne_learned_space.png"
    checks['tsne_visualization'] = {
        'file': str(tsne_file),
        'exists': tsne_file.exists(),
        'passed': tsne_file.exists(),
        'criticality': 'MEDIUM'
    }
    print(f"t-SNE visualization: {check_mark(tsne_file.exists())}")
    
    # Check for SHAP analysis
    shap_file = evidence_dir / "shap_feature_importance.png"
    checks['shap_analysis'] = {
        'file': str(shap_file),
        'exists': shap_file.exists(),
        'passed': shap_file.exists(),
        'criticality': 'MEDIUM'
    }
    print(f"SHAP analysis: {check_mark(shap_file.exists())}")
    
    n_present = sum(1 for c in checks.values() if c.get('exists', False))
    checks['_summary'] = {
        'n_artifacts': n_present,
        'n_expected': 3,
        'passed': n_present >= 1,  # At least one interpretability analysis
        'criticality': 'HIGH'
    }
    print(f"\nInterpretability artifacts: {n_present}/3 {check_mark(n_present >= 1)}")
    
    return checks

# ============================================================================
# 5. GENERATE AUDIT REPORT
# ============================================================================

def generate_audit_report(all_checks: Dict, output_path: Path):
    """Generate comprehensive audit report as Markdown"""
    
    # Count passes/fails by criticality
    high_pass = sum(1 for cat in all_checks.values() 
                   for check in cat.values() 
                   if isinstance(check, dict) and check.get('criticality') == 'HIGH' and check.get('passed'))
    high_total = sum(1 for cat in all_checks.values() 
                    for check in cat.values() 
                    if isinstance(check, dict) and check.get('criticality') == 'HIGH')
    
    medium_pass = sum(1 for cat in all_checks.values() 
                     for check in cat.values() 
                     if isinstance(check, dict) and check.get('criticality') == 'MEDIUM' and check.get('passed'))
    medium_total = sum(1 for cat in all_checks.values() 
                      for check in cat.values() 
                      if isinstance(check, dict) and check.get('criticality') == 'MEDIUM')
    
    # Calculate grade
    high_score = (high_pass / high_total * 100) if high_total > 0 else 100
    medium_score = (medium_pass / medium_total * 100) if medium_total > 0 else 100
    overall_score = 0.7 * high_score + 0.3 * medium_score
    
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B+"
    elif overall_score >= 70:
        grade = "B"
    elif overall_score >= 60:
        grade = "C+"
    else:
        grade = "C"
    
    # Generate Markdown report
    report = f"""# Lab Self-Audit Report
**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Overall Score**: {overall_score:.1f}/100  
**Grade**: {grade}

## Summary

| Criticality | Passed | Total | Score |
|-------------|--------|-------|-------|
| **HIGH** | {high_pass} | {high_total} | {high_score:.1f}% |
| **MEDIUM** | {medium_pass} | {medium_total} | {medium_score:.1f}% |

## Detailed Results

"""
    
    for category, checks in all_checks.items():
        report += f"\n### {category.replace('_', ' ').title()}\n\n"
        for check_name, check_data in checks.items():
            if isinstance(check_data, dict) and 'passed' in check_data:
                status = "✅ PASS" if check_data['passed'] else "❌ FAIL"
                crit = check_data.get('criticality', 'N/A')
                report += f"- **{check_name.replace('_', ' ').title()}**: {status} (Criticality: {crit})\n"
    
    report += "\n## Prioritized Action Items\n\n"
    
    # Extract failed HIGH criticality checks
    high_failures = []
    for category, checks in all_checks.items():
        for check_name, check_data in checks.items():
            if isinstance(check_data, dict) and check_data.get('criticality') == 'HIGH' and not check_data.get('passed'):
                high_failures.append((category, check_name, check_data))
    
    if high_failures:
        report += "### HIGH Priority (Must Fix)\n\n"
        for cat, name, data in high_failures:
            report += f"- **{name.replace('_', ' ').title()}** ({cat})\n"
    else:
        report += "✅ No HIGH priority failures!\n\n"
    
    # Write report
    output_path.write_text(report)
    print(f"\n📄 Audit report saved to: {output_path}")
    
    return overall_score, grade

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Lab Self-Audit Validation')
    parser.add_argument('--full', action='store_true', help='Run full audit (includes slow checks)')
    parser.add_argument('--quick', action='store_true', help='Run quick audit only')
    args = parser.parse_args()
    
    repo_path = Path(__file__).parent.parent
    results_path = repo_path / "evidence" / "phase10" / "tier2_clean" / "results.json"
    output_path = repo_path / "evidence" / "phase10" / "tier2_clean" / "AUDIT_REPORT.md"
    
    print(f"\n{Colors.INFO}{'='*70}")
    print(f"LAB SELF-AUDIT VALIDATION")
    print(f"{'='*70}{Colors.RESET}\n")
    
    all_checks = {}
    
    # Run checks
    if results_path.exists():
        all_checks['statistical_robustness'] = check_statistical_power(results_path)
        all_checks['baseline_coverage'] = check_baseline_coverage(results_path)
    else:
        print(f"{Colors.FAIL}❌ Results file not found: {results_path}{Colors.RESET}")
        sys.exit(1)
    
    all_checks['reproducibility'] = check_reproducibility(repo_path)
    all_checks['physics_interpretability'] = check_physics_interpretability(repo_path)
    
    # Generate report
    score, grade = generate_audit_report(all_checks, output_path)
    
    # Save machine-readable results
    json_output = repo_path / "evidence" / "phase10" / "tier2_clean" / "audit_results.json"
    with open(json_output, 'w') as f:
        json.dump({
            'timestamp': pd.Timestamp.now().isoformat(),
            'overall_score': score,
            'grade': grade,
            'checks': all_checks
        }, f, indent=2)
    
    print(f"\n{Colors.INFO}{'='*70}")
    print(f"AUDIT COMPLETE")
    print(f"{'='*70}{Colors.RESET}")
    print(f"Overall Score: {score:.1f}/100")
    print(f"Grade: {grade}")
    print(f"\nReports saved:")
    print(f"  - Markdown: {output_path}")
    print(f"  - JSON: {json_output}")
    
    # Exit with appropriate code
    if score >= 80:
        sys.exit(0)
    elif score >= 60:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Failure

if __name__ == '__main__':
    main()
