#!/usr/bin/env python3
"""
Merge multiple benchmark result files.

Usage:
    python scripts/merge_benchmark_results.py \
        --input1 evidence/phase10/tier2_clean/results.json \
        --input2 evidence/phase10/tier2_seeds_47-61/results.json \
        --output evidence/phase10/tier2_20seeds/results.json
"""

import json
import argparse
from pathlib import Path
import numpy as np

def merge_results(input1: Path, input2: Path, output: Path):
    """Merge two benchmark result files"""
    
    # Load both result files
    with open(input1) as f:
        results1 = json.load(f)
    with open(input2) as f:
        results2 = json.load(f)
    
    # Verify compatibility
    assert results1['dataset'] == results2['dataset'], "Datasets don't match"
    
    # Merge results
    merged = {
        'experiment': f"{results1['experiment']} (merged)",
        'dataset': results1['dataset'],
        'n_seeds': results1['n_seeds'] + results2['n_seeds'],
        'timestamp': results2['timestamp'],  # Use latest timestamp
        'results': {}
    }
    
    # Merge each strategy
    for strategy in results1['results'].keys():
        if strategy not in results2['results']:
            print(f"Warning: {strategy} not in second results file, skipping")
            continue
        
        # Combine RMSE histories
        histories1 = results1['results'][strategy]['rmse_histories']
        histories2 = results2['results'][strategy]['rmse_histories']
        combined_histories = histories1 + histories2
        
        # Compute combined statistics
        final_rmses = [h[-1] for h in combined_histories]
        
        merged['results'][strategy] = {
            'mean_rmse': float(np.mean(final_rmses)),
            'std_rmse': float(np.std(final_rmses)),
            'min_rmse': float(np.min(final_rmses)),
            'max_rmse': float(np.max(final_rmses)),
            'rmse_histories': combined_histories
        }
        
        print(f"{strategy.upper()}: {len(combined_histories)} seeds, "
              f"RMSE = {merged['results'][strategy]['mean_rmse']:.2f} ± "
              f"{merged['results'][strategy]['std_rmse']:.2f} K")
    
    # Recompute comparisons
    if 'dkl' in merged['results'] and 'random' in merged['results']:
        dkl_rmse = [h[-1] for h in merged['results']['dkl']['rmse_histories']]
        random_rmse = [h[-1] for h in merged['results']['random']['rmse_histories']]
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(dkl_rmse, random_rmse, equal_var=False)
        improvement = ((np.mean(random_rmse) - np.mean(dkl_rmse)) / np.mean(random_rmse)) * 100
        
        merged['comparisons'] = {
            'dkl_vs_random': {
                'improvement_percent': float(improvement),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        }
    
    if 'dkl' in merged['results'] and 'gp' in merged['results']:
        dkl_rmse = [h[-1] for h in merged['results']['dkl']['rmse_histories']]
        gp_rmse = [h[-1] for h in merged['results']['gp']['rmse_histories']]
        
        t_stat, p_value = stats.ttest_ind(dkl_rmse, gp_rmse, equal_var=False)
        improvement = ((np.mean(gp_rmse) - np.mean(dkl_rmse)) / np.mean(gp_rmse)) * 100
        
        merged['comparisons']['dkl_vs_gp'] = {
            'improvement_percent': float(improvement),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
        
        print(f"\nDKL vs GP: {improvement:.1f}% improvement, p={p_value:.4f} "
              f"({'significant' if p_value < 0.05 else 'not significant'})")
    
    # Save merged results
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n✅ Merged results saved to: {output}")
    print(f"   Total seeds: {merged['n_seeds']}")

def main():
    parser = argparse.ArgumentParser(description='Merge benchmark results')
    parser.add_argument('--input1', type=Path, required=True, help='First results file')
    parser.add_argument('--input2', type=Path, required=True, help='Second results file')
    parser.add_argument('--output', type=Path, required=True, help='Output merged file')
    args = parser.parse_args()
    
    merge_results(args.input1, args.input2, args.output)

if __name__ == '__main__':
    main()

