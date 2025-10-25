#!/usr/bin/env python3
"""
Convert DKL ablation results to format expected by compute_ablation_stats_enhanced.py
Extracts final RMSE for each method/seed pair.
"""
import json
from pathlib import Path

def convert_ablation_results(input_path: Path, output_path: Path):
    """Convert ablation results to statistical analysis format."""
    with open(input_path) as f:
        data = json.load(f)
    
    # Extract results
    records = []
    for method_name, method_data in data['results'].items():
        for seed_result in method_data['seed_results']:
            records.append({
                'method': method_name,
                'seed': seed_result['seed'],
                'rmse': seed_result['final_rmse']
            })
    
    # Save in format expected by statistical framework
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"✓ Converted {len(records)} results")
    print(f"  Methods: {len(data['results'])} ({', '.join(data['results'].keys())})")
    print(f"  Seeds: {data['metadata']['n_seeds']} ({data['metadata']['seeds']})")
    print(f"✓ Saved to: {output_path}")
    
    return records

if __name__ == '__main__':
    input_path = Path('experiments/ablations/dkl_ablation_results_real.json')
    output_path = Path('experiments/ablations/dkl_ablation_for_stats.json')
    
    records = convert_ablation_results(input_path, output_path)
    
    # Preview
    print("\nPreview:")
    for record in records[:6]:
        print(f"  {record}")

