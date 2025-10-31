#!/usr/bin/env python3
"""
Compare BlackwellSparseK vs FlashAttention-3 metrics
Usage: python compare_fa3_metrics.py --sparsek-csv s.csv --fa3-csv f.csv --output comp.json
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime


def parse_metrics_csv(csv_path):
    """Parse metrics from Nsight CSV"""
    metrics = {}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Timing
                if 'gpu__time_duration.sum' in row:
                    metrics['kernel_duration_us'] = float(row['gpu__time_duration.sum']) / 1000
                
                # Compute
                if 'sm__warps_active.avg.pct_of_peak_sustained_active' in row:
                    metrics['sm_efficiency_pct'] = float(row['sm__warps_active.avg.pct_of_peak_sustained_active'])
                
                # Memory
                if 'dram__throughput.avg.pct_of_peak_sustained_elapsed' in row:
                    metrics['dram_throughput_pct'] = float(row['dram__throughput.avg.pct_of_peak_sustained_elapsed'])
                
                # Tensor Core
                if 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active' in row:
                    metrics['tensor_core_active_pct'] = float(row['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'])
                if 'smsp__inst_executed_pipe_tensor.sum' in row:
                    metrics['tensor_ops_count'] = int(float(row['smsp__inst_executed_pipe_tensor.sum']))
    except Exception as e:
        print(f"Warning: Failed to parse {csv_path}: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compare SparseK vs FA3 metrics')
    parser.add_argument('--sparsek-csv', required=True, help='SparseK CSV file')
    parser.add_argument('--fa3-csv', required=True, help='FA3 CSV file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--timestamp', default=None, help='Timestamp for comparison')
    
    args = parser.parse_args()
    
    # Parse both CSVs
    sparsek_metrics = parse_metrics_csv(args.sparsek_csv)
    fa3_metrics = parse_metrics_csv(args.fa3_csv)
    
    # Calculate comparison
    comparison = {
        'timestamp': args.timestamp or datetime.now().isoformat(),
        'sparsek': sparsek_metrics,
        'fa3': fa3_metrics,
        'speedup': {},
        'verdict': 'Unknown'
    }
    
    # Calculate speedups
    if 'kernel_duration_us' in sparsek_metrics and 'kernel_duration_us' in fa3_metrics:
        sparsek_dur = sparsek_metrics['kernel_duration_us']
        fa3_dur = fa3_metrics['kernel_duration_us']
        if sparsek_dur > 0:
            comparison['speedup']['latency'] = fa3_dur / sparsek_dur
    
    # Determine verdict
    if comparison['speedup'].get('latency', 0) >= 0.9:
        if comparison['speedup']['latency'] >= 1.0:
            comparison['verdict'] = "🎯 PRODUCTION-VIABLE: SparseK >= FA3 baseline"
        else:
            comparison['verdict'] = "✅ TIER 2: SparseK >= 90% of FA3"
    else:
        comparison['verdict'] = "⚠️  NEEDS OPTIMIZATION: SparseK < 90% of FA3"
    
    # Load existing JSON or create new
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        data = {
            'project': 'BlackwellSparseK',
            'baseline': 'FlashAttention-3',
            'comparisons': []
        }
    
    # Append comparison
    data['comparisons'].append(comparison)
    data['comparisons'] = data['comparisons'][-50:]  # Keep last 50
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Comparison saved to {output_path}")
    print(f"📊 Total comparisons: {len(data['comparisons'])}")


if __name__ == '__main__':
    main()

