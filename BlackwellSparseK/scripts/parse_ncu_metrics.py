#!/usr/bin/env python3
"""
Parse Nsight Compute CSV metrics to JSON for regression tracking
Usage: python parse_ncu_metrics.py --csv bench.csv --output metrics.json
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime


def parse_ncu_csv(csv_path):
    """Parse Nsight Compute CSV export"""
    metrics = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract key metrics
            metrics['kernel_name'] = row.get('Kernel Name', 'unknown')
            
            # Timing
            if 'gpu__time_duration.sum' in row:
                metrics['gpu_time_us'] = float(row['gpu__time_duration.sum']) / 1000  # ns to μs
            if 'sm__cycles_elapsed.avg' in row:
                metrics['sm_cycles_avg'] = float(row['sm__cycles_elapsed.avg'])
            
            # Compute
            if 'sm__warps_active.avg.pct_of_peak_sustained_active' in row:
                metrics['warp_active_pct'] = float(row['sm__warps_active.avg.pct_of_peak_sustained_active'])
            if 'derived__sm__sass_active_cycles_pct' in row:
                metrics['sm_efficiency_pct'] = float(row['derived__sm__sass_active_cycles_pct'])
            
            # Memory
            if 'dram__throughput.avg.pct_of_peak_sustained_elapsed' in row:
                metrics['dram_throughput_pct'] = float(row['dram__throughput.avg.pct_of_peak_sustained_elapsed'])
            if 'dram__bytes.sum' in row:
                dram_bytes = float(row['dram__bytes.sum'])
                metrics['dram_bytes'] = dram_bytes
                metrics['dram_gb'] = dram_bytes / 1e9
            
            # Global memory
            if 'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum' in row:
                global_ld = float(row['l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum'])
                global_st = float(row.get('l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum', 0))
                metrics['global_memory_bytes'] = global_ld + global_st
                metrics['global_memory_gb'] = (global_ld + global_st) / 1e9
            
            # Tensor Core
            if 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active' in row:
                metrics['tensor_core_active_pct'] = float(row['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'])
            if 'smsp__inst_executed_pipe_tensor.sum' in row:
                metrics['tensor_ops_count'] = int(float(row['smsp__inst_executed_pipe_tensor.sum']))
            
            # FP operations
            fadd = float(row.get('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum', 0))
            fmul = float(row.get('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum', 0))
            ffma = float(row.get('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum', 0))
            
            total_flops = fadd + fmul + (ffma * 2)  # FFMA = 2 ops
            if 'gpu_time_us' in metrics and metrics['gpu_time_us'] > 0:
                metrics['compute_tflops'] = (total_flops / 1e12) / (metrics['gpu_time_us'] / 1e6)
            
            # FP8 operations
            if 'sm__sass_inst_executed_op_fp8_pred_on.sum' in row:
                metrics['fp8_ops_count'] = int(float(row['sm__sass_inst_executed_op_fp8_pred_on.sum']))
            
            # L2 cache
            if 'l2_tex_read_throughput.avg.pct_of_peak_sustained_elapsed' in row:
                metrics['l2_read_throughput_pct'] = float(row['l2_tex_read_throughput.avg.pct_of_peak_sustained_elapsed'])
            if 'l2_tex_write_throughput.avg.pct_of_peak_sustained_elapsed' in row:
                metrics['l2_write_throughput_pct'] = float(row['l2_tex_write_throughput.avg.pct_of_peak_sustained_elapsed'])
            
            # 128-bit memory operations
            if 'sm__sass_inst_executed_op_memory_128b.sum' in row:
                metrics['memory_128b_ops'] = int(float(row['sm__sass_inst_executed_op_memory_128b.sum']))
    
    # Derived metric: kernel duration (use GPU time)
    if 'gpu_time_us' in metrics:
        metrics['kernel_duration_us'] = metrics['gpu_time_us']
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Parse Nsight Compute metrics to JSON')
    parser.add_argument('--csv', required=True, help='Input CSV file from ncu')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--timestamp', default=None, help='Timestamp for this run')
    parser.add_argument('--comprehensive', action='store_true', help='Comprehensive profiling mode')
    
    args = parser.parse_args()
    
    # Parse CSV
    metrics = parse_ncu_csv(args.csv)
    
    # Add metadata
    run_data = {
        'timestamp': args.timestamp or datetime.now().isoformat(),
        'csv_file': str(args.csv),
        'comprehensive': args.comprehensive,
        **metrics
    }
    
    # Load existing JSON or create new
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        data = {
            'project': 'BlackwellSparseK',
            'runs': []
        }
    
    # Append new run
    data['runs'].append(run_data)
    
    # Keep last 100 runs
    data['runs'] = data['runs'][-100:]
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Metrics saved to {output_path}")
    print(f"📊 Total runs: {len(data['runs'])}")
    
    # Check for regressions (if > 1 run)
    if len(data['runs']) > 1:
        prev = data['runs'][-2]
        curr = data['runs'][-1]
        
        print("\n📈 Regression Check:")
        
        # Check SM efficiency
        if 'sm_efficiency_pct' in prev and 'sm_efficiency_pct' in curr:
            prev_eff = prev['sm_efficiency_pct']
            curr_eff = curr['sm_efficiency_pct']
            delta = curr_eff - prev_eff
            status = "✅" if delta >= 0 else "⚠️ "
            print(f"  {status} SM Efficiency: {prev_eff:.1f}% → {curr_eff:.1f}% ({delta:+.1f}%)")
        
        # Check kernel duration
        if 'kernel_duration_us' in prev and 'kernel_duration_us' in curr:
            prev_dur = prev['kernel_duration_us']
            curr_dur = curr['kernel_duration_us']
            speedup = prev_dur / curr_dur if curr_dur > 0 else 1.0
            status = "✅" if speedup >= 1.0 else "⚠️ "
            print(f"  {status} Kernel Duration: {prev_dur:.2f}μs → {curr_dur:.2f}μs ({speedup:.2f}x)")
        
        # Check DRAM throughput
        if 'dram_throughput_pct' in prev and 'dram_throughput_pct' in curr:
            prev_dram = prev['dram_throughput_pct']
            curr_dram = curr['dram_throughput_pct']
            delta = curr_dram - prev_dram
            status = "✅" if delta >= 0 else "⚠️ "
            print(f"  {status} DRAM Throughput: {prev_dram:.1f}% → {curr_dram:.1f}% ({delta:+.1f}%)")


if __name__ == '__main__':
    main()

