#!/usr/bin/env python3
"""Aggregate benchmark results to CSV and markdown."""
import json
import csv
from pathlib import Path
from collections import defaultdict

def load_results(jsonl_path):
    """Load results from JSONL file."""
    results = []
    if not jsonl_path.exists():
        return results
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def main():
    bench_dir = Path(__file__).parent
    out_dir = bench_dir / "out"
    
    # Load all results
    all_results = []
    for jsonl_file in out_dir.glob("*_results.jsonl"):
        all_results.extend(load_results(jsonl_file))
    
    if not all_results:
        print("No results found. Run benchmark scripts first.")
        return
    
    # Group by configuration
    configs = defaultdict(list)
    for result in all_results:
        key = (result['B'], result['H'], result['S'], result['d'], result['dtype'], result['causal'])
        configs[key].append(result)
    
    # Write CSV
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['impl', 'B', 'H', 'S', 'd', 'dtype', 'causal', 'ms_mean', 'ms_std', 
                        'qps', 'gpu_name', 'sm', 'driver', 'cuda', 'torch', 'commit'])
        
        for result in all_results:
            writer.writerow([
                result['impl'], result['B'], result['H'], result['S'], result['d'],
                result['dtype'], result['causal'], f"{result['ms_mean']:.4f}", 
                f"{result['ms_std']:.4f}", f"{result['qps']:.0f}",
                result.get('gpu_name', ''), result.get('sm', ''), 
                result.get('driver', ''), result.get('cuda', ''), 
                result.get('torch', ''), result.get('commit', '')
            ])
    
    print(f"CSV written to: {csv_path}")
    
    # Generate markdown table
    md_lines = []
    md_lines.append("# Benchmark Results Comparison\n")
    
    for config_key in sorted(configs.keys()):
        B, H, S, d, dtype, causal = config_key
        results = configs[config_key]
        
        # Find best
        best_ms = min(r['ms_mean'] for r in results)
        
        md_lines.append(f"\n## Configuration: B={B}, H={H}, S={S}, d={d}, dtype={dtype}, causal={causal}\n")
        md_lines.append("| Implementation | Latency (ms) | Queries/sec | Speedup vs Best |")
        md_lines.append("|----------------|--------------|-------------|-----------------|")
        
        for result in sorted(results, key=lambda x: x['ms_mean']):
            impl = result['impl']
            ms_mean = result['ms_mean']
            ms_std = result['ms_std']
            qps = result['qps']
            speedup = best_ms / ms_mean
            
            best_marker = " **BEST**" if ms_mean == best_ms else ""
            md_lines.append(f"| {impl} | {ms_mean:.4f} ± {ms_std:.4f} | {qps:.0f} | {speedup:.3f}×{best_marker} |")
    
    md_path = out_dir / "comparison.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Markdown written to: {md_path}")
    
    # Print summary
    print("\n=== Summary ===")
    for config_key in sorted(configs.keys()):
        B, H, S, d, dtype, causal = config_key
        results = configs[config_key]
        best_ms = min(r['ms_mean'] for r in results)
        best_impl = next(r['impl'] for r in results if r['ms_mean'] == best_ms)
        print(f"B={B}, H={H}, S={S}, d={d}, dtype={dtype}: Best={best_impl} ({best_ms:.4f} ms)")

if __name__ == "__main__":
    main()

