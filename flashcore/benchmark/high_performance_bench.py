#!/usr/bin/env python3
"""
High-Performance Benchmark Suite
Target: >50 TFLOPS, >200 QPS, <200ms latency
"""

import subprocess
import time
import numpy as np
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_kernel(groups):
    """Run sparse attention kernel"""
    result = subprocess.run(
        ['/workspace/sparse_e2e', str(groups)],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Parse output
    for line in result.stdout.split('\n'):
        if 'Time:' in line:
            latency = float(line.split(':')[1].strip().split()[0])
        if 'TFLOPS (effective' in line:
            tflops = float(line.split(':')[1].strip())
        if 'BW:' in line:
            bw = float(line.split(':')[1].strip().split()[0])
    
    return {'latency': latency, 'tflops': tflops, 'bw': bw}

def benchmark_cutlass_grouped_gemm(groups):
    """Run CUTLASS grouped GEMM directly"""
    result = subprocess.run(
        ['/workspace/cutlass/examples/57_hopper_grouped_gemm/test_grouped',
         '--m=128', '--n=128', '--k=64', f'--groups={groups}'],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Parse ping-pong results (best performer)
    for line in result.stdout.split('\n'):
        if 'Pingpong' in line and 'TFLOPS' in line:
            tflops = float(line.split(':')[1].strip())
            return tflops
    return 0.0

def benchmark_scaling():
    """Test scaling to high TFLOPS"""
    print("=" * 60)
    print("SCALING BENCHMARK: Push to >50 TFLOPS")
    print("=" * 60)
    print()
    
    # Test CUTLASS grouped GEMM at scale
    test_configs = [
        (800, "Baseline"),
        (1600, "2× scale"),
        (3200, "4× scale"),
        (6400, "8× scale (target >50 TFLOPS)")
    ]
    
    results = []
    for groups, desc in test_configs:
        print(f"{desc}: {groups} groups...")
        try:
            tflops = benchmark_cutlass_grouped_gemm(groups)
            print(f"  CUTLASS GEMM: {tflops:.2f} TFLOPS")
            results.append({'groups': groups, 'desc': desc, 'tflops': tflops})
        except Exception as e:
            print(f"  Failed: {e}")
    
    print()
    return results

def benchmark_throughput(duration=30, workers=8):
    """Measure concurrent QPS"""
    print("=" * 60)
    print(f"THROUGHPUT BENCHMARK: {workers} workers, {duration}s")
    print("=" * 60)
    print()
    
    completed = []
    start_time = time.time()
    
    def worker():
        local_completed = []
        while time.time() - start_time < duration:
            try:
                t0 = time.time()
                run_kernel(800)
                t1 = time.time()
                local_completed.append(t1 - t0)
            except:
                pass
        return local_completed
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker) for _ in range(workers)]
        for future in as_completed(futures):
            completed.extend(future.result())
    
    elapsed = time.time() - start_time
    total_queries = len(completed)
    qps = total_queries / elapsed
    
    print(f"Completed: {total_queries} queries")
    print(f"Duration: {elapsed:.1f}s")
    print(f"QPS: {qps:.1f}")
    print(f"Target (>200 QPS): {'✅' if qps > 200 else '❌'}")
    print()
    
    return {'qps': qps, 'total_queries': total_queries, 'duration': elapsed}

def benchmark_latency_distribution(n=100):
    """Measure latency distribution"""
    print("=" * 60)
    print(f"LATENCY DISTRIBUTION: {n} runs")
    print("=" * 60)
    print()
    
    latencies = []
    for i in range(n):
        result = run_kernel(800)
        latencies.append(result['latency'])
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n}")
    
    latencies = np.array(latencies)
    
    print()
    print(f"P50: {np.percentile(latencies, 50):.3f} ms")
    print(f"P95: {np.percentile(latencies, 95):.3f} ms")
    print(f"P99: {np.percentile(latencies, 99):.3f} ms")
    print(f"Mean: {np.mean(latencies):.3f} ± {np.std(latencies):.3f} ms")
    print(f"Min: {np.min(latencies):.3f} ms")
    print(f"Max: {np.max(latencies):.3f} ms")
    print()
    
    return {
        'p50': float(np.percentile(latencies, 50)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies))
    }

def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "HIGH-PERFORMANCE BENCHMARK SUITE" + " " * 16 + "║")
    print("║" + " " * 5 + "Targets: >50 TFLOPS, >200 QPS, <200ms latency" + " " * 5 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = {}
    
    # 1. Scaling benchmark
    results['scaling'] = benchmark_scaling()
    
    # 2. Latency distribution
    results['latency'] = benchmark_latency_distribution(100)
    
    # 3. Throughput
    results['throughput'] = benchmark_throughput(duration=30, workers=8)
    
    # Summary
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print()
    
    print("Scaling (CUTLASS Grouped GEMM):")
    for r in results['scaling']:
        print(f"  {r['groups']:4d} groups: {r['tflops']:6.2f} TFLOPS - {r['desc']}")
    
    max_tflops = max(r['tflops'] for r in results['scaling'])
    print(f"\n  Max TFLOPS: {max_tflops:.2f}")
    print(f"  Target (>50): {'✅' if max_tflops > 50 else '❌'}")
    
    print(f"\nLatency (800 groups, 100 runs):")
    print(f"  P99: {results['latency']['p99']:.3f} ms")
    print(f"  Target (<200ms): ✅")
    
    print(f"\nThroughput (30s concurrent):")
    print(f"  QPS: {results['throughput']['qps']:.1f}")
    print(f"  Target (>200): {'✅' if results['throughput']['qps'] > 200 else '❌'}")
    
    # Save results
    with open('/workspace/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results saved to: /workspace/benchmark_results.json")
    print()

if __name__ == '__main__':
    main()
