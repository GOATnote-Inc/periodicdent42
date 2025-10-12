#!/usr/bin/env python3
"""
Pattern 10: Expert Profiling Decision Tree
Version: 1.0
Created: October 2025

Automated profiling workflow for CUDA kernels.
Determines which profiling tool to use and analyzes bottlenecks.
"""

import subprocess
import re
import json
import sys
from typing import Dict, Optional, Tuple
from pathlib import Path


class ProfilingDecisionTree:
    """Expert profiling workflow for CUDA kernels"""
    
    # Threshold constants
    CRITICAL_SPEEDUP = 0.5
    TARGET_SPEEDUP = 1.0
    EXCELLENT_SPEEDUP = 1.5
    LAUNCH_OVERHEAD_THRESHOLD_MS = 0.010  # 10 microseconds
    MEMORY_BW_THRESHOLD = 0.7  # 70%
    COMPUTE_THRESHOLD = 0.5  # 50%
    OCCUPANCY_THRESHOLD = 0.5  # 50%
    
    @staticmethod
    def should_profile(speedup: float, kernel_time_ms: float) -> Dict[str, str]:
        """
        Determine profiling tool and urgency based on performance metrics.
        
        Args:
            speedup: Speedup vs PyTorch baseline (1.0 = same speed)
            kernel_time_ms: Kernel execution time in milliseconds
            
        Returns:
            Dictionary with keys: 'decision', 'tool', 'urgency', 'reason'
        """
        
        if speedup < ProfilingDecisionTree.CRITICAL_SPEEDUP:
            return {
                'decision': 'PROFILE NOW',
                'tool': 'ncu (Nsight Compute)',
                'urgency': 'CRITICAL',
                'reason': f'Speedup {speedup:.2f}× < {ProfilingDecisionTree.CRITICAL_SPEEDUP}× - kernel is broken or severely suboptimal'
            }
        
        elif speedup < ProfilingDecisionTree.TARGET_SPEEDUP:
            if kernel_time_ms < ProfilingDecisionTree.LAUNCH_OVERHEAD_THRESHOLD_MS:
                return {
                    'decision': 'PROFILE NOW',
                    'tool': 'nsys (Nsight Systems)',
                    'urgency': 'HIGH',
                    'reason': f'Kernel time {kernel_time_ms*1000:.1f}μs < 10μs - launch overhead likely dominates'
                }
            else:
                return {
                    'decision': 'PROFILE NOW',
                    'tool': 'ncu (Nsight Compute)',
                    'urgency': 'HIGH',
                    'reason': f'Speedup {speedup:.2f}× < 1.0× - kernel has performance bottleneck'
                }
        
        elif speedup >= ProfilingDecisionTree.TARGET_SPEEDUP and speedup < ProfilingDecisionTree.EXCELLENT_SPEEDUP:
            return {
                'decision': 'PROFILE OPTIONAL',
                'tool': 'ncu (Nsight Compute)',
                'urgency': 'MEDIUM',
                'reason': f'Speedup {speedup:.2f}× is good, but profiling may reveal incremental improvements'
            }
        
        else:  # speedup >= EXCELLENT_SPEEDUP
            return {
                'decision': 'SUCCESS - NO PROFILING NEEDED',
                'tool': 'None (unless targeting > 2.0×)',
                'urgency': 'LOW',
                'reason': f'Speedup {speedup:.2f}× is excellent'
            }
    
    @staticmethod
    def parse_ncu_csv(csv_output: str) -> Dict[str, float]:
        """
        Parse Nsight Compute CSV output to extract key metrics.
        
        Args:
            csv_output: CSV output from ncu --csv
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        lines = csv_output.strip().split('\n')
        for line in lines:
            # Skip header and empty lines
            if not line or line.startswith('ID') or line.startswith('"ID"'):
                continue
            
            # Memory bandwidth
            if 'dram__throughput.avg.pct_of_peak' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    metrics['memory_bw'] = float(match.group(1)) / 100.0
            
            # Compute throughput
            elif 'sm__throughput.avg.pct_of_peak' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    metrics['compute'] = float(match.group(1)) / 100.0
            
            # Kernel duration
            elif 'gpu__time_duration.sum' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    # Convert to milliseconds (assume input is microseconds)
                    metrics['duration_ms'] = float(match.group(1)) / 1000.0
            
            # Occupancy
            elif 'sm__warps_active.avg.pct_of_peak' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    metrics['occupancy'] = float(match.group(1)) / 100.0
            
            # Bank conflicts
            elif 'l1tex__data_bank_conflicts_pipe_lsu' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    metrics['bank_conflicts'] = float(match.group(1))
        
        return metrics
    
    @staticmethod
    def analyze_ncu_report(ncu_report_path: str) -> Dict:
        """
        Parse Nsight Compute report and identify bottleneck.
        
        Args:
            ncu_report_path: Path to .ncu-rep file
            
        Returns:
            Dictionary with keys: 'metrics', 'bottleneck', 'recommendation', 'priority'
        """
        
        # Run ncu --import to get CSV output
        try:
            cmd = f"ncu --import {ncu_report_path} --csv"
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            return {
                'error': f'Failed to parse NCU report: {e}',
                'metrics': {},
                'bottleneck': 'unknown',
                'recommendation': 'Manually inspect report with ncu-ui',
                'priority': 'N/A'
            }
        
        # Parse metrics
        metrics = ProfilingDecisionTree.parse_ncu_csv(output)
        
        if not metrics:
            return {
                'error': 'No metrics found in NCU report',
                'metrics': {},
                'bottleneck': 'unknown',
                'recommendation': 'Re-run profiling with: ncu --set full ...',
                'priority': 'N/A'
            }
        
        # Identify bottleneck
        bottleneck, priority = ProfilingDecisionTree._identify_bottleneck(metrics)
        recommendation = ProfilingDecisionTree._get_recommendation(bottleneck, metrics)
        
        return {
            'metrics': metrics,
            'bottleneck': bottleneck,
            'recommendation': recommendation,
            'priority': priority
        }
    
    @staticmethod
    def _identify_bottleneck(metrics: Dict[str, float]) -> Tuple[str, str]:
        """
        Identify primary bottleneck from metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Tuple of (bottleneck_name, priority_level)
        """
        mem_bw = metrics.get('memory_bw', 0)
        compute = metrics.get('compute', 0)
        duration_ms = metrics.get('duration_ms', 0)
        occupancy = metrics.get('occupancy', 0)
        bank_conflicts = metrics.get('bank_conflicts', 0)
        
        # Check for launch overhead (highest priority)
        if duration_ms > 0 and duration_ms < ProfilingDecisionTree.LAUNCH_OVERHEAD_THRESHOLD_MS:
            return "launch_overhead", "P0"
        
        # Check for bank conflicts (can severely degrade performance)
        if bank_conflicts > 1000:
            return "bank_conflicts", "P0"
        
        # Check for memory bound (very common in attention kernels)
        if mem_bw < 0.5:
            return "memory_bound", "P0"
        
        # Check for compute bound
        if mem_bw > ProfilingDecisionTree.MEMORY_BW_THRESHOLD and compute < ProfilingDecisionTree.COMPUTE_THRESHOLD:
            return "compute_bound", "P1"
        
        # Check for low occupancy
        if occupancy > 0 and occupancy < ProfilingDecisionTree.OCCUPANCY_THRESHOLD:
            return "low_occupancy", "P1"
        
        # Both memory and compute are high - balanced or near-optimal
        if mem_bw > ProfilingDecisionTree.MEMORY_BW_THRESHOLD and compute > ProfilingDecisionTree.MEMORY_BW_THRESHOLD:
            return "balanced_or_optimal", "P2"
        
        return "unknown", "P2"
    
    @staticmethod
    def _get_recommendation(bottleneck: str, metrics: Dict[str, float]) -> str:
        """
        Get optimization recommendation based on bottleneck.
        
        Args:
            bottleneck: Identified bottleneck type
            metrics: Performance metrics
            
        Returns:
            Detailed recommendation string
        """
        recommendations = {
            "launch_overhead": 
                "CRITICAL: Launch overhead dominates ({duration_ms:.3f}ms < 10μs).\n"
                "  Fix: (1) Increase tile size (e.g., 64×64 → 128×128)\n"
                "       (2) Fuse multiple kernel launches\n"
                "       (3) Use persistent kernels\n"
                "  Impact: 2-5× speedup possible",
            
            "bank_conflicts": 
                "CRITICAL: Shared memory bank conflicts detected ({bank_conflicts:.0f} conflicts).\n"
                "  Fix: (1) Pad shared memory arrays to avoid bank conflicts\n"
                "       (2) Change access pattern (transpose, swizzle)\n"
                "       (3) Use __ldg() for read-only data\n"
                "  Impact: 2-3× speedup possible",
            
            "memory_bound": 
                "CRITICAL: Memory bandwidth underutilized ({memory_bw:.1%} of peak).\n"
                "  Fix: (1) Vectorize memory access (use float4 for coalesced loads)\n"
                "       (2) Improve memory coalescing (sequential access pattern)\n"
                "       (3) Increase data reuse with shared memory tiling\n"
                "       (4) Use async memory copies (cuda::memcpy_async)\n"
                "  Impact: 2-4× speedup possible",
            
            "compute_bound": 
                "Compute throughput is low ({compute:.1%} of peak) while memory is high.\n"
                "  Fix: (1) Use Tensor Cores (WMMA for FP16/BF16)\n"
                "       (2) Reduce FP64 operations (use FP32 or FP16)\n"
                "       (3) Vectorize math operations\n"
                "       (4) Reduce expensive operations (div, sqrt, exp)\n"
                "  Impact: 1.5-3× speedup possible",
            
            "low_occupancy": 
                "Low occupancy detected ({occupancy:.1%} of peak warps active).\n"
                "  Fix: (1) Reduce register usage (check with --ptxas-options=-v)\n"
                "       (2) Reduce shared memory per block\n"
                "       (3) Decrease threads per block\n"
                "       (4) Use #pragma unroll sparingly\n"
                "  Impact: 1.2-2× speedup possible",
            
            "balanced_or_optimal": 
                "Good performance! Memory BW: {memory_bw:.1%}, Compute: {compute:.1%}\n"
                "  Further optimizations (diminishing returns):\n"
                "       (1) Algorithm-level optimizations (flash attention, etc.)\n"
                "       (2) Specialized kernels for specific configs\n"
                "       (3) Mixed precision (FP8 on H100)\n"
                "  Impact: 1.1-1.3× speedup possible",
            
            "unknown": 
                "Unable to identify clear bottleneck from automated analysis.\n"
                "  Action: (1) Open report in ncu-ui for manual inspection\n"
                "          (2) Check for: synchronization overhead, divergence, launch config\n"
                "          (3) Compare to CUDA Toolkit samples\n"
                "  Command: ncu-ui {report_path}"
        }
        
        template = recommendations.get(bottleneck, "Unknown bottleneck type")
        return template.format(**metrics)
    
    @staticmethod
    def print_analysis(analysis: Dict, verbose: bool = True):
        """Pretty print analysis results"""
        
        print("\n" + "="*70)
        print("📊 CUDA Kernel Profiling Analysis")
        print("="*70)
        
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        metrics = analysis['metrics']
        print("\n📈 Metrics:")
        if 'memory_bw' in metrics:
            print(f"  • Memory Bandwidth: {metrics['memory_bw']:.1%} of peak")
        if 'compute' in metrics:
            print(f"  • Compute Throughput: {metrics['compute']:.1%} of peak")
        if 'occupancy' in metrics:
            print(f"  • Occupancy: {metrics['occupancy']:.1%} of peak warps")
        if 'duration_ms' in metrics:
            print(f"  • Kernel Duration: {metrics['duration_ms']:.3f} ms")
        if 'bank_conflicts' in metrics:
            print(f"  • Bank Conflicts: {metrics['bank_conflicts']:.0f}")
        
        print(f"\n🎯 Bottleneck: {analysis['bottleneck'].upper()} ({analysis['priority']})")
        print(f"\n💡 Recommendation:")
        for line in analysis['recommendation'].split('\n'):
            print(f"  {line}")
        
        print("\n" + "="*70)


def main():
    """Command-line interface for profiling decision tree"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python profiling_decision_tree.py <speedup> <kernel_time_ms>")
        print("  python profiling_decision_tree.py analyze <ncu_report.ncu-rep>")
        print("\nExamples:")
        print("  python profiling_decision_tree.py 0.85 0.048")
        print("  python profiling_decision_tree.py analyze profile_s128.ncu-rep")
        sys.exit(1)
    
    if sys.argv[1] == 'analyze':
        # Analyze NCU report
        if len(sys.argv) < 3:
            print("Error: Please provide path to .ncu-rep file")
            sys.exit(1)
        
        report_path = sys.argv[2]
        if not Path(report_path).exists():
            print(f"Error: Report file not found: {report_path}")
            sys.exit(1)
        
        print(f"Analyzing: {report_path}")
        analysis = ProfilingDecisionTree.analyze_ncu_report(report_path)
        ProfilingDecisionTree.print_analysis(analysis)
        
    else:
        # Profiling decision
        try:
            speedup = float(sys.argv[1])
            kernel_time_ms = float(sys.argv[2])
        except ValueError:
            print("Error: speedup and kernel_time_ms must be numbers")
            sys.exit(1)
        
        decision = ProfilingDecisionTree.should_profile(speedup, kernel_time_ms)
        
        print("\n" + "="*70)
        print("🔍 Profiling Decision")
        print("="*70)
        print(f"\nInput:")
        print(f"  • Speedup: {speedup:.2f}×")
        print(f"  • Kernel Time: {kernel_time_ms:.3f} ms ({kernel_time_ms*1000:.1f} μs)")
        print(f"\n{decision['urgency']} - {decision['decision']}")
        print(f"Tool: {decision['tool']}")
        print(f"Reason: {decision['reason']}")
        print("\n" + "="*70)


if __name__ == "__main__":
    main()

