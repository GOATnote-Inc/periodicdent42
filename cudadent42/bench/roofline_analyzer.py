#!/usr/bin/env python3
"""
Roofline Model Analyzer
=======================
Determines if kernel is memory-bound or compute-bound.
Provides actionable optimization recommendations.
"""

from dataclasses import dataclass
from typing import Dict


# GPU Specifications Database
GPU_SPECS = {
    'A100-SXM4-80GB': {
        'peak_flops_fp32': 19500,
        'peak_flops_fp16': 312000,
        'peak_bandwidth': 1555
    },
    'A100-PCIe-80GB': {
        'peak_flops_fp32': 19500,
        'peak_flops_fp16': 312000,
        'peak_bandwidth': 1935
    },
    'H100-SXM5-80GB': {
        'peak_flops_fp32': 51000,
        'peak_flops_fp16': 756000,
        'peak_bandwidth': 2000
    },
    'V100-SXM2-32GB': {
        'peak_flops_fp32': 15700,
        'peak_flops_fp16': 125000,
        'peak_bandwidth': 900
    },
    'L4': {
        'peak_flops_fp32': 30300,
        'peak_flops_fp16': 121000,
        'peak_bandwidth': 300
    },
    'RTX4090': {
        'peak_flops_fp32': 82600,
        'peak_flops_fp16': 165000,
        'peak_bandwidth': 1008
    }
}


@dataclass
class RooflineResult:
    """Roofline analysis results"""
    arithmetic_intensity: float
    achieved_gflops: float
    achieved_bandwidth_gb_s: float
    theoretical_max_gflops: float
    efficiency_percent: float
    is_memory_bound: bool
    bottleneck: str
    recommendations: list


class RooflineAnalyzer:
    """
    Roofline model analyzer for CUDA kernels
    
    Determines performance bottleneck and provides recommendations.
    
    Usage:
        analyzer = RooflineAnalyzer(gpu_name="L4", dtype="fp16")
        result = analyzer.analyze(
            flop_count=1e9,
            memory_bytes=1e8,
            time_ms=10.0
        )
        analyzer.print_analysis(result)
    """
    
    def __init__(self, gpu_name: str, dtype: str = "fp32"):
        """
        Args:
            gpu_name: GPU model (e.g., "L4", "A100-SXM4-80GB")
            dtype: Data type ("fp32", "fp16", "bf16")
        """
        if gpu_name not in GPU_SPECS:
            # Try to match partial name
            matched = [k for k in GPU_SPECS.keys() if gpu_name in k]
            if matched:
                gpu_name = matched[0]
            else:
                raise ValueError(f"Unknown GPU: {gpu_name}. Available: {list(GPU_SPECS.keys())}")
        
        self.gpu_name = gpu_name
        self.dtype = dtype
        self.specs = GPU_SPECS[gpu_name]
        
        # Select appropriate FLOP count based on dtype
        if dtype in ["fp16", "bf16"]:
            self.peak_flops = self.specs['peak_flops_fp16']
        else:
            self.peak_flops = self.specs['peak_flops_fp32']
        
        self.peak_bandwidth = self.specs['peak_bandwidth']
    
    def analyze(
        self,
        flop_count: int,
        memory_bytes: int,
        time_ms: float
    ) -> RooflineResult:
        """
        Perform roofline analysis
        
        Args:
            flop_count: Total floating point operations
            memory_bytes: Total bytes accessed (read + write)
            time_ms: Measured kernel time in milliseconds
            
        Returns:
            RooflineResult with analysis and recommendations
        """
        # Compute metrics
        time_s = time_ms / 1000.0
        
        ai = flop_count / memory_bytes  # Arithmetic Intensity
        achieved_gflops = (flop_count / time_s) / 1e9
        achieved_bandwidth = (memory_bytes / time_s) / 1e9
        
        # Roofline model
        # Memory-bound performance = bandwidth × AI
        memory_bound_perf = self.peak_bandwidth * ai
        
        # Theoretical max is minimum of compute and memory roofs
        theoretical_max = min(self.peak_flops, memory_bound_perf)
        
        # Determine bottleneck
        is_memory_bound = memory_bound_perf < self.peak_flops
        
        # Efficiency
        efficiency = (achieved_gflops / theoretical_max) * 100
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            ai, efficiency, is_memory_bound, achieved_bandwidth, achieved_gflops
        )
        
        bottleneck = "Memory Bandwidth" if is_memory_bound else "Compute Throughput"
        
        return RooflineResult(
            arithmetic_intensity=ai,
            achieved_gflops=achieved_gflops,
            achieved_bandwidth_gb_s=achieved_bandwidth,
            theoretical_max_gflops=theoretical_max,
            efficiency_percent=efficiency,
            is_memory_bound=is_memory_bound,
            bottleneck=bottleneck,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        ai: float,
        efficiency: float,
        is_memory_bound: bool,
        achieved_bandwidth: float,
        achieved_gflops: float
    ) -> list:
        """Generate actionable recommendations"""
        recs = []
        
        # Bottleneck-specific recommendations
        if is_memory_bound:
            recs.append("MEMORY-BOUND: Focus on reducing memory traffic")
            recs.append("  - Fuse operations to increase arithmetic intensity")
            recs.append("  - Use shared memory to cache frequently accessed data")
            recs.append("  - Optimize memory coalescing (align accesses)")
            recs.append("  - Consider tiling/blocking strategies")
            
            if ai < 1.0:
                recs.append("  - Very low AI (<1): Element-wise operations")
                recs.append("    → Kernel fusion is critical")
        else:
            recs.append("COMPUTE-BOUND: Focus on ALU utilization")
            recs.append("  - Use tensor cores for matrix operations")
            recs.append("  - Reduce branch divergence in hot paths")
            recs.append("  - Optimize instruction-level parallelism")
            recs.append("  - Consider mixed precision (FP16/BF16)")
        
        # Efficiency-based recommendations
        if efficiency < 30:
            recs.append("LOW EFFICIENCY (<30%): Fundamental issues")
            recs.append("  - Profile with nsight compute for bottlenecks")
            recs.append("  - Check occupancy (registers/shared memory)")
            recs.append("  - Verify correct algorithm implementation")
        elif efficiency < 60:
            recs.append("MODERATE EFFICIENCY (30-60%): Room for optimization")
            recs.append("  - Profile memory access patterns")
            recs.append("  - Check for bank conflicts in shared memory")
            recs.append("  - Optimize thread block dimensions")
        elif efficiency > 80:
            recs.append("HIGH EFFICIENCY (>80%): Well-optimized")
            recs.append("  - Consider algorithmic improvements")
            recs.append("  - Minimal gains from micro-optimizations")
        
        # Bandwidth utilization
        if is_memory_bound:
            bw_util = (achieved_bandwidth / self.peak_bandwidth) * 100
            recs.append(f"Bandwidth utilization: {bw_util:.1f}% of peak")
            
            if bw_util < 50:
                recs.append("  - Poor memory coalescing likely")
                recs.append("  - Check access patterns with nsight compute")
        
        return recs
    
    def print_analysis(self, result: RooflineResult):
        """Print formatted analysis"""
        print(f"\n{'='*70}")
        print(f"ROOFLINE ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\nGPU: {self.gpu_name} ({self.dtype.upper()})")
        print(f"  Peak Compute:   {self.peak_flops:>8.0f} GFLOPS")
        print(f"  Peak Bandwidth: {self.peak_bandwidth:>8.0f} GB/s")
        
        print(f"\nKernel Characteristics:")
        print(f"  Arithmetic Intensity: {result.arithmetic_intensity:>8.2f} FLOP/Byte")
        print(f"  Achieved GFLOPS:      {result.achieved_gflops:>8.2f}")
        print(f"  Achieved Bandwidth:   {result.achieved_bandwidth_gb_s:>8.2f} GB/s")
        
        print(f"\nPerformance Limits:")
        print(f"  Theoretical Max:      {result.theoretical_max_gflops:>8.2f} GFLOPS")
        print(f"  Efficiency:           {result.efficiency_percent:>8.1f}%")
        
        # Visual bottleneck indicator
        marker = "MEMORY" if result.is_memory_bound else "COMPUTE"
        print(f"\nBottleneck: {marker}")
        
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  {rec}")
    
    def to_dict(self, result: RooflineResult) -> Dict:
        """Convert result to dictionary for JSON export"""
        return {
            'gpu': self.gpu_name,
            'dtype': self.dtype,
            'peak_flops_gflops': self.peak_flops,
            'peak_bandwidth_gb_s': self.peak_bandwidth,
            'arithmetic_intensity': result.arithmetic_intensity,
            'achieved_gflops': result.achieved_gflops,
            'achieved_bandwidth_gb_s': result.achieved_bandwidth_gb_s,
            'theoretical_max_gflops': result.theoretical_max_gflops,
            'efficiency_percent': result.efficiency_percent,
            'is_memory_bound': result.is_memory_bound,
            'bottleneck': result.bottleneck,
            'recommendations': result.recommendations
        }


if __name__ == "__main__":
    """Example usage"""
    
    # Example: Attention kernel on L4 GPU
    # Config: B=32, H=8, S=128, D=64, FP16
    
    # FLOP count: 4 * B * H * S * S * D
    flops = 4 * 32 * 8 * 128 * 128 * 64
    
    # Memory: Q, K, V, O (4 tensors, each B*H*S*D elements, 2 bytes each)
    memory_bytes = 4 * 32 * 8 * 128 * 64 * 2
    
    # Measured time (from integrated test)
    time_ms = 0.0526
    
    # Analyze
    analyzer = RooflineAnalyzer(gpu_name="L4", dtype="fp16")
    result = analyzer.analyze(flops, memory_bytes, time_ms)
    analyzer.print_analysis(result)

