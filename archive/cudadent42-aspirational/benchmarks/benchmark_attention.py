"""
Performance benchmarks for FlashAttention-Science warp-specialized kernel.

Compares against:
  - PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
  - FlashAttention-2 (if available)
  - Basic implementation (our Day 1-6 kernel)

Generates:
  - Performance comparison graphs
  - Detailed timing statistics
  - Memory usage analysis
  - Speedup metrics

Usage:
    python benchmarks/benchmark_attention.py
    python benchmarks/benchmark_attention.py --save-results --output-dir results/

@author GOATnote Autonomous Research Lab Initiative
@date 2025-10-11
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Try to import our CUDA extension
try:
    import flashmoe_science_ext
    HAS_CUDA_EXT = True
except ImportError:
    print("WARNING: CUDA extension not built. Run setup.py first.")
    HAS_CUDA_EXT = False

# Try to import FlashAttention-2 for comparison
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("INFO: FlashAttention-2 not installed (optional for comparison)")

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_PLOTTING = False
    print("INFO: Matplotlib/Seaborn not installed (optional for graphs)")


class AttentionBenchmark:
    """Benchmark suite for attention kernels."""
    
    def __init__(self, device='cuda', dtype=torch.bfloat16, warmup_runs=5, benchmark_runs=20):
        """
        Initialize benchmark.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            dtype: Data type (torch.bfloat16 or torch.float16)
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
        """
        self.device = device
        self.dtype = dtype
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        # Check CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        if device == 'cuda':
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {self.gpu_name} ({self.gpu_memory:.2f} GB)")
        else:
            self.gpu_name = "CPU"
            self.gpu_memory = 0
    
    def create_inputs(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int) -> Tuple:
        """
        Create random input tensors.
        
        Returns:
            (Q, K, V) tuple of tensors
        """
        torch.manual_seed(42)  # Reproducible
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=self.dtype, device=self.device)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=self.dtype, device=self.device)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=self.dtype, device=self.device)
        
        return Q, K, V
    
    def benchmark_pytorch_sdpa(self, Q, K, V, causal=False) -> Dict:
        """Benchmark PyTorch SDPA."""
        softmax_scale = 1.0 / (Q.size(-1) ** 0.5)
        
        # Warmup
        for _ in range(self.warmup_runs):
            O = F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=softmax_scale)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(self.benchmark_runs):
            O = F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=softmax_scale)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / self.benchmark_runs
        
        # Memory usage
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        
        return {
            'name': 'PyTorch SDPA',
            'time_ms': avg_time * 1000,
            'memory_gb': mem_used,
            'throughput_tokens': Q.size(2) / avg_time,
        }
    
    def benchmark_warp_specialized(self, Q, K, V, causal=False) -> Dict:
        """Benchmark our warp-specialized kernel."""
        if not HAS_CUDA_EXT:
            return {'name': 'Warp Specialized', 'time_ms': float('nan'), 'memory_gb': 0, 'throughput_tokens': 0}
        
        softmax_scale = 1.0 / (Q.size(-1) ** 0.5)
        
        # Warmup
        for _ in range(self.warmup_runs):
            O = flashmoe_science_ext.flash_attention_warp_specialized(Q, K, V, causal=causal, softmax_scale=softmax_scale)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(self.benchmark_runs):
            O = flashmoe_science_ext.flash_attention_warp_specialized(Q, K, V, causal=causal, softmax_scale=softmax_scale)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / self.benchmark_runs
        
        # Memory usage
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        
        return {
            'name': 'Warp Specialized (Ours)',
            'time_ms': avg_time * 1000,
            'memory_gb': mem_used,
            'throughput_tokens': Q.size(2) / avg_time,
        }
    
    def benchmark_basic(self, Q, K, V, causal=False) -> Dict:
        """Benchmark our basic kernel (Day 1-6)."""
        if not HAS_CUDA_EXT:
            return {'name': 'Basic', 'time_ms': float('nan'), 'memory_gb': 0, 'throughput_tokens': 0}
        
        softmax_scale = 1.0 / (Q.size(-1) ** 0.5)
        
        try:
            # Warmup
            for _ in range(self.warmup_runs):
                O = flashmoe_science_ext.flash_attention_forward(Q, K, V, causal=causal, softmax_scale=softmax_scale)
                torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            for _ in range(self.benchmark_runs):
                O = flashmoe_science_ext.flash_attention_forward(Q, K, V, causal=causal, softmax_scale=softmax_scale)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / self.benchmark_runs
            
            # Memory usage
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            
            return {
                'name': 'Basic (Day 1-6)',
                'time_ms': avg_time * 1000,
                'memory_gb': mem_used,
                'throughput_tokens': Q.size(2) / avg_time,
            }
        except AttributeError:
            return {'name': 'Basic (Day 1-6)', 'time_ms': float('nan'), 'memory_gb': 0, 'throughput_tokens': 0}
    
    def benchmark_flashattn2(self, Q, K, V, causal=False) -> Dict:
        """Benchmark FlashAttention-2 (if available)."""
        if not HAS_FLASH_ATTN:
            return {'name': 'FlashAttention-2', 'time_ms': float('nan'), 'memory_gb': 0, 'throughput_tokens': 0}
        
        # FlashAttention-2 expects (batch, seq, heads, dim)
        Q_fa2 = Q.transpose(1, 2)  # (B, S, H, D)
        K_fa2 = K.transpose(1, 2)
        V_fa2 = V.transpose(1, 2)
        
        softmax_scale = 1.0 / (Q.size(-1) ** 0.5)
        
        # Warmup
        for _ in range(self.warmup_runs):
            O = flash_attn_func(Q_fa2, K_fa2, V_fa2, causal=causal, softmax_scale=softmax_scale)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(self.benchmark_runs):
            O = flash_attn_func(Q_fa2, K_fa2, V_fa2, causal=causal, softmax_scale=softmax_scale)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / self.benchmark_runs
        
        # Memory usage
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        
        return {
            'name': 'FlashAttention-2',
            'time_ms': avg_time * 1000,
            'memory_gb': mem_used,
            'throughput_tokens': Q.size(2) / avg_time,
        }
    
    def run_benchmark_suite(self, batch_size=4, num_heads=8, seq_lens=[512, 1024, 2048, 4096], head_dim=64, causal=False):
        """
        Run comprehensive benchmark across sequence lengths.
        
        Returns:
            List of benchmark results
        """
        results = []
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK CONFIGURATION")
        print(f"{'='*80}")
        print(f"Batch size:  {batch_size}")
        print(f"Num heads:   {num_heads}")
        print(f"Head dim:    {head_dim}")
        print(f"Seq lengths: {seq_lens}")
        print(f"Causal:      {causal}")
        print(f"Dtype:       {self.dtype}")
        print(f"{'='*80}\n")
        
        for seq_len in seq_lens:
            print(f"Benchmarking seq_len={seq_len}...")
            
            Q, K, V = self.create_inputs(batch_size, num_heads, seq_len, head_dim)
            
            # Run all benchmarks
            result = {
                'seq_len': seq_len,
                'batch_size': batch_size,
                'num_heads': num_heads,
                'head_dim': head_dim,
            }
            
            # PyTorch SDPA (baseline)
            print("  - PyTorch SDPA...", end=' ')
            result['pytorch'] = self.benchmark_pytorch_sdpa(Q, K, V, causal)
            print(f"{result['pytorch']['time_ms']:.2f} ms")
            
            # Our warp-specialized kernel
            print("  - Warp Specialized...", end=' ')
            result['warp_specialized'] = self.benchmark_warp_specialized(Q, K, V, causal)
            print(f"{result['warp_specialized']['time_ms']:.2f} ms")
            
            # Our basic kernel
            print("  - Basic...", end=' ')
            result['basic'] = self.benchmark_basic(Q, K, V, causal)
            if not np.isnan(result['basic']['time_ms']):
                print(f"{result['basic']['time_ms']:.2f} ms")
            else:
                print("N/A")
            
            # FlashAttention-2
            print("  - FlashAttention-2...", end=' ')
            result['flashattn2'] = self.benchmark_flashattn2(Q, K, V, causal)
            if not np.isnan(result['flashattn2']['time_ms']):
                print(f"{result['flashattn2']['time_ms']:.2f} ms")
            else:
                print("N/A")
            
            # Calculate speedups
            baseline_time = result['pytorch']['time_ms']
            result['speedup_vs_pytorch'] = baseline_time / result['warp_specialized']['time_ms']
            
            if not np.isnan(result['flashattn2']['time_ms']):
                result['speedup_vs_fa2'] = result['flashattn2']['time_ms'] / result['warp_specialized']['time_ms']
            else:
                result['speedup_vs_fa2'] = float('nan')
            
            print(f"  Speedup vs PyTorch: {result['speedup_vs_pytorch']:.2f}x")
            if not np.isnan(result['speedup_vs_fa2']):
                print(f"  Speedup vs FA2: {result['speedup_vs_fa2']:.2f}x")
            
            results.append(result)
            print()
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print summary table of results."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"{'Seq Len':<10} {'PyTorch':<12} {'Ours':<12} {'FA2':<12} {'Speedup':>10}")
        print(f"{'-'*80}")
        
        for result in results:
            seq_len = result['seq_len']
            pytorch_time = result['pytorch']['time_ms']
            ours_time = result['warp_specialized']['time_ms']
            fa2_time = result['flashattn2']['time_ms']
            speedup = result['speedup_vs_pytorch']
            
            fa2_str = f"{fa2_time:.2f} ms" if not np.isnan(fa2_time) else "N/A"
            
            print(f"{seq_len:<10} {pytorch_time:>10.2f} ms {ours_time:>10.2f} ms {fa2_str:<12} {speedup:>9.2f}x")
        
        print(f"{'='*80}\n")
    
    def plot_results(self, results: List[Dict], output_dir='results/'):
        """Generate performance comparison graphs."""
        if not HAS_PLOTTING:
            print("Plotting libraries not available. Skipping graphs.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        seq_lens = [r['seq_len'] for r in results]
        pytorch_times = [r['pytorch']['time_ms'] for r in results]
        ours_times = [r['warp_specialized']['time_ms'] for r in results]
        fa2_times = [r['flashattn2']['time_ms'] if not np.isnan(r['flashattn2']['time_ms']) else None for r in results]
        
        # Plot 1: Time comparison
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lens, pytorch_times, 'o-', label='PyTorch SDPA', linewidth=2, markersize=8)
        plt.plot(seq_lens, ours_times, 's-', label='Warp Specialized (Ours)', linewidth=2, markersize=8)
        
        if any(fa2_times):
            fa2_times_clean = [t if t is not None else 0 for t in fa2_times]
            plt.plot(seq_lens, fa2_times_clean, '^-', label='FlashAttention-2', linewidth=2, markersize=8)
        
        plt.xlabel('Sequence Length', fontsize=14)
        plt.ylabel('Time (ms)', fontsize=14)
        plt.title(f'Attention Kernel Performance on {self.gpu_name}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
        print(f"Saved: {os.path.join(output_dir, 'performance_comparison.png')}")
        
        # Plot 2: Speedup
        speedups = [r['speedup_vs_pytorch'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lens, speedups, 'o-', linewidth=2, markersize=8, color='green')
        plt.axhline(y=2.0, color='r', linestyle='--', label='Target: 2.0x')
        plt.xlabel('Sequence Length', fontsize=14)
        plt.ylabel('Speedup vs PyTorch SDPA', fontsize=14)
        plt.title(f'Speedup Achieved on {self.gpu_name}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=300)
        print(f"Saved: {os.path.join(output_dir, 'speedup_comparison.png')}")
        
        plt.close('all')
    
    def save_results(self, results: List[Dict], output_dir='results/', filename='benchmark_results.json'):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Add metadata
        output = {
            'metadata': {
                'gpu': self.gpu_name,
                'gpu_memory_gb': self.gpu_memory,
                'dtype': str(self.dtype),
                'warmup_runs': self.warmup_runs,
                'benchmark_runs': self.benchmark_runs,
            },
            'results': results,
        }
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved results: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark FlashAttention-Science kernels')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--head-dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[512, 1024, 2048, 4096],
                       help='Sequence lengths to benchmark')
    parser.add_argument('--causal', action='store_true', help='Use causal masking')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                       help='Data type')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    parser.add_argument('--output-dir', type=str, default='results/', help='Output directory')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--runs', type=int, default=20, help='Benchmark iterations')
    
    args = parser.parse_args()
    
    # Convert dtype
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    # Check CUDA extension
    if not HAS_CUDA_EXT:
        print("ERROR: CUDA extension not built. Run 'python setup.py build_ext --inplace' first.")
        return 1
    
    # Create benchmark
    benchmark = AttentionBenchmark(
        device='cuda',
        dtype=dtype,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs
    )
    
    # Run benchmark suite
    results = benchmark.run_benchmark_suite(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_lens=args.seq_lens,
        head_dim=args.head_dim,
        causal=args.causal
    )
    
    # Print summary
    benchmark.print_summary(results)
    
    # Save results
    if args.save_results:
        benchmark.save_results(results, args.output_dir)
        benchmark.plot_results(results, args.output_dir)
    
    return 0


if __name__ == '__main__':
    exit(main())

