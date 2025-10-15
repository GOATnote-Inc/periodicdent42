"""
robust-kbench Reporter: Generate benchmark reports in multiple formats
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from .runner import BenchmarkResult


class BenchmarkReporter:
    """Generates benchmark reports in JSON, CSV, and Markdown"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, results: List[BenchmarkResult], filename: str = "rbk_report.json"):
        """Save results as JSON"""
        output_path = self.output_dir / filename
        
        data = {
            "results": [r.to_dict() for r in results],
            "summary": self._generate_summary(results),
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def save_csv(self, results: List[BenchmarkResult], filename: str = "rbk_report.csv"):
        """Save results as CSV"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "kernel_name", "shape", "B", "H", "S", "D", "causal", "dtype",
                "p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms", "tflops"
            ])
            
            # Data rows
            for r in results:
                s = r.shape
                writer.writerow([
                    r.kernel_name,
                    str(s),
                    s.batch_size,
                    s.num_heads,
                    s.seq_len,
                    s.head_dim,
                    s.causal,
                    s.dtype,
                    f"{r.p50_latency_ms:.4f}",
                    f"{r.p90_latency_ms:.4f}",
                    f"{r.p95_latency_ms:.4f}",
                    f"{r.p99_latency_ms:.4f}",
                    f"{r.mean_latency_ms:.4f}",
                    f"{r.std_latency_ms:.4f}",
                    f"{r.tflops:.2f}",
                ])
        
        return output_path
    
    def save_markdown(self, results: List[BenchmarkResult], filename: str = "rbk_report.md"):
        """Save results as Markdown table"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("# robust-kbench Benchmark Report\n\n")
            
            # Summary
            summary = self._generate_summary(results)
            f.write("## Summary\n\n")
            f.write(f"- **Total Shapes**: {summary['total_shapes']}\n")
            f.write(f"- **Kernels**: {', '.join(summary['kernels'])}\n")
            f.write(f"- **Mean Latency (p50)**: {summary['mean_p50_ms']:.3f} ms\n")
            f.write(f"- **Mean Throughput**: {summary['mean_tflops']:.2f} TFLOP/s\n\n")
            
            # Detailed results table
            f.write("## Detailed Results\n\n")
            f.write("| Kernel | Shape | p50 (ms) | p90 (ms) | p99 (ms) | TFLOP/s |\n")
            f.write("|--------|-------|----------|----------|----------|--------|\n")
            
            # Sort by p50 latency
            sorted_results = sorted(results, key=lambda r: r.p50_latency_ms)
            
            for r in sorted_results:
                f.write(f"| {r.kernel_name} | {r.shape} | "
                        f"{r.p50_latency_ms:.3f} | {r.p90_latency_ms:.3f} | "
                        f"{r.p99_latency_ms:.3f} | {r.tflops:.2f} |\n")
            
            f.write("\n")
            
            # Top 5 fastest
            f.write("## Top 5 Fastest (by p50)\n\n")
            for i, r in enumerate(sorted_results[:5], 1):
                f.write(f"{i}. **{r.kernel_name}** on {r.shape}: "
                        f"{r.p50_latency_ms:.3f} ms ({r.tflops:.2f} TFLOP/s)\n")
        
        return output_path
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not results:
            return {}
        
        kernels = list(set(r.kernel_name for r in results))
        mean_p50 = sum(r.p50_latency_ms for r in results) / len(results)
        mean_tflops = sum(r.tflops for r in results) / len(results)
        
        return {
            "total_shapes": len(results),
            "kernels": kernels,
            "mean_p50_ms": mean_p50,
            "mean_tflops": mean_tflops,
        }

