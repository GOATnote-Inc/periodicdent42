"""
EvoEngineer Evaluator: Benchmark execution and correctness validation
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import subprocess
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    p50_latency_ms: float
    p90_latency_ms: float
    tflops: float
    passed: bool
    error: Optional[str] = None


class CorrectnessGate:
    """Validates kernel correctness against SDPA"""
    
    def __init__(self, test_script: Path):
        self.test_script = Path(test_script)
    
    def validate(self, candidate_hash: str) -> bool:
        """Run correctness tests for a candidate"""
        try:
            result = subprocess.run(
                ["python3", str(self.test_script), "--candidate", candidate_hash],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


class BenchmarkEvaluator:
    """Runs performance benchmarks for kernel candidates"""
    
    def __init__(self, bench_script: Path, output_dir: Path):
        self.bench_script = Path(bench_script)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        candidate_hash: str,
        shapes: list,
        warmups: int = 20,
        iters: int = 100,
    ) -> Optional[BenchmarkResult]:
        """Run benchmark for a candidate on specified shapes"""
        
        output_file = self.output_dir / f"bench_{candidate_hash}.json"
        
        try:
            # Run benchmark script
            result = subprocess.run(
                [
                    "python3",
                    str(self.bench_script),
                    "--candidate", candidate_hash,
                    "--shapes", json.dumps(shapes),
                    "--warmups", str(warmups),
                    "--iters", str(iters),
                    "--output", str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            if result.returncode != 0:
                return BenchmarkResult(
                    p50_latency_ms=float('inf'),
                    p90_latency_ms=float('inf'),
                    tflops=0.0,
                    passed=False,
                    error=result.stderr,
                )
            
            # Parse results
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                
                return BenchmarkResult(
                    p50_latency_ms=data["p50_latency_ms"],
                    p90_latency_ms=data["p90_latency_ms"],
                    tflops=data["tflops"],
                    passed=True,
                )
            else:
                return BenchmarkResult(
                    p50_latency_ms=float('inf'),
                    p90_latency_ms=float('inf'),
                    tflops=0.0,
                    passed=False,
                    error="No output file generated",
                )
                
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                p50_latency_ms=float('inf'),
                p90_latency_ms=float('inf'),
                tflops=0.0,
                passed=False,
                error="Benchmark timeout",
            )
        except Exception as e:
            return BenchmarkResult(
                p50_latency_ms=float('inf'),
                p90_latency_ms=float('inf'),
                tflops=0.0,
                passed=False,
                error=str(e),
            )

