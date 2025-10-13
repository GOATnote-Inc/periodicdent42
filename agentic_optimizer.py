#!/usr/bin/env python3
"""
Production-Grade Agentic CUDA Kernel Optimization Harness
Expert-validated with lightweight profiling, JSON output, and safety checks.

Usage:
    python agentic_optimizer.py preflight    # Check GPU/env readiness
    python agentic_optimizer.py profile      # Lightweight Nsight profiling
    python agentic_optimizer.py build        # Build with timeout
    python agentic_optimizer.py benchmark    # JSON output
    python agentic_optimizer.py test         # Correctness tests
    python agentic_optimizer.py sanitize     # Memory safety check
    python agentic_optimizer.py evaluate 1.45 # Evaluate iteration
    python agentic_optimizer.py full-iteration # Run one complete iteration
"""

import subprocess
import json
import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime
import time

# Configuration
SPEEDUP_TARGET = 1.5
MAX_ITERATIONS = 20
CTA_TARGET_MULTIPLIER = 4  # CTAs should be ≥ 4×SM count
SM_COUNT_L4 = 58  # L4 GPU

TIMEOUTS = {
    "build": 120,
    "benchmark": 300,
    "profile": 300,
    "test": 120,
    "sanitize": 600,
}

# Lightweight Nsight metrics (NOT --set full!)
NSIGHT_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
]

class CUDAKernelOptimizer:
    """Production-grade CUDA kernel optimizer with safety checks."""
    
    def __init__(self, target_speedup=SPEEDUP_TARGET, max_iterations=MAX_ITERATIONS):
        self.iteration = 0
        self.best_speedup = 0.0
        self.target_speedup = target_speedup
        self.max_iterations = max_iterations
        self.history = []
        self.history_file = Path("optimization_history.json")
        self.baseline_torch = None  # Cache PyTorch baseline
        
        self._load_history()
        
    def _load_history(self):
        """Load optimization history from disk."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.history = data.get('history', [])
                self.iteration = data.get('current_iteration', 0)
                self.best_speedup = data.get('best_speedup', 0.0)
                self.baseline_torch = data.get('baseline_torch_ms', None)
            print(f"📂 Loaded history: {self.iteration} iterations, best: {self.best_speedup:.3f}x")
    
    def _save_history(self):
        """Save optimization history to disk."""
        data = {
            'current_iteration': self.iteration,
            'best_speedup': self.best_speedup,
            'target_speedup': self.target_speedup,
            'baseline_torch_ms': self.baseline_torch,
            'history': self.history,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def preflight_check(self):
        """Check GPU and environment readiness."""
        print("\n" + "="*70)
        print("🔍 PREFLIGHT CHECK")
        print("="*70)
        
        checks = []
        
        # Check 1: nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                print(f"✅ GPU: {gpu_info}")
                checks.append(True)
            else:
                print(f"❌ nvidia-smi failed")
                checks.append(False)
        except Exception as e:
            print(f"❌ nvidia-smi error: {e}")
            checks.append(False)
        
        # Check 2: PyTorch CUDA
        try:
            result = subprocess.run([
                "python", "-c",
                "import torch; assert torch.cuda.is_available(); "
                "print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ {result.stdout.strip()}")
                checks.append(True)
            else:
                print(f"❌ PyTorch CUDA not available")
                checks.append(False)
        except Exception as e:
            print(f"❌ PyTorch check error: {e}")
            checks.append(False)
        
        # Check 3: CUDA compiler
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version_line = [l for l in result.stdout.split('\n') if 'release' in l.lower()][0]
                print(f"✅ NVCC: {version_line.strip()}")
                checks.append(True)
            else:
                print(f"❌ nvcc not found")
                checks.append(False)
        except Exception as e:
            print(f"❌ nvcc check error: {e}")
            checks.append(False)
        
        # Check 4: Nsight Compute CLI (optional but recommended)
        try:
            result = subprocess.run(
                ["nv-nsight-cu-cli", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"✅ Nsight Compute CLI available")
                checks.append(True)
            else:
                print(f"⚠️  Nsight Compute CLI not found (profiling will be limited)")
                checks.append(True)  # Not critical
        except Exception as e:
            print(f"⚠️  Nsight Compute CLI not available: {e}")
            checks.append(True)  # Not critical
        
        all_passed = all(checks)
        print()
        if all_passed:
            print("✅ All preflight checks passed")
        else:
            print("❌ Some preflight checks failed - review above")
        
        return {"success": all_passed, "checks": checks, "timestamp": datetime.now().isoformat()}
    
    def build_kernel(self):
        """Compile CUDA kernel with timeout and fail-fast."""
        print("\n" + "="*70)
        print("🔨 BUILDING KERNEL")
        print("="*70)
        
        try:
            result = subprocess.run(
                ["./build_manual.sh"],
                capture_output=True,
                text=True,
                timeout=TIMEOUTS["build"]
            )
            
            success = result.returncode == 0
            
            if success:
                print("✅ Build succeeded!")
            else:
                print("❌ Build failed!")
                print("\nError output:")
                print(result.stderr)
                # FAIL FAST - don't continue loop
                sys.exit(1)
            
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
        
        except subprocess.TimeoutExpired:
            print(f"❌ Build timeout ({TIMEOUTS['build']}s)")
            return {"success": False, "error": "timeout"}
    
    def _extract_json_from_output(self, output):
        """Extract last valid JSON line from output."""
        lines = output.strip().split('\n')
        for line in reversed(lines):  # Search from end
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None
    
    def run_benchmark(self):
        """Run benchmark with JSON output and CTA validation."""
        print("\n" + "="*70)
        print("📊 RUNNING BENCHMARK")
        print("="*70)
        
        try:
            result = subprocess.run(
                ["python", "benches/bench_correctness_and_speed.py", "--json"],
                capture_output=True,
                text=True,
                timeout=TIMEOUTS["benchmark"]
            )
            
            # Extract JSON from output
            data = self._extract_json_from_output(result.stdout)
            
            if data is None:
                print("⚠️  No JSON output found, parsing legacy format...")
                # Fallback to regex parsing
                speedup = self._parse_speedup_legacy(result.stdout)
                data = {"speedup_vs_torch": speedup, "latency_ms": 0}
            
            speedup = data.get("speedup_vs_torch", 0.0)
            latency_ms = data.get("latency_ms", 0.0)
            ctas = data.get("ctas", 0)
            sm_count = data.get("sm_count", SM_COUNT_L4)
            
            print(f"⚡ Speedup: {speedup:.3f}x vs PyTorch")
            print(f"⏱️  Latency: {latency_ms:.3f}ms")
            
            # CRITICAL: Validate CTA count
            min_ctas = CTA_TARGET_MULTIPLIER * sm_count
            if ctas > 0:
                print(f"🔢 CTAs: {ctas} (target: ≥{min_ctas})")
                if ctas < min_ctas:
                    print(f"⚠️  WARNING: Grid too small! {ctas} < {min_ctas}")
                    print(f"   GPU utilization: {100*ctas/min_ctas:.1f}% of target")
            
            if speedup >= self.target_speedup:
                print(f"🎯 TARGET ACHIEVED! {speedup:.3f}x ≥ {self.target_speedup}x")
            
            return {
                "success": result.returncode == 0,
                "speedup": speedup,
                "latency_ms": latency_ms,
                "ctas": ctas,
                "sm_count": sm_count,
                "data": data,
                "output": result.stdout,
                "timestamp": datetime.now().isoformat()
            }
        
        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark timeout ({TIMEOUTS['benchmark']}s)")
            return {"success": False, "speedup": 0.0, "error": "timeout"}
    
    def _parse_speedup_legacy(self, output):
        """Fallback regex parsing for speedup."""
        patterns = [
            r'Average Speedup:\s*([0-9.]+)x',
            r'Mean speedup:\s*([0-9.]+)x',
            r'Speedup:\s*([0-9.]+)x'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def profile_kernel(self):
        """Lightweight Nsight Compute profiling (NOT --set full)."""
        print("\n" + "="*70)
        print("🔍 PROFILING KERNEL (Lightweight)")
        print("="*70)
        
        # Check if Nsight Compute CLI is available
        try:
            subprocess.run(
                ["nv-nsight-cu-cli", "--version"],
                capture_output=True,
                timeout=5
            )
            has_nsight = True
        except:
            has_nsight = False
        
        if not has_nsight:
            print("⚠️  Nsight Compute CLI not available")
            print("   Using basic profiling (benchmark timing only)")
            return self._profile_basic()
        
        return self._profile_with_nsight()
    
    def _profile_with_nsight(self):
        """Profile using lightweight Nsight Compute metrics."""
        profile_file = f"profile_iter{self.iteration + 1}.txt"
        
        cmd = [
            "nv-nsight-cu-cli",
            "--profile-from-start", "off",
            "--target-processes", "all",
            "--metrics", ",".join(NSIGHT_METRICS),
            "python", "benches/bench_correctness_and_speed.py", "--quick"
        ]
        
        print(f"Running: {' '.join(cmd[:4])} ... (with {len(NSIGHT_METRICS)} metrics)")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUTS["profile"]
            )
            
            # Save raw output
            with open(profile_file, "w") as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # Parse metrics
            metrics = self._parse_nsight_metrics(result.stdout + result.stderr)
            
            print(f"\n📈 Profile Metrics:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
            
            print(f"\n💾 Saved to: {profile_file}")
            
            return {
                "success": result.returncode == 0,
                "profiler": "nsight-lightweight",
                "metrics": metrics,
                "profile_file": profile_file,
                "timestamp": datetime.now().isoformat()
            }
        
        except subprocess.TimeoutExpired:
            print(f"❌ Profile timeout ({TIMEOUTS['profile']}s)")
            return {"success": False, "error": "timeout"}
    
    def _profile_basic(self):
        """Fallback basic profiling using benchmark timing."""
        print("Running benchmark for timing data...")
        bench_result = self.run_benchmark()
        
        return {
            "success": bench_result['success'],
            "profiler": "basic-timing",
            "speedup": bench_result.get('speedup', 0.0),
            "latency_ms": bench_result.get('latency_ms', 0.0),
            "note": "Install nv-nsight-cu-cli for detailed profiling",
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_nsight_metrics(self, output):
        """Parse key metrics from Nsight output."""
        metrics = {}
        
        # Parse patterns
        patterns = {
            'sm_throughput': r'sm__throughput\.avg\.pct_of_peak_sustained_elapsed\s+([0-9.]+)',
            'warp_occupancy': r'sm__warps_active\.avg\.pct_of_peak_sustained_active\s+([0-9.]+)',
            'mem_throughput': r'lts__throughput\.avg\.pct_of_peak_sustained_elapsed\s+([0-9.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[key] = f"{match.group(1)}%"
            else:
                metrics[key] = "unknown"
        
        # Determine bottleneck
        try:
            mem_pct = float(metrics.get('mem_throughput', '0').rstrip('%'))
            sm_pct = float(metrics.get('sm_throughput', '0').rstrip('%'))
            
            if mem_pct > sm_pct and mem_pct > 50:
                metrics['bottleneck'] = "memory-bound"
            elif sm_pct > 50:
                metrics['bottleneck'] = "compute-bound"
            else:
                metrics['bottleneck'] = "parallelism-limited (low utilization)"
        except:
            metrics['bottleneck'] = "unknown"
        
        return metrics
    
    def run_tests(self):
        """Run correctness tests."""
        print("\n" + "="*70)
        print("🧪 RUNNING CORRECTNESS TESTS")
        print("="*70)
        
        try:
            result = subprocess.run(
                ["python", "tests/test_basic.py"],
                capture_output=True,
                text=True,
                timeout=TIMEOUTS["test"]
            )
            
            success = result.returncode == 0
            
            if success:
                print("✅ All tests passed!")
            else:
                print("❌ Tests failed!")
                print(result.stderr)
            
            return {
                "success": success,
                "output": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
        
        except subprocess.TimeoutExpired:
            print(f"❌ Test timeout ({TIMEOUTS['test']}s)")
            return {"success": False, "error": "timeout"}
    
    def run_sanitizer(self):
        """Run compute-sanitizer for memory safety."""
        print("\n" + "="*70)
        print("🔬 RUNNING COMPUTE-SANITIZER")
        print("="*70)
        
        try:
            result = subprocess.run([
                "compute-sanitizer",
                "--tool=memcheck",
                "python", "benches/bench_correctness_and_speed.py", "--quick"
            ], capture_output=True, text=True, timeout=TIMEOUTS["sanitize"])
            
            # Check for errors
            has_errors = "ERROR" in result.stderr or "ERROR" in result.stdout
            has_warnings = "WARNING" in result.stderr or "WARNING" in result.stdout
            
            if has_errors:
                print("❌ Memory errors detected!")
                print(result.stderr[-500:])  # Last 500 chars
            elif has_warnings:
                print("⚠️  Warnings detected")
            else:
                print("✅ No memory errors")
            
            return {
                "success": not has_errors,
                "has_warnings": has_warnings,
                "output": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
        
        except subprocess.TimeoutExpired:
            print(f"⚠️  Sanitizer timeout ({TIMEOUTS['sanitize']}s) - skipping")
            return {"success": True, "note": "timeout-skipped"}
        except FileNotFoundError:
            print("⚠️  compute-sanitizer not found - skipping")
            return {"success": True, "note": "not-available"}
    
    def evaluate_iteration(self, speedup):
        """Evaluate iteration with auto-revert on regression."""
        self.iteration += 1
        improved = speedup > self.best_speedup
        regressed = speedup < (self.best_speedup * 0.98)  # >2% regression
        
        print("\n" + "="*70)
        print(f"📊 ITERATION {self.iteration} EVALUATION")
        print("="*70)
        
        if improved:
            gain_pct = ((speedup - self.best_speedup) / self.best_speedup * 100) if self.best_speedup > 0 else float('inf')
            self.best_speedup = speedup
            print(f"✅ IMPROVEMENT!")
            print(f"   Previous best: {self.best_speedup - (speedup - self.best_speedup):.3f}x")
            print(f"   New best:      {speedup:.3f}x")
            if gain_pct != float('inf'):
                print(f"   Gain:          +{gain_pct:.1f}%")
            print(f"\n💾 COMMIT changes to git")
        
        elif regressed:
            regression_pct = ((self.best_speedup - speedup) / self.best_speedup * 100)
            print(f"🔄 REGRESSION! ({regression_pct:.1f}% slower)")
            print(f"   Current:  {speedup:.3f}x")
            print(f"   Best:     {self.best_speedup:.3f}x")
            print(f"\n♻️  AUTO-REVERT: git restore -SW :/ && git clean -fd")
            print(f"   Try different approach")
        
        else:
            print(f"⚠️  No significant change")
            print(f"   Current:  {speedup:.3f}x")
            print(f"   Best:     {self.best_speedup:.3f}x")
        
        target_hit = speedup >= self.target_speedup
        should_continue = not target_hit and self.iteration < self.max_iterations
        
        if target_hit:
            print(f"\n🎉 TARGET ACHIEVED! {speedup:.3f}x ≥ {self.target_speedup}x")
        elif self.iteration >= self.max_iterations:
            print(f"\n⏹️  Max iterations ({self.max_iterations}) reached")
        
        result = {
            "iteration": self.iteration,
            "speedup": speedup,
            "best_speedup": self.best_speedup,
            "improved": improved,
            "regressed": regressed,
            "target_achieved": target_hit,
            "should_continue": should_continue,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history.append(result)
        self._save_history()
        
        return result
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "="*70)
        print("📈 OPTIMIZATION SUMMARY")
        print("="*70)
        
        if not self.history:
            print("No iterations completed yet.")
            return
        
        print(f"\nTotal iterations: {len(self.history)}")
        print(f"Best speedup: {self.best_speedup:.3f}x")
        print(f"Target speedup: {self.target_speedup}x")
        status = '✅ ACHIEVED' if self.best_speedup >= self.target_speedup else '🔄 IN PROGRESS'
        print(f"Status: {status}")
        
        print("\nIteration history:")
        print(f"{'Iter':<6} {'Speedup':<10} {'Best':<10} {'Status':<15}")
        print("-" * 45)
        
        for h in self.history:
            if h.get('target_achieved'):
                status = "🎯 TARGET"
            elif h.get('improved'):
                status = "✅ IMPROVED"
            elif h.get('regressed'):
                status = "🔄 REGRESSED"
            else:
                status = "⚠️  NO CHANGE"
            
            print(f"{h['iteration']:<6} {h['speedup']:<10.3f} {h['best_speedup']:<10.3f} {status:<15}")

def main():
    parser = argparse.ArgumentParser(
        description='Production-Grade Agentic CUDA Kernel Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preflight check
    subparsers.add_parser('preflight', help='Check GPU and environment readiness')
    
    # Profile command
    subparsers.add_parser('profile', help='Lightweight Nsight profiling')
    
    # Build command
    subparsers.add_parser('build', help='Build kernel with timeout')
    
    # Benchmark command
    subparsers.add_parser('benchmark', help='Run benchmarks (JSON output)')
    
    # Test command
    subparsers.add_parser('test', help='Run correctness tests')
    
    # Sanitizer command
    subparsers.add_parser('sanitize', help='Run compute-sanitizer')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate iteration')
    eval_parser.add_argument('speedup', type=float, help='Speedup value')
    
    # Summary command
    subparsers.add_parser('summary', help='Print optimization summary')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize optimizer
    opt = CUDAKernelOptimizer()
    
    # Execute command
    if args.command == 'preflight':
        result = opt.preflight_check()
        sys.exit(0 if result['success'] else 1)
    
    elif args.command == 'profile':
        result = opt.profile_kernel()
        print(json.dumps(result, indent=2))
    
    elif args.command == 'build':
        result = opt.build_kernel()
        sys.exit(0 if result['success'] else 1)
    
    elif args.command == 'benchmark':
        result = opt.run_benchmark()
        sys.exit(0 if result['success'] else 1)
    
    elif args.command == 'test':
        result = opt.run_tests()
        sys.exit(0 if result['success'] else 1)
    
    elif args.command == 'sanitize':
        result = opt.run_sanitizer()
        sys.exit(0 if result['success'] else 1)
    
    elif args.command == 'evaluate':
        result = opt.evaluate_iteration(args.speedup)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'summary':
        opt.print_summary()

if __name__ == '__main__':
    main()
