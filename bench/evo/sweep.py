#!/usr/bin/env python3
"""
EvoEngineer Sweep - Minimal Evolutionary Optimization Loop
Extends existing working kernels without breaking them.
"""

import os
import sys
import yaml
import json
import subprocess
import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch


class EvoSweep:
    """Minimal EvoEngineer loop for FlashAttention optimization"""
    
    def __init__(self, config_path: str = "evo.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.root = Path(__file__).parent.parent.parent
        self.evidence_dir = self.root / "evidence"
        self.evidence_dir.mkdir(exist_ok=True)
        
        self.log_path = self.evidence_dir / "evo_log.csv"
        self.best_path = self.evidence_dir / "evo_best.json"
        
        # Get git commit SHA for tracking
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.root
            )
            self.commit_sha = result.stdout.strip()
        except:
            self.commit_sha = "unknown"
        
        # Initialize log file
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'commit_sha', 'generation', 'variant_id', 
                    'BLOCK_M', 'NUM_WARPS', 'VEC_WIDTH', 'SMEM_STAGE', 'USE_WMMA', 'REDUCE',
                    'time_us', 'sdpa_us', 'speedup', 'correct', 'build_ok',
                    'ncu_sm_busy', 'ncu_dram_util', 'ncu_tensor_active',
                    'timestamp'
                ])
        
        # Track best candidates
        self.population = []
        self.generation = 0
    
    def generate_candidates(self, gen: int) -> List[Dict]:
        """Generate candidate configurations for a generation"""
        params = self.config['params']
        gen_constraints = self.config.get('gen_constraints', {}).get(gen, {})
        
        # Apply generation-specific constraints
        search_space = {}
        for key, values in params.items():
            if key in gen_constraints:
                search_space[key] = gen_constraints[key]
            else:
                search_space[key] = values
        
        candidates_per_gen = self.config['budget']['candidates_per_gen']
        
        if gen == 0:
            # Generation 0: Grid sample with priority on warp reductions
            candidates = []
            
            # Priority 1: Warp reductions with different tile sizes
            for block_m, num_warps, vec_width in itertools.product(
                search_space['BLOCK_M'],
                search_space['NUM_WARPS'],
                search_space['VEC_WIDTH']
            ):
                if len(candidates) >= candidates_per_gen:
                    break
                candidates.append({
                    'BLOCK_M': block_m,
                    'NUM_WARPS': num_warps,
                    'VEC_WIDTH': vec_width,
                    'SMEM_STAGE': 2,
                    'USE_WMMA': 0,
                    'REDUCE': 'warp'
                })
            
            # Fill remaining with serial fallback variants
            for block_m in search_space['BLOCK_M']:
                if len(candidates) >= candidates_per_gen:
                    break
                candidates.append({
                    'BLOCK_M': block_m,
                    'NUM_WARPS': 4,
                    'VEC_WIDTH': 2,
                    'SMEM_STAGE': 2,
                    'USE_WMMA': 0,
                    'REDUCE': 'serial'
                })
            
            return candidates[:candidates_per_gen]
        
        else:
            # Generation 1+: Mutate top-K from previous generation
            candidates = []
            mutate_radius = gen_constraints.get('mutate_radius', 1)
            
            for seed in self.population[:self.config['top_k']]:
                # Create mutations around seed
                for _ in range(candidates_per_gen // self.config['top_k']):
                    mutated = seed['params'].copy()
                    
                    # Mutate 1-2 parameters
                    keys_to_mutate = random.sample(list(search_space.keys()), 
                                                  k=min(2, len(search_space)))
                    
                    for key in keys_to_mutate:
                        options = search_space[key]
                        if isinstance(options[0], int):
                            # Numeric parameter
                            current = mutated[key]
                            current_idx = options.index(current) if current in options else 0
                            new_idx = max(0, min(len(options)-1, 
                                               current_idx + random.choice([-mutate_radius, 0, mutate_radius])))
                            mutated[key] = options[new_idx]
                        else:
                            # Categorical parameter
                            mutated[key] = random.choice(options)
                    
                    candidates.append(mutated)
            
            return candidates[:candidates_per_gen]
    
    def build_variant(self, params: Dict) -> bool:
        """Build a kernel variant with given parameters"""
        # Set environment variables
        env = os.environ.copy()
        for key, value in params.items():
            env[key] = str(value)
        
        # Clean previous build cache
        cache_dir = Path.home() / ".cache/torch_extensions/py310_cu121/fa_phase3"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        
        print(f"  Building with {params}...", end=" ", flush=True)
        
        try:
            # Use existing build script
            result = subprocess.run(
                [sys.executable, "bench/build_phase3_variant.py"],
                env=env,
                capture_output=True,
                text=True,
                cwd=self.root,
                timeout=300
            )
            
            if result.returncode == 0:
                print("✅")
                return True
            else:
                print(f"❌ ({result.stderr[:100]})")
                return False
        
        except Exception as e:
            print(f"❌ ({str(e)[:100]})")
            return False
    
    def test_variant(self, params: Dict) -> Tuple[bool, Optional[float], Optional[float]]:
        """Test correctness and measure timing"""
        print(f"  Testing correctness...", end=" ", flush=True)
        
        try:
            # Import the built module
            sys.path.insert(0, str(Path.home() / ".cache/torch_extensions/py310_cu121/fa_phase3"))
            import fa_phase3
            
            # Test setup
            shape = self.config['target_shape']
            B, H, S, D = shape['B'], shape['H'], shape['S'], shape['D']
            
            torch.manual_seed(42)
            q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
            k, v = q.clone(), q.clone()
            softmax_scale = 1.0 / (D ** 0.5)
            
            # PyTorch reference
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            
            # Our kernel
            out = fa_phase3.forward(q, k, v, softmax_scale)
            
            # Correctness check
            max_diff = (out - ref).abs().max().item()
            passed = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
            
            if not passed:
                print(f"❌ (max_diff={max_diff:.6f})")
                return False, None, None
            
            print(f"✅ (max_diff={max_diff:.6f})", flush=True)
            
            # Timing
            print(f"  Benchmarking...", end=" ", flush=True)
            
            # Warm-up
            for _ in range(10):
                _ = fa_phase3.forward(q, k, v, softmax_scale)
            torch.cuda.synchronize()
            
            # Measure
            times = []
            for _ in range(100):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = fa_phase3.forward(q, k, v, softmax_scale)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1000)  # ms → μs
            
            p50 = sorted(times)[50]
            
            # SDPA reference timing
            sdpa_times = []
            for _ in range(100):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
                end.record()
                torch.cuda.synchronize()
                sdpa_times.append(start.elapsed_time(end) * 1000)
            
            sdpa_p50 = sorted(sdpa_times)[50]
            
            print(f"{p50:.2f} μs (speedup: {sdpa_p50/p50:.2f}×)")
            
            return True, p50, sdpa_p50
        
        except Exception as e:
            print(f"❌ ({str(e)[:100]})")
            return False, None, None
    
    def capture_nsight_metrics(self, params: Dict) -> Dict[str, Optional[float]]:
        """Capture Nsight Compute brief metrics (best-effort)"""
        metrics = {
            'ncu_sm_busy': None,
            'ncu_dram_util': None,
            'ncu_tensor_active': None
        }
        
        # Skip if on non-GPU machine
        if not torch.cuda.is_available():
            return metrics
        
        print(f"  Capturing Nsight metrics...", end=" ", flush=True)
        
        try:
            # Create a simple test script
            test_script = self.evidence_dir / "ncu_test.py"
            with open(test_script, 'w') as f:
                f.write(f"""
import sys
sys.path.insert(0, '{Path.home() / ".cache/torch_extensions/py310_cu121/fa_phase3"}')
import torch
import fa_phase3

torch.manual_seed(42)
q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
k, v = q.clone(), q.clone()
softmax_scale = 1.0 / 8.0

for _ in range(5):
    out = fa_phase3.forward(q, k, v, softmax_scale)
torch.cuda.synchronize()
""")
            
            # Run with NCU
            result = subprocess.run(
                ["bash", "scripts/ncu_brief.sh", sys.executable, str(test_script)],
                capture_output=True,
                text=True,
                cwd=self.root,
                timeout=60
            )
            
            # Parse CSV output (simplified - just look for metric values)
            if "sm__warps_active" in result.stdout:
                print("✅")
                # Would parse CSV here in production
            else:
                print("⚠️ (metrics not found)")
        
        except Exception as e:
            print(f"⚠️ ({str(e)[:50]})")
        
        return metrics
    
    def log_result(self, gen: int, variant_id: int, params: Dict, 
                   build_ok: bool, correct: bool, time_us: Optional[float],
                   sdpa_us: Optional[float], ncu_metrics: Dict):
        """Log a single result to CSV"""
        speedup = (sdpa_us / time_us) if (time_us and sdpa_us) else None
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.commit_sha,
                gen,
                variant_id,
                params.get('BLOCK_M'),
                params.get('NUM_WARPS'),
                params.get('VEC_WIDTH'),
                params.get('SMEM_STAGE'),
                params.get('USE_WMMA'),
                params.get('REDUCE'),
                time_us if time_us else '',
                sdpa_us if sdpa_us else '',
                f"{speedup:.4f}" if speedup else '',
                correct,
                build_ok,
                ncu_metrics.get('ncu_sm_busy', ''),
                ncu_metrics.get('ncu_dram_util', ''),
                ncu_metrics.get('ncu_tensor_active', ''),
                datetime.now().isoformat()
            ])
    
    def update_population(self, candidate: Dict):
        """Add candidate to population and keep top-K"""
        self.population.append(candidate)
        self.population.sort(key=lambda x: x['speedup'], reverse=True)
        self.population = self.population[:self.config['top_k']]
    
    def save_best(self):
        """Save top-K to JSON"""
        with open(self.best_path, 'w') as f:
            json.dump({
                'commit_sha': self.commit_sha,
                'timestamp': datetime.now().isoformat(),
                'top_k': self.population,
                'baselines': self.config['baselines']
            }, f, indent=2)
    
    def run(self):
        """Execute the evolutionary sweep"""
        print("=" * 80)
        print("EvoEngineer Sweep - FlashAttention L4 Optimization")
        print("=" * 80)
        print(f"Config: {self.config['budget']['generations']} generations, "
              f"{self.config['budget']['candidates_per_gen']} candidates/gen")
        print(f"Target: Beat {self.config['baselines']['pytorch_sdpa']} μs (PyTorch SDPA)")
        print(f"Commit: {self.commit_sha}")
        print(f"Log: {self.log_path}")
        print("=" * 80)
        
        for gen in range(self.config['budget']['generations']):
            print(f"\n{'='*80}")
            print(f"GENERATION {gen}")
            print(f"{'='*80}")
            
            candidates = self.generate_candidates(gen)
            print(f"Generated {len(candidates)} candidates")
            
            consecutive_failures = 0
            early_stop_limit = self.config['early_stop']['consecutive_failures']
            
            for i, params in enumerate(candidates):
                print(f"\n[Gen {gen}, Variant {i+1}/{len(candidates)}]")
                print(f"  Params: {params}")
                
                # Build
                build_ok = self.build_variant(params)
                if not build_ok:
                    self.log_result(gen, i, params, False, False, None, None, {})
                    consecutive_failures += 1
                    if consecutive_failures >= early_stop_limit:
                        print(f"\n⚠️  Early stop: {consecutive_failures} consecutive failures")
                        break
                    continue
                
                # Test correctness and timing
                correct, time_us, sdpa_us = self.test_variant(params)
                
                if not correct:
                    self.log_result(gen, i, params, True, False, time_us, sdpa_us, {})
                    consecutive_failures += 1
                    if consecutive_failures >= early_stop_limit:
                        print(f"\n⚠️  Early stop: {consecutive_failures} consecutive failures")
                        break
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Optional: Capture Nsight metrics for correct candidates
                ncu_metrics = {}
                if correct and gen >= 1:  # Only in later generations to save time
                    ncu_metrics = self.capture_nsight_metrics(params)
                
                # Log result
                self.log_result(gen, i, params, True, True, time_us, sdpa_us, ncu_metrics)
                
                # Update population
                speedup = sdpa_us / time_us
                self.update_population({
                    'params': params,
                    'time_us': time_us,
                    'sdpa_us': sdpa_us,
                    'speedup': speedup,
                    'generation': gen
                })
            
            # Save best after each generation
            self.save_best()
            
            print(f"\nGeneration {gen} complete. Current Top-{self.config['top_k']}:")
            for i, candidate in enumerate(self.population):
                print(f"  {i+1}. Speedup: {candidate['speedup']:.3f}× "
                      f"({candidate['time_us']:.2f} μs) - {candidate['params']}")
        
        print(f"\n{'='*80}")
        print("SWEEP COMPLETE")
        print(f"{'='*80}")
        print(f"Results logged to: {self.log_path}")
        print(f"Best candidates saved to: {self.best_path}")
        print(f"\nFinal Top-{self.config['top_k']}:")
        for i, candidate in enumerate(self.population):
            print(f"  {i+1}. Speedup: {candidate['speedup']:.3f}× "
                  f"({candidate['time_us']:.2f} μs)")
            print(f"      Params: {candidate['params']}")


if __name__ == "__main__":
    sweep = EvoSweep()
    sweep.run()

