#!/usr/bin/env python3
"""
FlashCore v12: EvoTuner - Fitness-Driven Kernel Optimization
Implements LLM-guided search with correctness-as-fitness scoring
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

ROOT = Path(__file__).parent.parent

# Configuration space
CONFIG_SPACE = {
    'BLOCK_M': [32, 48, 64],
    'BLOCK_N': [32, 48, 64],
    'STAGES': [2, 3],
    'COMPUTE_WARPS': [10, 11, 12],
    'LOAD_WARPS': [3, 4, 5],
    'SOFTMAX_WARPS': [1, 2],
}

class Variant:
    """Kernel variant configuration"""
    def __init__(self, **params):
        self.params = params
        self.id = self._compute_id()
        self.metrics = {}
    
    def _compute_id(self):
        """Generate unique ID from parameters"""
        param_str = '_'.join(f"{k}{v}" for k, v in sorted(self.params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    def score(self) -> float:
        """Fast scoring function (for filtering)"""
        if not self.metrics:
            return -1e9
        
        regs = self.metrics.get('regs', 128)
        occupancy = self.metrics.get('occupancy', 0.0)
        passed = self.metrics.get('passed', False)
        
        score = 1000.0  # Base score
        score -= 5.0 * max(0, regs - 64)  # Penalize high register usage
        score -= 100.0 * max(0, 0.6 - occupancy)  # Penalize low occupancy
        score -= 1e6 if not passed else 0  # Hard kill if failed
        
        return score
    
    def fitness(self) -> float:
        """Robust fitness function (for final selection)"""
        if not self.metrics:
            return -1e9
        
        latency_us = self.metrics.get('latency_us', 1e6)
        max_err = self.metrics.get('max_error', 1.0)
        passed = self.metrics.get('passed', False)
        has_nan = self.metrics.get('has_nan', False)
        deterministic = self.metrics.get('deterministic', False)
        
        # Reference: SDPA @ ~30 µs
        ref_us = 30.0
        
        # Fitness components
        speedup = ref_us / latency_us if latency_us > 0 else 0.0
        error_penalty = min(1.0, max_err / 1e-4)
        
        fitness = 1.0 * speedup  # Reward speedup
        fitness -= 0.4 * error_penalty  # Penalize errors
        fitness -= 1.0 if has_nan else 0.0  # Hard penalty for NaN
        fitness -= 1.0 if not passed else 0.0  # Hard penalty for failure
        fitness -= 0.5 if not deterministic else 0.0  # Penalty for non-determinism
        
        return fitness

def generate_initial_population(n=24) -> List[Variant]:
    """Generate initial population with diverse configurations"""
    import itertools
    import random
    
    variants = []
    
    # Grid corners (guaranteed coverage)
    for block_m in [CONFIG_SPACE['BLOCK_M'][0], CONFIG_SPACE['BLOCK_M'][-1]]:
        for block_n in [CONFIG_SPACE['BLOCK_N'][0], CONFIG_SPACE['BLOCK_N'][-1]]:
            for stages in CONFIG_SPACE['STAGES']:
                variants.append(Variant(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    STAGES=stages,
                    COMPUTE_WARPS=11,
                    LOAD_WARPS=4,
                    SOFTMAX_WARPS=1,
                ))
    
    # Random sampling (exploration)
    while len(variants) < n:
        params = {k: random.choice(v) for k, v in CONFIG_SPACE.items()}
        # Ensure total warps = 16
        total_warps = params['COMPUTE_WARPS'] + params['LOAD_WARPS'] + params['SOFTMAX_WARPS']
        if total_warps == 16:
            variants.append(Variant(**params))
    
    return variants[:n]

def build_and_test_variant(variant: Variant) -> bool:
    """Build and test a variant, populate metrics"""
    print(f"\n{'='*60}")
    print(f"Testing variant: {variant.id}")
    print(f"  Params: {variant.params}")
    print(f"{'='*60}")
    
    # TODO: Write params to kernel config header
    # For now, just run baseline
    
    # Build
    try:
        result = subprocess.run(
            ['bash', 'tools/bench.sh', variant.id],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        # Parse results
        bench_json = ROOT / f'results/bench_{variant.id}.json'
        if bench_json.exists():
            data = json.loads(bench_json.read_text())
            variant.metrics = {
                'latency_us': data.get('latency_us', 1e6),
                'max_error': data.get('max_error', 1.0),
                'regs': data['ptxas'].get('registers', 128),
                'smem_bytes': data['ptxas'].get('smem_bytes', 0),
                'passed': result.returncode == 0,
                'occupancy': 0.5,  # TODO: Extract from NCU
                'has_nan': False,  # TODO: Check in logs
                'deterministic': True,  # TODO: Run 3x and check hash
            }
            return True
        else:
            variant.metrics = {'passed': False, 'latency_us': 1e6}
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Variant {variant.id} timed out")
        variant.metrics = {'passed': False, 'latency_us': 1e6}
        return False
    except Exception as e:
        print(f"❌ Variant {variant.id} failed: {e}")
        variant.metrics = {'passed': False, 'latency_us': 1e6}
        return False

def run_iteration(population: List[Variant]) -> List[Variant]:
    """Run one iteration of EvoTuner"""
    print(f"\n{'='*60}")
    print(f"EvoTuner Iteration: {len(population)} variants")
    print(f"{'='*60}")
    
    # Test all variants
    for variant in population:
        if not variant.metrics:  # Skip if already tested
            build_and_test_variant(variant)
    
    # Score and sort
    population.sort(key=lambda v: v.fitness(), reverse=True)
    
    # Report top 5
    print(f"\n{'='*60}")
    print(f"Top 5 Variants (by fitness):")
    print(f"{'='*60}")
    for i, variant in enumerate(population[:5]):
        print(f"{i+1}. {variant.id}: fitness={variant.fitness():.3f}, "
              f"latency={variant.metrics.get('latency_us', 0):.2f} µs, "
              f"error={variant.metrics.get('max_error', 0):.6f}")
    
    return population

def save_results(population: List[Variant], output_file: Path):
    """Save all results to JSON"""
    results = {
        'variants': [
            {
                'id': v.id,
                'params': v.params,
                'metrics': v.metrics,
                'score': v.score(),
                'fitness': v.fitness(),
            }
            for v in population
        ],
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Results saved to {output_file}")

def main():
    print("════════════════════════════════════════════════════════════════")
    print("FlashCore v12: EvoTuner - Fitness-Driven Optimization")
    print("════════════════════════════════════════════════════════════════")
    
    # Generate initial population
    population = generate_initial_population(n=24)
    print(f"Generated {len(population)} initial variants")
    
    # Run iteration
    population = run_iteration(population)
    
    # Save results
    output_file = ROOT / 'results/evo_tuner_results.json'
    save_results(population, output_file)
    
    # Check for excellence
    best = population[0]
    if best.metrics.get('latency_us', 1e6) <= 28.0:
        print("\n🎉 EXCELLENCE ACHIEVED: ≤28 µs!")
        return 0
    else:
        print(f"\n⚠️  Best: {best.metrics.get('latency_us', 0):.2f} µs (target: ≤28 µs)")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

