#!/usr/bin/env python3
"""
Loop 1: CUDA Kernel Iteration with Optuna

Systematic search for optimal FA-S512 kernel configuration using:
- Latin Hypercube Sampling for initial exploration
- Optuna TPE + MedianPruner for exploitation
- Hard gates from CUDA doctrine
- Nsight-driven promotion (optional)

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-13
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cudadent42.bench.candidate_kernel import candidate_kernel
from cudadent42.bench.search_space import SEARCH_SPACE, hard_gates
from cudadent42.bench.common.stats import bootstrap_ci


class Loop1Optimizer:
    """
    Loop 1: Iterative CUDA kernel optimization
    
    Searches for optimal kernel configuration through systematic iteration:
    1. Baseline (PyTorch SDPA)
    2. LHS seed (broad exploration)
    3. Optuna TPE (focused exploitation)
    4. Confirmation (bootstrap CIs)
    """
    
    def __init__(
        self,
        baseline_ms: float,
        target_speedup: float = 1.10,
        budget_minutes: int = 120,
        output_dir: str = "cudadent42/bench/artifacts/loop1"
    ):
        """
        Args:
            baseline_ms: Baseline latency (PyTorch SDPA) in milliseconds
            target_speedup: Target speedup over baseline (1.10 = 10% faster)
            budget_minutes: Time budget in minutes
            output_dir: Output directory for results
        """
        self.baseline_ms = baseline_ms
        self.target_speedup = target_speedup
        self.target_ms = baseline_ms / target_speedup
        self.budget_minutes = budget_minutes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.candidates = []  # Configs that passed gates
        self.rejected = []    # Configs that failed gates
        
        print(f"{'='*70}")
        print("LOOP 1: CUDA KERNEL ITERATION")
        print(f"{'='*70}")
        print(f"Baseline:       {baseline_ms:.4f} ms")
        print(f"Target:         {self.target_ms:.4f} ms ({target_speedup:.2f}× speedup)")
        print(f"Budget:         {budget_minutes} minutes")
        print(f"Output:         {self.output_dir}")
        print(f"{'='*70}\n")
    
    def _time_remaining(self) -> float:
        """Return minutes remaining in budget"""
        if self.start_time is None:
            return self.budget_minutes
        elapsed = (time.time() - self.start_time) / 60.0
        return max(0, self.budget_minutes - elapsed)
    
    def _lhs_sample(self, n_samples: int = 20) -> List[Dict[str, int]]:
        """
        Latin Hypercube Sampling for initial exploration
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            List of configurations
        """
        print(f"\n{'='*70}")
        print(f"PHASE 1: LHS SAMPLING ({n_samples} configs)")
        print(f"{'='*70}\n")
        
        # Simple LHS: divide each dimension into n_samples bins
        configs = []
        
        for i in range(n_samples):
            config = {}
            for key, values in SEARCH_SPACE.items():
                # Randomly select from available values
                # (True LHS would ensure Latin Hypercube property)
                config[key] = np.random.choice(values)
            configs.append(config)
        
        return configs
    
    def _evaluate_config(
        self,
        config: Dict[str, int],
        iterations: int = 40,
        phase: str = "search"
    ) -> Dict:
        """
        Evaluate a single configuration
        
        Args:
            config: Kernel configuration
            iterations: Number of timing iterations
            phase: "seed", "search", or "confirm"
        
        Returns:
            Result dict from candidate_kernel()
        """
        print(f"[{phase.upper()}] Testing config:")
        for k, v in config.items():
            print(f"  {k:12s}: {v}")
        
        result = candidate_kernel(config, iterations=iterations, warmup=10)
        
        median_ms = result['median_ms']
        passes = result['meta']['passes_gates']
        gate_result = result['meta']['gate_result']
        
        print(f"  → Latency: {median_ms:.4f} ms")
        print(f"  → Gates:   {'PASS' if passes else f'FAIL ({gate_result})'}")
        
        if passes:
            speedup = self.baseline_ms / median_ms
            improvement_pct = (self.baseline_ms - median_ms) / self.baseline_ms * 100
            print(f"  → Speedup: {speedup:.3f}× ({improvement_pct:+.1f}%)")
            
            if median_ms < self.target_ms:
                print(f"  ✅ BEATS TARGET ({self.target_ms:.4f} ms)")
                self.candidates.append(result)
            else:
                deficit_pct = (median_ms - self.target_ms) / self.target_ms * 100
                print(f"  ⚠️  Below target by {deficit_pct:.1f}%")
        else:
            self.rejected.append(result)
        
        print()
        return result
    
    def _optuna_objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function
        
        Args:
            trial: Optuna trial
        
        Returns:
            Latency in milliseconds (lower is better)
        """
        # Check time budget
        if self._time_remaining() <= 0:
            raise optuna.exceptions.OptunaError("Time budget exceeded")
        
        # Sample configuration
        config = {}
        for key, values in SEARCH_SPACE.items():
            config[key] = trial.suggest_categorical(key, values)
        
        # Evaluate
        result = self._evaluate_config(config, iterations=40, phase="optuna")
        
        # Report for pruning
        trial.report(result['median_ms'], step=0)
        
        # Prune if obviously bad
        if not result['meta']['passes_gates']:
            raise optuna.exceptions.TrialPruned(f"Failed gates: {result['meta']['gate_result']}")
        
        return result['median_ms']
    
    def run(self) -> Dict:
        """
        Run complete optimization loop
        
        Returns:
            Dict with best config, results, and summary
        """
        self.start_time = time.time()
        
        # === PHASE 1: LHS Seed ===
        lhs_configs = self._lhs_sample(n_samples=20)
        
        for i, config in enumerate(lhs_configs):
            if self._time_remaining() <= 0:
                print("⏱️  Time budget exhausted in LHS phase")
                break
            
            print(f"LHS {i+1}/{len(lhs_configs)}")
            self._evaluate_config(config, iterations=40, phase="lhs")
        
        # === PHASE 2: Optuna TPE ===
        print(f"\n{'='*70}")
        print(f"PHASE 2: OPTUNA TPE SEARCH")
        print(f"{'='*70}\n")
        
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        )
        
        # Run optimization
        try:
            study.optimize(
                self._optuna_objective,
                n_trials=100,
                timeout=self._time_remaining() * 60,
                show_progress_bar=True
            )
        except optuna.exceptions.OptunaError as e:
            print(f"\nOptuna stopped: {e}")
        
        # === PHASE 3: Confirmation ===
        if len(self.candidates) == 0:
            print(f"\n{'='*70}")
            print("❌ NO CANDIDATES FOUND")
            print(f"{'='*70}\n")
            print(f"Rejected: {len(self.rejected)}")
            print("\nTop rejection reasons:")
            reasons = {}
            for r in self.rejected:
                reason = r['meta']['gate_result']
                reasons[reason] = reasons.get(reason, 0) + 1
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
                print(f"  {reason}: {count}")
            
            return {
                'success': False,
                'best_config': None,
                'best_latency_ms': float('inf'),
                'speedup': 0.0,
                'candidates_found': 0,
                'rejected_count': len(self.rejected)
            }
        
        # Find best candidate
        best_result = min(self.candidates, key=lambda x: x['median_ms'])
        best_config = best_result['meta']['config']
        best_median = best_result['median_ms']
        
        print(f"\n{'='*70}")
        print(f"PHASE 3: CONFIRMATION (N=100, Bootstrap CI)")
        print(f"{'='*70}\n")
        print(f"Best candidate from search:")
        print(f"  Median: {best_median:.4f} ms")
        print(f"  Config: {best_config}")
        print(f"\nRe-running with N=100 for statistical confidence...")
        
        # Confirm with more iterations
        confirm_result = self._evaluate_config(best_config, iterations=100, phase="confirm")
        confirm_latencies = np.array(confirm_result['latencies'])
        confirm_median = np.median(confirm_latencies)
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_ci(
            confirm_latencies,
            statistic=np.median,
            confidence=0.95,
            n_bootstrap=10000,
            seed=42
        )
        
        # Baseline CI (approximation - would need actual baseline latencies)
        baseline_ci_lower = self.baseline_ms * 0.995
        baseline_ci_upper = self.baseline_ms * 1.005
        
        # Check CI overlap
        cis_overlap = not (ci_upper < baseline_ci_lower or ci_lower > baseline_ci_upper)
        
        speedup = self.baseline_ms / confirm_median
        improvement_pct = (self.baseline_ms - confirm_median) / self.baseline_ms * 100
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Baseline:  {self.baseline_ms:.4f} ms")
        print(f"Best:      {confirm_median:.4f} ms (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        print(f"Speedup:   {speedup:.3f}× ({improvement_pct:+.1f}%)")
        print(f"Target:    {self.target_speedup:.2f}×")
        print(f"CIs Overlap: {cis_overlap}")
        print(f"Significant: {not cis_overlap}")
        
        # Success criteria
        success = speedup >= self.target_speedup and not cis_overlap
        
        if success:
            print(f"\n🎉 SUCCESS: Achieved target speedup!")
        else:
            print(f"\n⚠️  Did not achieve target, but found improvement")
        
        print(f"\nCandidates evaluated: {len(self.candidates)}")
        print(f"Configs rejected:     {len(self.rejected)}")
        print(f"Time elapsed:         {(time.time() - self.start_time) / 60:.1f} min")
        print(f"{'='*70}\n")
        
        # Save results
        result_summary = {
            'success': success,
            'best_config': best_config,
            'best_latency_ms': confirm_median,
            'ci_95': [ci_lower, ci_upper],
            'speedup': speedup,
            'improvement_pct': improvement_pct,
            'target_speedup': self.target_speedup,
            'cis_overlap': cis_overlap,
            'candidates_found': len(self.candidates),
            'rejected_count': len(self.rejected),
            'time_minutes': (time.time() - self.start_time) / 60.0,
            'baseline_ms': self.baseline_ms,
        }
        
        output_file = self.output_dir / "loop1_results.json"
        with open(output_file, 'w') as f:
            json.dump(result_summary, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        
        return result_summary


def main():
    """
    Run Loop 1 optimization
    
    Assumes baseline has already been established (PyTorch SDPA)
    """
    # Baseline from previous measurements (with TF32 fix)
    BASELINE_MS = 0.3226
    
    optimizer = Loop1Optimizer(
        baseline_ms=BASELINE_MS,
        target_speedup=1.10,  # 10% improvement goal
        budget_minutes=120     # 2 hours
    )
    
    result = optimizer.run()
    
    if result['success']:
        print("\n✅ Loop 1 complete: Target achieved!")
    else:
        print("\n⚠️  Loop 1 complete: Target not achieved")
        print("💡 Recommendation: Profile baseline with Nsight, or pivot to different workload")


if __name__ == "__main__":
    main()

