"""
Agentic Loop: Self-improving optimization with safety rails.

Agent:
1. Runs simulations (automated)
2. Evaluates performance (metrics)
3. Proposes improvements (via LLM)
4. Validates safety (constraints)
5. Implements changes (if safe)
6. Logs feedback (learning)
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result from one simulation run."""
    config: Dict
    experiments_to_95: float
    final_best_value: float
    convergence_rate: float
    safety_score: float
    cost_estimate: float
    timestamp: float


@dataclass
class SafetyConstraints:
    """Safety constraints for agentic loop."""
    max_experiments: int = 100
    min_experiments: int = 10
    max_exploration_budget: int = 50
    min_exploitation_steps: int = 3
    max_cost_per_experiment: float = 500.0
    require_human_approval: bool = False


class AgenticOptimizer:
    """
    Self-improving optimizer with safety rails.
    
    Loop:
    1. Run simulations (test current config)
    2. Evaluate metrics (sample efficiency, safety, cost)
    3. Propose improvements (via heuristics or LLM)
    4. Validate safety (check constraints)
    5. Apply if safe (update config)
    6. Log feedback (track history)
    """
    
    def __init__(
        self,
        base_config: Dict,
        safety_constraints: SafetyConstraints,
        target_metric: str = "experiments_to_95",
        n_simulations_per_iteration: int = 5,
    ):
        self.config = base_config
        self.safety = safety_constraints
        self.target_metric = target_metric
        self.n_sims = n_simulations_per_iteration
        
        # History
        self.iteration_history: List[Dict] = []
        self.best_config = base_config.copy()
        self.best_metric = np.inf
        
        logger.info(f"AgenticOptimizer initialized")
        logger.info(f"Target: minimize {target_metric}")
        logger.info(f"Safety: {safety_constraints}")
    
    def run_iteration(self) -> Dict:
        """
        One iteration of agentic loop.
        
        Returns:
            metrics: Performance + safety + cost
        """
        iteration_id = len(self.iteration_history)
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration_id}")
        logger.info(f"{'='*60}")
        
        # 1. Run simulations with current config
        logger.info("Step 1: Running simulations...")
        results = self._run_simulations(self.config)
        
        # 2. Evaluate performance
        logger.info("Step 2: Evaluating metrics...")
        metrics = self._evaluate_metrics(results)
        
        # 3. Propose improvements
        logger.info("Step 3: Proposing improvements...")
        proposed_config = self._propose_improvements(metrics)
        
        # 4. Validate safety
        logger.info("Step 4: Validating safety...")
        is_safe, safety_report = self._validate_safety(proposed_config)
        
        # 5. Apply if safe
        if is_safe:
            logger.info("✓ Config is safe, applying...")
            self.config = proposed_config
            
            # Update best if better
            current_metric = metrics[self.target_metric]["mean"]
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_config = proposed_config.copy()
                logger.info(f"✓ New best {self.target_metric}: {current_metric:.2f}")
        else:
            logger.warning(f"✗ Config unsafe, rejected: {safety_report}")
        
        # 6. Log feedback
        iteration_summary = {
            "iteration": iteration_id,
            "config": self.config,
            "metrics": metrics,
            "safety": safety_report,
            "applied": is_safe,
            "timestamp": time.time(),
        }
        self.iteration_history.append(iteration_summary)
        
        return iteration_summary
    
    def _run_simulations(self, config: Dict) -> List[SimulationResult]:
        """Run N simulations with given config."""
        results = []
        
        for i in range(self.n_sims):
            # Simulate experiment optimization
            # (In production, this calls actual hybrid_optimizer)
            result = self._simulate_experiment_run(config)
            results.append(result)
            logger.info(f"  Sim {i+1}/{self.n_sims}: {self.target_metric}={result.experiments_to_95:.1f}")
        
        return results
    
    def _simulate_experiment_run(self, config: Dict) -> SimulationResult:
        """
        Simulate one experiment optimization run.
        
        Uses Branin function as test objective.
        """
        from src.reasoning.hybrid_optimizer import HybridOptimizer
        
        # Extract config
        exploration_thresh = config.get("exploration_threshold", 0.5)
        exploitation_budget = config.get("exploitation_budget", 10)
        
        # Create optimizer
        opt = HybridOptimizer(
            bounds={"x": (-5, 10), "y": (0, 15)},
            exploration_threshold=exploration_thresh,
            exploitation_budget=exploitation_budget,
        )
        
        # Branin function
        def branin(x):
            x1, x2 = x[0], x[1]
            a, b, c, r, s, t = 1.0, 5.1/(4*np.pi**2), 5.0/np.pi, 6.0, 10.0, 1.0/(8*np.pi)
            return -(a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)
        
        # Run optimization
        target_value = -0.398 * 0.95  # 95% of global optimum
        experiments_to_95 = 0
        
        for step in range(100):
            params = opt.propose_next()
            value = branin(params) + np.random.normal(0, 0.1)
            opt.update(params, value)
            
            if opt.best_value >= target_value and experiments_to_95 == 0:
                experiments_to_95 = step + 1
        
        if experiments_to_95 == 0:
            experiments_to_95 = 100  # Didn't reach target
        
        # Calculate metrics
        convergence_rate = opt.best_value / -0.398 if opt.best_value < 0 else 0
        safety_score = 1.0  # Placeholder (would check constraint violations)
        cost_estimate = experiments_to_95 * 50.0  # $50/experiment average
        
        return SimulationResult(
            config=config,
            experiments_to_95=float(experiments_to_95),
            final_best_value=float(opt.best_value),
            convergence_rate=float(convergence_rate),
            safety_score=safety_score,
            cost_estimate=cost_estimate,
            timestamp=time.time(),
        )
    
    def _evaluate_metrics(self, results: List[SimulationResult]) -> Dict:
        """Aggregate metrics from simulations."""
        metrics = {}
        
        for field in ["experiments_to_95", "final_best_value", "convergence_rate", "safety_score", "cost_estimate"]:
            values = [getattr(r, field) for r in results]
            metrics[field] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        
        return metrics
    
    def _propose_improvements(self, metrics: Dict) -> Dict:
        """
        Propose improved config based on metrics.
        
        Heuristics:
        - If slow convergence: increase exploration threshold
        - If unstable: decrease exploration threshold
        - If efficient: increase exploitation budget
        """
        new_config = self.config.copy()
        
        # Current performance
        experiments = metrics["experiments_to_95"]["mean"]
        stability = 1.0 / (metrics["experiments_to_95"]["std"] + 1e-6)
        
        # Heuristic adjustments
        if experiments > 25:  # Too slow
            # Increase exploration
            new_config["exploration_threshold"] = min(
                self.config["exploration_threshold"] * 1.2,
                0.9
            )
            logger.info(f"  Heuristic: Slow convergence, increasing exploration")
        elif experiments < 15:  # Very fast
            # Decrease exploration (exploit more)
            new_config["exploration_threshold"] = max(
                self.config["exploration_threshold"] * 0.8,
                0.1
            )
            logger.info(f"  Heuristic: Fast convergence, decreasing exploration")
        
        if stability > 5.0:  # Stable
            # Can afford longer exploitation
            new_config["exploitation_budget"] = min(
                self.config["exploitation_budget"] + 2,
                self.safety.max_exploration_budget
            )
            logger.info(f"  Heuristic: Stable, increasing exploitation budget")
        elif stability < 2.0:  # Unstable
            # Shorten exploitation
            new_config["exploitation_budget"] = max(
                self.config["exploitation_budget"] - 2,
                self.safety.min_exploitation_steps
            )
            logger.info(f"  Heuristic: Unstable, decreasing exploitation budget")
        
        return new_config
    
    def _validate_safety(self, config: Dict) -> Tuple[bool, Dict]:
        """
        Check if config satisfies safety constraints.
        
        Returns:
            is_safe: bool
            report: Dict with violations
        """
        violations = []
        
        # Check exploitation budget
        budget = config.get("exploitation_budget", 0)
        if budget < self.safety.min_exploitation_steps:
            violations.append(f"exploitation_budget ({budget}) < min ({self.safety.min_exploitation_steps})")
        if budget > self.safety.max_exploration_budget:
            violations.append(f"exploitation_budget ({budget}) > max ({self.safety.max_exploration_budget})")
        
        # Check exploration threshold
        thresh = config.get("exploration_threshold", 0)
        if thresh < 0.0 or thresh > 1.0:
            violations.append(f"exploration_threshold ({thresh}) not in [0, 1]")
        
        # Check if human approval required
        if self.safety.require_human_approval and len(self.iteration_history) > 0:
            violations.append("Human approval required (not implemented)")
        
        is_safe = len(violations) == 0
        
        report = {
            "is_safe": is_safe,
            "violations": violations,
            "config": config,
        }
        
        return is_safe, report
    
    def run_agentic_loop(self, n_iterations: int = 10) -> Dict:
        """
        Run full agentic loop for N iterations.
        
        Returns:
            summary: Best config, metrics, history
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"STARTING AGENTIC LOOP ({n_iterations} iterations)")
        logger.info(f"{'#'*60}\n")
        
        for i in range(n_iterations):
            self.run_iteration()
            
            # Early stopping if converged
            if len(self.iteration_history) >= 3:
                recent = [
                    h["metrics"][self.target_metric]["mean"]
                    for h in self.iteration_history[-3:]
                ]
                improvement = max(recent) - min(recent)
                if improvement < 0.5:
                    logger.info(f"\n✓ Converged (improvement < 0.5 in last 3 iterations)")
                    break
        
        # Summary
        summary = {
            "best_config": self.best_config,
            "best_metric": self.best_metric,
            "n_iterations": len(self.iteration_history),
            "total_simulations": len(self.iteration_history) * self.n_sims,
            "history": self.iteration_history,
        }
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"AGENTIC LOOP COMPLETE")
        logger.info(f"{'#'*60}")
        logger.info(f"Best {self.target_metric}: {self.best_metric:.2f}")
        logger.info(f"Best config: {self.best_config}")
        
        return summary
    
    def save_history(self, filepath: str):
        """Save iteration history to JSON."""
        with open(filepath, 'w') as f:
            json.dump({
                "best_config": self.best_config,
                "best_metric": float(self.best_metric),
                "history": self.iteration_history,
            }, f, indent=2)
        logger.info(f"Saved history to {filepath}")

