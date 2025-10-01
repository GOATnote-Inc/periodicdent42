"""
Adaptive optimization router - EXPERIMENTAL PROTOTYPE.

## Research Question:
Can we improve optimization efficiency by automatically selecting between
Bayesian Optimization and RL based on estimated noise levels?

## Current Evidence:
- Preliminary: RL outperformed BO at noise=2.0 on Branin function (p=0.0001, n=10)
- Unknown: Does this hold for other functions, noise levels, or real experiments?
- Untested: Can we reliably estimate noise and make good routing decisions?

## This is NOT:
- A production-ready system
- A validated method
- A guaranteed improvement

## This IS:
- An experimental tool to gather evidence
- A prototype for testing hypotheses
- A framework for systematic evaluation

## Use Responsibly:
- Document all experiments and results
- Report both successes AND failures
- Maintain scientific skepticism
- Avoid overstating capabilities
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.reasoning.adaptive.noise_estimator import NoiseEstimator, NoiseEstimate

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Available optimization methods."""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HYBRID = "hybrid"  # Future: combine RL exploration + BO exploitation
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """
    Container for routing decisions with full transparency.
    
    Includes all information needed to understand and audit the decision.
    """
    method: OptimizationMethod
    confidence: float  # 0.0 to 1.0 - how confident are we in this choice?
    noise_estimate: NoiseEstimate
    reasoning: str  # Human-readable explanation
    threshold_used: float  # What threshold was applied?
    alternative_methods: Dict[OptimizationMethod, float]  # Other options considered
    warnings: List[str]  # Any concerns about this decision
    
    def __str__(self) -> str:
        return (
            f"Method: {self.method.value}\n"
            f"Confidence: {self.confidence:.2%}\n"
            f"Noise: {self.noise_estimate}\n"
            f"Reasoning: {self.reasoning}\n"
            f"Warnings: {', '.join(self.warnings) if self.warnings else 'None'}"
        )


class AdaptiveRouter:
    """
    Experimental routing between optimization methods based on noise.
    
    ## Hypothesis:
    - Low noise (σ < 0.5): Bayesian Optimization is most sample-efficient
    - Medium noise (0.5 ≤ σ < 1.5): Unclear, needs more study
    - High noise (σ ≥ 1.5): RL may be more robust (preliminary evidence)
    
    ## Thresholds:
    These are TENTATIVE based on limited validation:
    - Conservative: Use BO unless strong evidence for RL
    - Based on single test function (Branin)
    - May not generalize to other problems
    
    ## Confidence Levels:
    - High (>0.8): Good data, clear decision
    - Medium (0.5-0.8): Some uncertainty, monitor performance
    - Low (<0.5): Insufficient data, default to BO (safest)
    """
    
    # TENTATIVE thresholds - subject to revision based on evidence
    DEFAULT_THRESHOLDS = {
        "bo_preferred": 0.5,    # σ < 0.5 → definitely use BO
        "rl_considered": 1.0,   # σ ≥ 1.0 → consider RL
        "rl_preferred": 1.5,    # σ ≥ 1.5 → prefer RL (preliminary evidence)
    }
    
    def __init__(
        self,
        noise_estimator: Optional[NoiseEstimator] = None,
        thresholds: Optional[Dict[str, float]] = None,
        require_high_confidence: bool = True
    ):
        """
        Initialize adaptive router.
        
        Args:
            noise_estimator: Custom noise estimator (default: uses standard)
            thresholds: Custom routing thresholds (default: uses DEFAULT_THRESHOLDS)
            require_high_confidence: If True, defaults to BO when confidence low
        """
        self.noise_estimator = noise_estimator or NoiseEstimator()
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.require_high_confidence = require_high_confidence
        
        # Track routing history for analysis
        self.routing_history: List[RoutingDecision] = []
        
        logger.info(
            f"AdaptiveRouter initialized with thresholds: {self.thresholds} "
            f"(EXPERIMENTAL - thresholds may change based on validation)"
        )
    
    def route(
        self,
        pilot_data: Dict[str, Any],
        problem_context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Decide which optimization method to use.
        
        Args:
            pilot_data: Data for noise estimation (see NoiseEstimator.estimate)
            problem_context: Optional context (dimensionality, constraints, etc.)
        
        Returns:
            RoutingDecision with method selection and full reasoning
        """
        warnings = []
        
        # Step 1: Estimate noise
        noise_estimate = self.noise_estimator.estimate(pilot_data)
        
        if not noise_estimate.reliable:
            warnings.append(
                f"Noise estimation unreliable (n={noise_estimate.sample_size}, "
                f"method={noise_estimate.method}). Defaulting to Bayesian Optimization."
            )
            
            decision = RoutingDecision(
                method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
                confidence=0.3,  # Low confidence
                noise_estimate=noise_estimate,
                reasoning="Insufficient data for reliable noise estimation. Using BO as safe default.",
                threshold_used=self.thresholds["bo_preferred"],
                alternative_methods={
                    OptimizationMethod.REINFORCEMENT_LEARNING: 0.0,
                    OptimizationMethod.HYBRID: 0.0
                },
                warnings=warnings
            )
            
            self.routing_history.append(decision)
            return decision
        
        # Step 2: Apply routing logic
        noise_std = noise_estimate.std
        
        # Clear BO territory
        if noise_std < self.thresholds["bo_preferred"]:
            decision = RoutingDecision(
                method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
                confidence=0.9,
                noise_estimate=noise_estimate,
                reasoning=f"Low noise (σ={noise_std:.3f}) - Bayesian Optimization excels here.",
                threshold_used=self.thresholds["bo_preferred"],
                alternative_methods={
                    OptimizationMethod.REINFORCEMENT_LEARNING: 0.1,
                    OptimizationMethod.HYBRID: 0.3
                },
                warnings=warnings
            )
        
        # Gray zone - need more research
        elif noise_std < self.thresholds["rl_considered"]:
            warnings.append(
                "Medium noise level - both BO and RL may work. "
                "This regime needs more validation."
            )
            
            # Conservative: use BO in gray zone
            decision = RoutingDecision(
                method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
                confidence=0.6,
                noise_estimate=noise_estimate,
                reasoning=(
                    f"Medium noise (σ={noise_std:.3f}) - unclear optimal method. "
                    "Using BO conservatively, but consider testing RL."
                ),
                threshold_used=self.thresholds["rl_considered"],
                alternative_methods={
                    OptimizationMethod.REINFORCEMENT_LEARNING: 0.4,
                    OptimizationMethod.HYBRID: 0.5
                },
                warnings=warnings
            )
        
        # Consider RL but not strong evidence yet
        elif noise_std < self.thresholds["rl_preferred"]:
            warnings.append(
                "Moderate-high noise - RL may help but evidence is preliminary. "
                "Monitor performance and compare to BO baseline."
            )
            
            decision = RoutingDecision(
                method=OptimizationMethod.REINFORCEMENT_LEARNING,
                confidence=0.65,
                noise_estimate=noise_estimate,
                reasoning=(
                    f"Moderate-high noise (σ={noise_std:.3f}) - RL may be more robust. "
                    "CAUTION: Based on limited validation. Compare to BO."
                ),
                threshold_used=self.thresholds["rl_preferred"],
                alternative_methods={
                    OptimizationMethod.BAYESIAN_OPTIMIZATION: 0.35,
                    OptimizationMethod.HYBRID: 0.6
                },
                warnings=warnings
            )
        
        # High noise - preliminary evidence for RL
        else:
            if noise_std >= 2.0:
                confidence = 0.75  # Stronger evidence at σ≥2.0
                reasoning = (
                    f"High noise (σ={noise_std:.3f}) - RL outperformed BO in validation "
                    f"at σ=2.0 (p=0.0001). However, only tested on Branin function."
                )
            else:
                confidence = 0.65
                reasoning = (
                    f"High noise (σ={noise_std:.3f}) - RL may be more robust, "
                    "but evidence is preliminary. Monitor performance."
                )
            
            warnings.append(
                "Using RL for high-noise optimization. "
                "This is based on limited validation (Branin function only). "
                "Please validate on your specific problem and report results."
            )
            
            decision = RoutingDecision(
                method=OptimizationMethod.REINFORCEMENT_LEARNING,
                confidence=confidence,
                noise_estimate=noise_estimate,
                reasoning=reasoning,
                threshold_used=self.thresholds["rl_preferred"],
                alternative_methods={
                    OptimizationMethod.BAYESIAN_OPTIMIZATION: 0.25,
                    OptimizationMethod.HYBRID: 0.5
                },
                warnings=warnings
            )
        
        # Log decision
        logger.info(f"Routing decision: {decision.method.value} (confidence={decision.confidence:.2%})")
        if warnings:
            for warning in warnings:
                logger.warning(warning)
        
        self.routing_history.append(decision)
        return decision
    
    def explain_decision(self, decision: RoutingDecision) -> str:
        """
        Generate detailed explanation of routing decision.
        
        Useful for debugging, transparency, and scientific documentation.
        """
        explanation = []
        explanation.append("=" * 70)
        explanation.append("ADAPTIVE ROUTING DECISION (EXPERIMENTAL)")
        explanation.append("=" * 70)
        explanation.append("")
        explanation.append(f"Selected Method: {decision.method.value.upper()}")
        explanation.append(f"Decision Confidence: {decision.confidence:.1%}")
        explanation.append("")
        explanation.append("Noise Estimation:")
        explanation.append(f"  {decision.noise_estimate}")
        explanation.append("")
        explanation.append("Reasoning:")
        explanation.append(f"  {decision.reasoning}")
        explanation.append("")
        explanation.append("Alternative Options Considered:")
        for method, score in decision.alternative_methods.items():
            explanation.append(f"  - {method.value}: {score:.1%}")
        explanation.append("")
        explanation.append(f"Threshold Applied: σ = {decision.threshold_used}")
        explanation.append("")
        
        if decision.warnings:
            explanation.append("⚠️  WARNINGS:")
            for warning in decision.warnings:
                explanation.append(f"  - {warning}")
            explanation.append("")
        
        explanation.append("=" * 70)
        explanation.append("REMEMBER: This is experimental. Validate on your problem!")
        explanation.append("=" * 70)
        
        return "\n".join(explanation)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on routing decisions made so far.
        
        Useful for analyzing whether the router is working as expected.
        """
        if not self.routing_history:
            return {"total_decisions": 0, "message": "No routing decisions made yet"}
        
        method_counts = {}
        confidence_scores = []
        noise_estimates = []
        
        for decision in self.routing_history:
            method = decision.method.value
            method_counts[method] = method_counts.get(method, 0) + 1
            confidence_scores.append(decision.confidence)
            noise_estimates.append(decision.noise_estimate.std)
        
        return {
            "total_decisions": len(self.routing_history),
            "method_distribution": method_counts,
            "average_confidence": np.mean(confidence_scores),
            "confidence_std": np.std(confidence_scores),
            "noise_range": (min(noise_estimates), max(noise_estimates)),
            "average_noise": np.mean(noise_estimates),
        }


def quick_route(observations: List[float]) -> OptimizationMethod:
    """
    Quick and dirty routing for prototyping.
    
    CAUTION: Makes strong assumptions and has no confidence assessment!
    For production use, use AdaptiveRouter class with proper pilot experiments.
    
    Args:
        observations: Sequential measurements for noise estimation
    
    Returns:
        Recommended optimization method (may not be optimal!)
    """
    if len(observations) < 3:
        logger.warning("Insufficient data for routing, defaulting to BO")
        return OptimizationMethod.BAYESIAN_OPTIMIZATION
    
    # Quick noise estimate
    diffs = np.diff(observations)
    noise_std = np.std(diffs) / np.sqrt(2)
    
    # Apply tentative thresholds
    if noise_std < 0.5:
        return OptimizationMethod.BAYESIAN_OPTIMIZATION
    elif noise_std < 1.5:
        return OptimizationMethod.BAYESIAN_OPTIMIZATION  # Conservative in gray zone
    else:
        return OptimizationMethod.REINFORCEMENT_LEARNING

