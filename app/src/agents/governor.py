"""
Governor Agent: Budget allocation, safety checks, and resource management.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BudgetAllocation:
    """Budget allocation for discovery cycle."""
    
    dft_budget: float  # USD
    ml_budget: float  # USD
    max_dft_jobs: int
    max_ml_predictions: int


class GovernorAgent:
    """
    Governor Agent: Orchestrator and resource manager.
    
    Responsibilities:
    - Allocate compute budget (DFT vs ML)
    - Enforce safety constraints (toxicity, reactivity)
    - Prioritize experiments by information gain
    - Trigger emergency stops
    - Monitor agent health
    """
    
    def __init__(self, monthly_budget: float = 5000.0):
        """
        Initialize Governor Agent.
        
        Args:
            monthly_budget: Total monthly budget in USD
        """
        self.monthly_budget = monthly_budget
        self.spent_this_month = 0.0
        self.dft_jobs_queued = 0
        self.ml_predictions_made = 0
        
        logger.info(f"GovernorAgent initialized with ${monthly_budget:.2f}/month budget")
    
    def allocate_budget(self) -> BudgetAllocation:
        """
        Allocate budget for next discovery cycle.
        
        Strategy:
        - DFT: 99% of budget ($4,950/month → 99 jobs @ $50/job)
        - ML: 1% of budget ($50/month → 200K predictions @ $0.00025/pred)
        
        Returns:
            BudgetAllocation with limits
        """
        remaining = self.monthly_budget - self.spent_this_month
        
        dft_budget = min(remaining * 0.99, 5000)  # Cap at $5K
        ml_budget = remaining - dft_budget
        
        max_dft_jobs = int(dft_budget / 50)  # $50 per DFT job
        max_ml_predictions = int(ml_budget / 0.00025)  # $0.00025 per ML prediction
        
        allocation = BudgetAllocation(
            dft_budget=dft_budget,
            ml_budget=ml_budget,
            max_dft_jobs=max_dft_jobs,
            max_ml_predictions=max_ml_predictions,
        )
        
        logger.info(
            f"Budget allocated: DFT=${dft_budget:.2f} ({max_dft_jobs} jobs), "
            f"ML=${ml_budget:.2f} ({max_ml_predictions} preds)"
        )
        
        return allocation
    
    def check_safety(self, structure) -> tuple[bool, Optional[str]]:
        """
        Check if structure is safe to synthesize.
        
        Blocks:
        - Toxic elements (As, Cd, Hg, Pb, ...)
        - Radioactive (U, Pu, ...)
        - Explosive (perchlorate, azide, ...)
        
        Returns:
            (is_safe, reason_if_unsafe)
        """
        # Extract elements
        elements = {str(site.specie.symbol) for site in structure}
        
        # Toxic elements
        toxic = {"As", "Cd", "Hg", "Pb", "Tl", "Be"}
        if elements & toxic:
            return False, f"Contains toxic elements: {elements & toxic}"
        
        # Radioactive
        radioactive = {"U", "Pu", "Th", "Ra", "Ac", "Np", "Am"}
        if elements & radioactive:
            return False, f"Contains radioactive elements: {elements & radioactive}"
        
        # Explosive anions (would need to check composition, simplified here)
        # In real implementation, check for ClO4-, N3-, etc.
        
        return True, None
    
    def prioritize_experiments(self, candidates: List[Dict]) -> List[Dict]:
        """
        Prioritize experiments by information gain.
        
        Score = (Expected Tc) × (Uncertainty) / (Cost)
        
        High Tc + High Uncertainty + Low Cost = High Priority
        """
        def score(candidate):
            tc = candidate.get("tc_kelvin", 0)
            uncertainty = candidate.get("tc_std", 0)
            cost = candidate.get("cost", 50)  # Default DFT cost
            
            # Information gain score
            return (tc * uncertainty) / cost
        
        sorted_candidates = sorted(candidates, key=score, reverse=True)
        
        logger.info(f"Prioritized {len(candidates)} candidates by information gain")
        
        return sorted_candidates
    
    def emergency_stop(self, reason: str):
        """Trigger emergency stop of all running jobs."""
        logger.critical(f"🚨 EMERGENCY STOP: {reason}")
        # In production: cancel all DFT jobs, pause ML endpoints
        raise RuntimeError(f"Emergency stop triggered: {reason}")
    
    def get_status(self) -> Dict:
        """Get current governor status."""
        return {
            "monthly_budget": self.monthly_budget,
            "spent_this_month": self.spent_this_month,
            "remaining": self.monthly_budget - self.spent_this_month,
            "dft_jobs_queued": self.dft_jobs_queued,
            "ml_predictions_made": self.ml_predictions_made,
        }

