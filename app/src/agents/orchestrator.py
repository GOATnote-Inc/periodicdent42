"""
Discovery Orchestrator: Coordinates multi-agent discovery cycles.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryCycleResult:
    """Result from one complete discovery cycle."""
    
    cycle_id: str
    start_time: str
    end_time: str
    
    # Counts
    candidates_generated: int
    candidates_filtered: int
    candidates_refined: int
    candidates_verified: int
    top_candidates: int
    
    # Top ranked structures
    ranked_structures: List[Dict]
    evidence_packs: List[str]
    
    # Budget
    cost_ml: float
    cost_dft: float
    cost_total: float


class DiscoveryOrchestrator:
    """
    Discovery Orchestrator: Coordinates multi-agent workflows.
    
    Workflow:
    1. Governor checks budget
    2. Proposer generates 10K candidates
    3. Filter screens with S2SNet → 1K
    4. Refiner predicts with BETE-NET + BEE → Tc + σ
    5. Verifier queues high-uncertainty for DFT
    6. Ranker generates top 20 + evidence packs
    7. Curator ingests DFT results + retrains
    """
    
    def __init__(self):
        """Initialize orchestrator with all agents."""
        from app.src.agents.governor import GovernorAgent
        from app.src.agents.proposer import ProposerAgent
        from app.src.agents.filter import FilterAgent
        from app.src.agents.refiner import RefinerAgent
        from app.src.agents.verifier import VerifierAgent
        from app.src.agents.ranker import RankerAgent
        from app.src.agents.curator import CuratorAgent
        
        self.agents = {
            "governor": GovernorAgent(),
            "proposer": ProposerAgent(),
            "filter": FilterAgent(),
            "refiner": RefinerAgent(),
            "verifier": VerifierAgent(),
            "ranker": RankerAgent(),
            "curator": CuratorAgent(),
        }
        
        self.cycle_count = 0
        
        logger.info("DiscoveryOrchestrator initialized with 7 agents")
    
    def run_discovery_cycle(self, cycle_config: Dict = None) -> DiscoveryCycleResult:
        """
        Run one complete discovery cycle.
        
        Args:
            cycle_config: Optional configuration overrides
        
        Returns:
            DiscoveryCycleResult with metrics and outputs
        """
        cycle_config = cycle_config or {}
        self.cycle_count += 1
        cycle_id = f"cycle-{self.cycle_count:03d}"
        
        start_time = datetime.utcnow().isoformat() + "Z"
        logger.info(f"🚀 Starting discovery cycle {cycle_id}")
        
        # Step 1: Check budget
        logger.info("Step 1: Checking budget...")
        allocation = self.agents["governor"].allocate_budget()
        
        if allocation.max_ml_predictions < 10000:
            logger.warning(f"⚠️  Insufficient budget for full cycle: {allocation}")
            return self._abort_cycle(cycle_id, start_time, "Insufficient budget")
        
        # Step 2: Generate candidates
        logger.info(f"Step 2: Generating {cycle_config.get('n_candidates', 10000)} candidates...")
        candidates = self.agents["proposer"].generate_candidates(
            n=cycle_config.get("n_candidates", 10000)
        )
        logger.info(f"✅ Generated {len(candidates)} candidates")
        
        # Step 3: Fast screening
        logger.info("Step 3: Fast screening with S2SNet...")
        filtered = self.agents["filter"].screen(
            candidates,
            top_k=cycle_config.get("filter_top_k", 1000)
        )
        logger.info(f"✅ Filtered to {len(filtered)} candidates")
        
        # Step 4: High-fidelity refinement
        logger.info("Step 4: Refining with BETE-NET + BEE...")
        refined = self.agents["refiner"].predict_with_uncertainty(filtered)
        logger.info(f"✅ Refined {len(refined)} candidates")
        
        # Step 5: Queue for verification
        logger.info("Step 5: Queueing high-uncertainty candidates for DFT...")
        verification_queue = self.agents["verifier"].queue_high_uncertainty(
            refined,
            max_jobs=allocation.max_dft_jobs
        )
        logger.info(f"✅ Queued {len(verification_queue)} for DFT verification")
        
        # Step 6: Rank and generate evidence
        logger.info("Step 6: Ranking and generating evidence packs...")
        ranked = self.agents["ranker"].rank(
            refined,
            top_k=cycle_config.get("top_k", 20)
        )
        evidence_packs = [
            self.agents["ranker"].generate_evidence_pack(s, p)
            for s, p in ranked
        ]
        logger.info(f"✅ Ranked top {len(ranked)} candidates with evidence packs")
        
        # Step 7: Ingest completed DFT results
        logger.info("Step 7: Ingesting completed DFT results...")
        completed_dft = self.agents["verifier"].get_completed_jobs()
        if completed_dft:
            self.agents["curator"].ingest_dft_results(completed_dft)
            logger.info(f"✅ Ingested {len(completed_dft)} DFT results")
            
            # Trigger retrain if enough new data
            if len(completed_dft) >= 100:
                logger.info("Triggering model retraining...")
                self.agents["curator"].trigger_retrain()
        
        end_time = datetime.utcnow().isoformat() + "Z"
        
        # Compute costs
        cost_ml = len(candidates) * 0.00001 + len(filtered) * 0.000024  # S2SNet + BETE-NET
        cost_dft = len(verification_queue) * 50.0
        cost_total = cost_ml + cost_dft
        
        result = DiscoveryCycleResult(
            cycle_id=cycle_id,
            start_time=start_time,
            end_time=end_time,
            candidates_generated=len(candidates),
            candidates_filtered=len(filtered),
            candidates_refined=len(refined),
            candidates_verified=len(verification_queue),
            top_candidates=len(ranked),
            ranked_structures=ranked,
            evidence_packs=evidence_packs,
            cost_ml=cost_ml,
            cost_dft=cost_dft,
            cost_total=cost_total,
        )
        
        logger.info(
            f"✅ Discovery cycle {cycle_id} complete: "
            f"{len(ranked)} top candidates, ${cost_total:.2f} total cost"
        )
        
        return result
    
    def _abort_cycle(self, cycle_id: str, start_time: str, reason: str):
        """Abort cycle with minimal result."""
        logger.error(f"❌ Cycle {cycle_id} aborted: {reason}")
        
        return DiscoveryCycleResult(
            cycle_id=cycle_id,
            start_time=start_time,
            end_time=datetime.utcnow().isoformat() + "Z",
            candidates_generated=0,
            candidates_filtered=0,
            candidates_refined=0,
            candidates_verified=0,
            top_candidates=0,
            ranked_structures=[],
            evidence_packs=[],
            cost_ml=0.0,
            cost_dft=0.0,
            cost_total=0.0,
        )
    
    def get_status(self) -> Dict:
        """Get status of all agents."""
        return {
            "cycle_count": self.cycle_count,
            "agents": {
                name: agent.get_status() if hasattr(agent, "get_status") else "active"
                for name, agent in self.agents.items()
            },
        }

