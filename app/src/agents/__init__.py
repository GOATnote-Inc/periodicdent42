"""
Multi-agent superconductor discovery system.

Agents:
- Governor: Budget allocation, safety checks, prioritization
- Proposer: Generate candidate structures
- Filter: Fast S2SNet screening
- Refiner: High-fidelity BETE-NET + BEE predictions
- Verifier: DFT validation queue
- Ranker: Evidence-based ranking
- Curator: Data ingestion and model retraining

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Proprietary License
"""

from app.src.agents.governor import GovernorAgent
from app.src.agents.proposer import ProposerAgent
from app.src.agents.filter import FilterAgent
from app.src.agents.refiner import RefinerAgent
from app.src.agents.verifier import VerifierAgent
from app.src.agents.ranker import RankerAgent
from app.src.agents.curator import CuratorAgent
from app.src.agents.orchestrator import DiscoveryOrchestrator

__all__ = [
    "GovernorAgent",
    "ProposerAgent",
    "FilterAgent",
    "RefinerAgent",
    "VerifierAgent",
    "RankerAgent",
    "CuratorAgent",
    "DiscoveryOrchestrator",
]

__version__ = "1.0.0-preview"

