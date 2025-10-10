"""
HTC (High-Temperature Superconductor) Optimization Module

This module provides comprehensive superconductor discovery capabilities:
- Tc prediction with McMillan-Allen-Dynes theory
- Multi-objective optimization (Tc vs pressure)
- Constraint validation (ξ ≤ 4.0 stability bounds)
- Uncertainty quantification

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

from __future__ import annotations

__version__ = "1.0.0"

# Core imports will be added as modules are created
try:
    from app.src.htc.domain import (
        SuperconductorPredictor,
        SuperconductorPrediction,
        predict_tc_with_uncertainty,
        compute_pareto_front,
        validate_against_known_materials,
        XiConstraintValidator,
        load_benchmark_materials,
        allen_dynes_tc,
        mcmillan_tc,
    )
    __all__ = [
        "SuperconductorPredictor",
        "SuperconductorPrediction",
        "predict_tc_with_uncertainty",
        "compute_pareto_front",
        "validate_against_known_materials",
        "XiConstraintValidator",
        "load_benchmark_materials",
        "allen_dynes_tc",
        "mcmillan_tc",
    ]
except ImportError:
    # Graceful degradation if dependencies not available
    __all__ = []

