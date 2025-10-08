"""
BETE-NET inference wrapper for Periodic Labs.

This module provides production-grade inference for electron-phonon coupling
and superconductor Tc prediction using the BETE-NET ensemble.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0 (see LICENSE)

Incorporates BETE-NET from University of Florida Hennig Group
(Apache 2.0, https://github.com/henniggroup/BETE-NET)
"""

from app.src.bete_net_io.inference import predict_tc, load_structure
from app.src.bete_net_io.batch import batch_screen
from app.src.bete_net_io.evidence import create_evidence_pack

__version__ = "1.0.0"
__all__ = ["predict_tc", "load_structure", "batch_screen", "create_evidence_pack"]

