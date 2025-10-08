"""
matprov: Materials Provenance Tracking System

A Python package for tracking materials synthesis experiments with cryptographic
provenance, integrating with DVC for data versioning and MLflow for experiment tracking.

Core features:
- Experiment schema with Pydantic validation
- DVC integration for multi-GB file tracking
- Merkle tree for experiment lineage
- Sigstore integration for keyless signing
- CLI for common operations
"""

__version__ = "0.1.0"

from matprov.schema import (
    ExperimentMetadata,
    SynthesisParameters,
    CharacterizationData,
    PredictionLink,
    Outcome,
    MaterialsExperiment,
)

__all__ = [
    "ExperimentMetadata",
    "SynthesisParameters",
    "CharacterizationData",
    "PredictionLink",
    "Outcome",
    "MaterialsExperiment",
]

