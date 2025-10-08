"""
Prediction Registry: Track ML model predictions and experimental validation

SQLAlchemy-based database for tracking:
- Model predictions (material formula, predicted Tc, uncertainty)
- Experimental outcomes (actual Tc, validation status)
- Prediction errors (predicted - actual)
- Model performance metrics
"""

from matprov.registry.models import (
    Base,
    Model,
    Prediction,
    ExperimentOutcome,
    PredictionError,
)
from matprov.registry.database import Database
from matprov.registry.queries import PredictionQueries

__all__ = [
    "Base",
    "Model",
    "Prediction",
    "ExperimentOutcome",
    "PredictionError",
    "Database",
    "PredictionQueries",
]

