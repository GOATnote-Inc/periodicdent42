"""
Uncertainty Analysis Module for HTC

ISO GUM-compliant uncertainty quantification and sensitivity analysis
for superconductor predictions.

This is a streamlined version focused on HTC integration.
Full uncertainty analysis capabilities from uncertainty_analysis.py
can be added as needed.

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyBudget:
    """
    Uncertainty budget for a measurement or prediction.
    
    Follows ISO GUM (Guide to the expression of Uncertainty in Measurement)
    """
    
    value: float
    standard_uncertainty: float  # Combined uncertainty (u_c)
    type_a_uncertainty: float    # Statistical (random)
    type_b_uncertainty: float    # Systematic (bias)
    degrees_of_freedom: float
    coverage_factor: float = 2.0  # k for 95% CI
    
    def __post_init__(self) -> None:
        """Calculate expanded uncertainty: U = k * u_c"""
        self.expanded_uncertainty = self.coverage_factor * self.standard_uncertainty
    
    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Calculate confidence interval at given level"""
        from scipy import stats
        
        if self.degrees_of_freedom == np.inf:
            k = stats.norm.ppf((1 + level) / 2)
        else:
            k = stats.t.ppf((1 + level) / 2, self.degrees_of_freedom)
        
        U = k * self.standard_uncertainty
        return (self.value - U, self.value + U)
    
    def relative_uncertainty(self) -> float:
        """Relative standard uncertainty (u_r = u_c / |value|)"""
        return abs(self.standard_uncertainty / self.value) if self.value != 0 else np.inf


def propagate_uncertainty_simple(
    value: float,
    uncertainties: dict[str, float],
    sensitivities: dict[str, float]
) -> float:
    """
    Simple uncertainty propagation using sensitivity coefficients.
    
    Parameters
    ----------
    value : float
        Central value
    uncertainties : dict
        Input uncertainties {param: uncertainty}
    sensitivities : dict
        Sensitivity coefficients {param: dValue/dParam}
    
    Returns
    -------
    total_uncertainty : float
        Combined standard uncertainty
    """
    variance = 0.0
    for param, uncertainty in uncertainties.items():
        sensitivity = sensitivities.get(param, 0.0)
        variance += (sensitivity * uncertainty) ** 2
    
    return np.sqrt(variance)


logger.info("HTC uncertainty module initialized (simplified version)")

