"""
Visualization Module for HTC

Publication-quality figures for superconductor discovery results.
Simplified version for initial integration.

Copyright 2025 GOATnote Autonomous Research Lab Initiative  
Licensed under Apache 2.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_tc_distribution(
    predictions: list[Any],
    save_path: Optional[Path] = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot Tc distribution for predicted materials.
    
    Parameters
    ----------
    predictions : list
        List of SuperconductorPrediction objects
    save_path : Path, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    tc_values = [p.tc_predicted for p in predictions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(tc_values, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Critical Temperature (K)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Distribution of Predicted Tc Values', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved: {save_path}")
    
    return fig, ax


def plot_pareto_front(
    predictions: list[Any],
    pareto_front: list[Any],
    save_path: Optional[Path] = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot Pareto front for multi-objective optimization.
    
    Parameters
    ----------
    predictions : list
        All predictions
    pareto_front : list
        Pareto-optimal predictions
    save_path : Path, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    # All points
    all_tc = [p.tc_predicted for p in predictions]
    all_pressure = [p.pressure_required_gpa for p in predictions]
    
    # Pareto optimal points
    pareto_tc = [p.tc_predicted for p in pareto_front]
    pareto_pressure = [p.pressure_required_gpa for p in pareto_front]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(all_pressure, all_tc, alpha=0.5, s=60, label='All materials', color='gray')
    ax.scatter(pareto_pressure, pareto_tc, s=100, marker='*', 
               label='Pareto optimal', color='red', edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Required Pressure (GPa)', fontweight='bold')
    ax.set_ylabel('Critical Temperature (K)', fontweight='bold')
    ax.set_title('Multi-Objective Optimization: Tc vs Pressure', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved: {save_path}")
    
    return fig, ax


logger.info("HTC visualization module initialized")

