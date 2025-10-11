"""
Active Learning Metrics (NeurIPS/ICLR Compliance)

Implements Area Under Active Learning Curve (AUALC) computation with proper
budget normalization and cross-seed comparison utilities.

Protocol Contract:
- AUALC = Σ [acc(i) · Δbudget(i)] / total_budget
- Budget normalized to [0, 1] for cross-dataset comparison
- Supports resampling to common grid for averaging across seeds
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)


class AUALCMetrics:
    """
    Compute Area Under Active Learning Curve (AUALC).
    
    AUALC measures the cumulative accuracy-weighted budget spent during
    active learning. Higher AUALC indicates better sample efficiency.
    
    Formula:
        AUALC = Σ [accuracy(round_i) · Δbudget_i] / total_budget
        where Δbudget_i = cumulative_labels(i) - cumulative_labels(i-1)
    
    Example:
        >>> tracker = AUALCMetrics(total_train_size=1000)
        >>> tracker.add_round(accuracy=0.75, labels_added=50, round_idx=1)
        >>> tracker.add_round(accuracy=0.80, labels_added=50, round_idx=2)
        >>> aualc = tracker.compute_aualc()
        >>> print(f"AUALC: {aualc:.4f}")
        AUALC: 0.7750
    """
    
    def __init__(self, total_train_size: int):
        """
        Initialize AUALC tracker.
        
        Args:
            total_train_size: Total size of training pool (for normalization)
        """
        if total_train_size <= 0:
            raise ValueError(
                f"total_train_size must be positive, got {total_train_size}"
            )
        
        self.total_train_size = total_train_size
        self.accuracies: List[float] = []
        self.cumulative_labels: List[int] = []
        self.round_indices: List[int] = []
        
        logger.debug(
            f"Initialized AUALCMetrics with total_train_size={total_train_size}"
        )
    
    def add_round(
        self, 
        accuracy: float, 
        labels_added: int, 
        round_idx: int
    ) -> None:
        """
        Add results from one active learning round.
        
        Args:
            accuracy: Test set accuracy after this round (0.0 to 1.0)
            labels_added: Number of labels added in this round
            round_idx: Round index (for logging/debugging)
        
        Raises:
            ValueError: If accuracy out of range or labels_added invalid
        """
        # Validation
        if not (0.0 <= accuracy <= 1.0):
            raise ValueError(
                f"Accuracy must be in [0, 1], got {accuracy} in round {round_idx}"
            )
        
        if labels_added <= 0:
            raise ValueError(
                f"labels_added must be positive, got {labels_added} in round {round_idx}"
            )
        
        # Compute cumulative labels
        cumulative = (
            self.cumulative_labels[-1] + labels_added
            if self.cumulative_labels
            else labels_added
        )
        
        if cumulative > self.total_train_size:
            warnings.warn(
                f"Round {round_idx}: cumulative labels ({cumulative}) exceeds "
                f"total_train_size ({self.total_train_size}). Capping."
            )
            cumulative = self.total_train_size
        
        # Store
        self.accuracies.append(accuracy)
        self.cumulative_labels.append(cumulative)
        self.round_indices.append(round_idx)
        
        logger.debug(
            f"Round {round_idx}: accuracy={accuracy:.4f}, "
            f"cumulative_labels={cumulative}/{self.total_train_size}"
        )
    
    def compute_aualc(self) -> float:
        """
        Compute Area Under Active Learning Curve.
        
        Returns:
            AUALC value (0.0 to 1.0, higher is better)
        
        Raises:
            RuntimeError: If no rounds have been added
        """
        if not self.accuracies:
            raise RuntimeError("Cannot compute AUALC: no rounds added")
        
        # Compute budget fractions
        budget_fractions = [
            labels / self.total_train_size 
            for labels in self.cumulative_labels
        ]
        
        # Check monotonicity
        if not all(
            budget_fractions[i] <= budget_fractions[i + 1] 
            for i in range(len(budget_fractions) - 1)
        ):
            warnings.warn(
                "Budget fractions are not monotonic! "
                "This indicates a logic error in tracking."
            )
        
        # Compute AUALC using trapezoidal rule
        aualc = 0.0
        for i, acc in enumerate(self.accuracies):
            if i == 0:
                # First round: area from 0 to budget_fractions[0]
                delta_budget = budget_fractions[i]
            else:
                # Subsequent rounds: area between previous and current
                delta_budget = budget_fractions[i] - budget_fractions[i - 1]
            
            # Accumulate: accuracy * budget_spent
            aualc += acc * delta_budget
        
        # Normalize by final budget fraction
        final_budget_fraction = budget_fractions[-1]
        if final_budget_fraction > 0:
            aualc_normalized = aualc / final_budget_fraction
        else:
            aualc_normalized = 0.0
        
        logger.info(
            f"AUALC computed: {aualc_normalized:.4f} "
            f"(final_budget_fraction={final_budget_fraction:.4f})"
        )
        
        return aualc_normalized
    
    def get_history(self) -> Dict[str, List]:
        """
        Get complete history of active learning rounds.
        
        Returns:
            Dictionary with keys:
            - 'accuracies': List of test accuracies
            - 'cumulative_labels': List of cumulative labels acquired
            - 'budget_fractions': List of budget fractions (0.0 to 1.0)
            - 'round_indices': List of round indices
        """
        budget_fractions = [
            labels / self.total_train_size 
            for labels in self.cumulative_labels
        ]
        
        return {
            'accuracies': self.accuracies,
            'cumulative_labels': self.cumulative_labels,
            'budget_fractions': budget_fractions,
            'round_indices': self.round_indices,
        }
    
    def summary(self) -> Dict:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.accuracies:
            return {
                'aualc': 0.0,
                'n_rounds': 0,
                'total_labels_used': 0,
                'final_accuracy': 0.0,
            }
        
        return {
            'aualc': self.compute_aualc(),
            'n_rounds': len(self.accuracies),
            'total_labels_used': self.cumulative_labels[-1],
            'budget_fraction_used': self.cumulative_labels[-1] / self.total_train_size,
            'final_accuracy': self.accuracies[-1],
            'initial_accuracy': self.accuracies[0],
            'accuracy_improvement': self.accuracies[-1] - self.accuracies[0],
        }


def compute_aualc(
    test_accuracies: List[float],
    labels_acquired: List[int],
    total_pool_size: int
) -> float:
    """
    Convenience function to compute AUALC from lists.
    
    Args:
        test_accuracies: Test accuracy after each round
        labels_acquired: Cumulative labels acquired after each round
        total_pool_size: Total size of training pool
    
    Returns:
        AUALC value (higher is better)
    
    Example:
        >>> aualc = compute_aualc(
        ...     test_accuracies=[0.75, 0.80, 0.85],
        ...     labels_acquired=[50, 100, 150],
        ...     total_pool_size=1000
        ... )
        >>> print(f"AUALC: {aualc:.4f}")
    """
    if len(test_accuracies) != len(labels_acquired):
        raise ValueError(
            f"Mismatched lengths: accuracies={len(test_accuracies)}, "
            f"labels={len(labels_acquired)}"
        )
    
    if not test_accuracies:
        raise ValueError("Empty input lists")
    
    # Use AUALCMetrics class
    tracker = AUALCMetrics(total_pool_size)
    
    for i, (acc, cum_labels) in enumerate(zip(test_accuracies, labels_acquired)):
        # Compute labels added in this round
        if i == 0:
            labels_added = cum_labels
        else:
            labels_added = cum_labels - labels_acquired[i - 1]
        
        tracker.add_round(
            accuracy=acc,
            labels_added=labels_added,
            round_idx=i + 1
        )
    
    return tracker.compute_aualc()


def resample_to_common_grid(
    budget_fractions: List[float],
    accuracies: List[float],
    target_grid: Optional[List[float]] = None,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample active learning curve to common budget grid.
    
    Enables averaging across multiple seeds with different budgets.
    Uses linear interpolation between measured points.
    
    Args:
        budget_fractions: Original budget fractions (0.0 to 1.0)
        accuracies: Corresponding test accuracies
        target_grid: Target budget grid (if None, create uniform grid)
        n_points: Number of points in uniform grid (if target_grid is None)
    
    Returns:
        Tuple of (resampled_budget_fractions, resampled_accuracies)
    
    Example:
        >>> # Resample 3 seeds to common grid
        >>> seed1_fracs, seed1_accs = resample_to_common_grid(
        ...     budget_fractions=[0.1, 0.2, 0.3],
        ...     accuracies=[0.70, 0.75, 0.80]
        ... )
        >>> seed2_fracs, seed2_accs = resample_to_common_grid(
        ...     budget_fractions=[0.15, 0.25],
        ...     accuracies=[0.68, 0.77]
        ... )
        >>> # Now average across seeds
        >>> mean_accs = (seed1_accs + seed2_accs) / 2
    """
    budget_fractions = np.asarray(budget_fractions)
    accuracies = np.asarray(accuracies)
    
    if len(budget_fractions) != len(accuracies):
        raise ValueError(
            f"Mismatched lengths: budget_fractions={len(budget_fractions)}, "
            f"accuracies={len(accuracies)}"
        )
    
    # Create target grid
    if target_grid is None:
        target_grid = np.linspace(
            budget_fractions.min(),
            budget_fractions.max(),
            n_points
        )
    else:
        target_grid = np.asarray(target_grid)
    
    # Interpolate
    resampled_accuracies = np.interp(
        target_grid,
        budget_fractions,
        accuracies,
        left=accuracies[0],  # Extrapolate with first value
        right=accuracies[-1]  # Extrapolate with last value
    )
    
    return target_grid, resampled_accuracies


def compute_aualc_gain(
    aualc_method: float,
    aualc_baseline: float
) -> float:
    """
    Compute AUALC gain over baseline.
    
    Args:
        aualc_method: AUALC of method being evaluated
        aualc_baseline: AUALC of baseline (e.g., random sampling)
    
    Returns:
        AUALC gain (positive means improvement)
    
    Example:
        >>> gain = compute_aualc_gain(aualc_method=0.82, aualc_baseline=0.75)
        >>> print(f"Gain: +{gain:.2f} pp·frac")
        Gain: +0.07 pp·frac
    """
    return aualc_method - aualc_baseline


def verify_aualc_improvement(
    aualc_method: float,
    aualc_baseline: float,
    min_gain: float = 0.02,
    p_value: Optional[float] = None,
    alpha: float = 0.05
) -> Dict[str, bool]:
    """
    Verify that method improves over baseline (statistical + practical).
    
    Args:
        aualc_method: AUALC of method
        aualc_baseline: AUALC of baseline
        min_gain: Minimum practical gain (default: 0.02 = 2 pp·frac)
        p_value: Statistical significance (if available)
        alpha: Significance level (default: 0.05)
    
    Returns:
        Dictionary with 'practical_gain' and 'statistical_significance' bools
    """
    gain = compute_aualc_gain(aualc_method, aualc_baseline)
    
    result = {
        'practical_gain': gain >= min_gain,
        'gain': gain,
        'min_gain': min_gain,
    }
    
    if p_value is not None:
        result['statistical_significance'] = p_value < alpha
        result['p_value'] = p_value
        result['alpha'] = alpha
    
    return result

