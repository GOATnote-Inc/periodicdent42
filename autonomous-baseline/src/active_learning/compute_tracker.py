"""
Compute Budget Tracking (NeurIPS/ICLR Compliance)

Tracks computational costs (time) during active learning experiments.

Protocol Contract:
- Track preprocessing, selection, training, evaluation times separately
- Compute Time-To-Accuracy (TTA) metrics
- Enable fair comparison across methods
- All times in seconds (convert to minutes for reporting)
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComputeTracker:
    """
    Track computational costs during active learning.
    
    Measures wall-clock time for:
    - Preprocessing (shared across all methods)
    - Selection (acquisition function + diversity)
    - Training (model fitting)
    - Evaluation (test set metrics)
    
    Example:
        >>> tracker = ComputeTracker()
        >>> 
        >>> # Preprocessing
        >>> t0 = time.time()
        >>> preprocess_data()
        >>> tracker.log_preprocessing(time.time() - t0)
        >>> 
        >>> for round_num in range(10):
        >>>     # Selection
        >>>     t0 = time.time()
        >>>     selected = acquisition_fn()
        >>>     tracker.log_selection(time.time() - t0)
        >>>     
        >>>     # Training
        >>>     t0 = time.time()
        >>>     model.fit(X_train, y_train)
        >>>     tracker.log_training(time.time() - t0)
        >>>     
        >>>     # Evaluation
        >>>     t0 = time.time()
        >>>     accuracy = evaluate_model(model)
        >>>     tracker.log_evaluation(time.time() - t0, accuracy)
        >>> 
        >>> # Summary
        >>> summary = tracker.summary()
        >>> print(f"Total time: {summary['total_time_min']:.2f} minutes")
        >>> print(f"TTA (90% acc): {tracker.compute_tta(0.90):.2f} minutes")
    """
    
    # Shared preprocessing (all methods)
    preprocessing_time: float = 0.0
    
    # Per-round times
    selection_times: List[float] = field(default_factory=list)
    training_times: List[float] = field(default_factory=list)
    evaluation_times: List[float] = field(default_factory=list)
    
    # Test metrics for TTA computation
    test_metrics: List[Dict[str, float]] = field(default_factory=list)
    
    def log_preprocessing(self, duration: float) -> None:
        """
        Log shared preprocessing time.
        
        Args:
            duration: Time in seconds
        """
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")
        
        self.preprocessing_time += duration
        
        logger.debug(f"Preprocessing: +{duration:.2f}s (total: {self.preprocessing_time:.2f}s)")
    
    def log_selection(self, duration: float) -> None:
        """
        Log acquisition function selection time.
        
        Args:
            duration: Time in seconds
        """
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")
        
        self.selection_times.append(duration)
        
        logger.debug(
            f"Round {len(self.selection_times)}: Selection {duration:.2f}s"
        )
    
    def log_training(self, duration: float) -> None:
        """
        Log model training time.
        
        Args:
            duration: Time in seconds
        """
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")
        
        self.training_times.append(duration)
        
        logger.debug(
            f"Round {len(self.training_times)}: Training {duration:.2f}s"
        )
    
    def log_evaluation(
        self, 
        duration: float, 
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log evaluation time and metrics.
        
        Args:
            duration: Time in seconds
            metrics: Test set metrics (e.g., {'accuracy': 0.85, 'rmse': 0.12})
        """
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")
        
        self.evaluation_times.append(duration)
        
        if metrics is not None:
            self.test_metrics.append(metrics)
        
        logger.debug(
            f"Round {len(self.evaluation_times)}: Evaluation {duration:.2f}s"
        )
    
    def compute_tta(
        self, 
        target_accuracy: float,
        metric_name: str = 'accuracy'
    ) -> Optional[float]:
        """
        Compute Time-To-Accuracy: minutes to reach target.
        
        Args:
            target_accuracy: Target accuracy (e.g., 0.90 for 90%)
            metric_name: Metric to track (e.g., 'accuracy', 'r2')
        
        Returns:
            Time in minutes to reach target, or None if not reached
        
        Example:
            >>> tta = tracker.compute_tta(target_accuracy=0.90)
            >>> if tta is not None:
            >>>     print(f"Reached 90% accuracy in {tta:.2f} minutes")
            >>> else:
            >>>     print("Did not reach 90% accuracy")
        """
        if not self.test_metrics:
            logger.warning("No test metrics available for TTA computation")
            return None
        
        cumulative_time = self.preprocessing_time
        
        for i, metrics in enumerate(self.test_metrics):
            # Add time for this round
            cumulative_time += (
                self.selection_times[i] if i < len(self.selection_times) else 0.0
            ) + (
                self.training_times[i] if i < len(self.training_times) else 0.0
            ) + (
                self.evaluation_times[i] if i < len(self.evaluation_times) else 0.0
            )
            
            # Check if target reached
            metric_value = metrics.get(metric_name)
            
            if metric_value is not None:
                # For metrics where higher is better (accuracy, r2)
                if metric_name in ['accuracy', 'r2', 'f1', 'auc']:
                    if metric_value >= target_accuracy:
                        logger.info(
                            f"TTA: Reached {metric_name}={target_accuracy:.4f} "
                            f"in {cumulative_time/60:.2f} minutes (round {i+1})"
                        )
                        return cumulative_time / 60.0  # Convert to minutes
                
                # For metrics where lower is better (rmse, mae)
                elif metric_name in ['rmse', 'mae', 'mse']:
                    if metric_value <= target_accuracy:
                        logger.info(
                            f"TTA: Reached {metric_name}={target_accuracy:.4f} "
                            f"in {cumulative_time/60:.2f} minutes (round {i+1})"
                        )
                        return cumulative_time / 60.0
        
        logger.info(
            f"TTA: Did not reach {metric_name}={target_accuracy:.4f} "
            f"(best: {metrics.get(metric_name, 'N/A')})"
        )
        return None
    
    def compute_total_time(self) -> float:
        """
        Compute total wall-clock time in minutes.
        
        Returns:
            Total time in minutes
        """
        return (
            self.preprocessing_time +
            sum(self.selection_times) +
            sum(self.training_times) +
            sum(self.evaluation_times)
        ) / 60.0
    
    def summary(self) -> Dict:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary with timing statistics
        """
        n_rounds = max(
            len(self.selection_times),
            len(self.training_times),
            len(self.evaluation_times)
        )
        
        summary = {
            'preprocessing_time_sec': self.preprocessing_time,
            'total_selection_time_sec': sum(self.selection_times),
            'total_training_time_sec': sum(self.training_times),
            'total_evaluation_time_sec': sum(self.evaluation_times),
            'total_time_min': self.compute_total_time(),
            'n_rounds': n_rounds,
        }
        
        # Per-round averages
        if n_rounds > 0:
            summary['mean_selection_time_sec'] = (
                sum(self.selection_times) / len(self.selection_times)
                if self.selection_times else 0.0
            )
            summary['mean_training_time_sec'] = (
                sum(self.training_times) / len(self.training_times)
                if self.training_times else 0.0
            )
            summary['mean_evaluation_time_sec'] = (
                sum(self.evaluation_times) / len(self.evaluation_times)
                if self.evaluation_times else 0.0
            )
        
        # Time breakdowns (percentages)
        total_sec = (
            self.preprocessing_time +
            sum(self.selection_times) +
            sum(self.training_times) +
            sum(self.evaluation_times)
        )
        
        if total_sec > 0:
            summary['pct_preprocessing'] = 100 * self.preprocessing_time / total_sec
            summary['pct_selection'] = 100 * sum(self.selection_times) / total_sec
            summary['pct_training'] = 100 * sum(self.training_times) / total_sec
            summary['pct_evaluation'] = 100 * sum(self.evaluation_times) / total_sec
        
        return summary
    
    def compute_time_per_label(self, n_labels: int) -> float:
        """
        Compute average time per label acquired (seconds/label).
        
        Args:
            n_labels: Total number of labels acquired
        
        Returns:
            Time per label in seconds
        """
        if n_labels <= 0:
            return 0.0
        
        total_time_sec = self.compute_total_time() * 60.0
        return total_time_sec / n_labels
    
    def compare_efficiency(
        self, 
        other: "ComputeTracker",
        method_name: str = "Method",
        baseline_name: str = "Baseline"
    ) -> Dict:
        """
        Compare computational efficiency against another tracker.
        
        Args:
            other: Another ComputeTracker instance
            method_name: Name of this method
            baseline_name: Name of baseline method
        
        Returns:
            Dictionary with comparison statistics
        """
        self_summary = self.summary()
        other_summary = other.summary()
        
        comparison = {
            f'{method_name}_total_time_min': self_summary['total_time_min'],
            f'{baseline_name}_total_time_min': other_summary['total_time_min'],
            'time_ratio': (
                self_summary['total_time_min'] / other_summary['total_time_min']
                if other_summary['total_time_min'] > 0 else float('inf')
            ),
            'time_difference_min': (
                self_summary['total_time_min'] - other_summary['total_time_min']
            ),
        }
        
        if comparison['time_ratio'] < 1.0:
            logger.info(
                f"{method_name} is {1/comparison['time_ratio']:.2f}x FASTER "
                f"than {baseline_name}"
            )
        else:
            logger.info(
                f"{method_name} is {comparison['time_ratio']:.2f}x SLOWER "
                f"than {baseline_name}"
            )
        
        return comparison
    
    def format_summary_table(self) -> str:
        """
        Format summary as ASCII table for logging.
        
        Returns:
            Formatted string
        """
        summary = self.summary()
        
        lines = [
            "=" * 60,
            "COMPUTE BUDGET SUMMARY",
            "=" * 60,
            f"Preprocessing:  {summary['preprocessing_time_sec']:8.2f}s "
            f"({summary.get('pct_preprocessing', 0):5.1f}%)",
            f"Selection:      {summary['total_selection_time_sec']:8.2f}s "
            f"({summary.get('pct_selection', 0):5.1f}%)",
            f"Training:       {summary['total_training_time_sec']:8.2f}s "
            f"({summary.get('pct_training', 0):5.1f}%)",
            f"Evaluation:     {summary['total_evaluation_time_sec']:8.2f}s "
            f"({summary.get('pct_evaluation', 0):5.1f}%)",
            "-" * 60,
            f"Total:          {summary['total_time_min']:8.2f} minutes",
            f"Rounds:         {summary['n_rounds']}",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"ComputeTracker("
            f"total_time={self.compute_total_time():.2f}min, "
            f"rounds={len(self.training_times)})"
        )


def compute_speedup(
    method_time: float,
    baseline_time: float
) -> float:
    """
    Compute speedup factor.
    
    Args:
        method_time: Time for method (minutes)
        baseline_time: Time for baseline (minutes)
    
    Returns:
        Speedup factor (>1.0 means faster, <1.0 means slower)
    """
    if baseline_time <= 0:
        return float('inf')
    
    return baseline_time / method_time

