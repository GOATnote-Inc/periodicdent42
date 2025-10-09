"""Active learning pipeline for efficient data acquisition.

Integrates:
- Phase 5: OOD detection (filter candidates)
- Phase 6: Active learning (acquisition + diversity)
- Phase 4: Conformal prediction (GO/NO-GO gates)

Usage:
    python -m src.pipelines.al_pipeline configs/al_ucb.yaml
"""

from pathlib import Path
from typing import Optional, Callable
import json
import time

import numpy as np
import pandas as pd

from src.guards.ood_detectors import create_ood_detector
from src.active_learning.acquisition import create_acquisition_function
from src.active_learning.diversity import create_diversity_selector
from src.active_learning.loop import ActiveLearningLoop, go_no_go_gate
from src.models.base import BaseUncertaintyModel


class ActiveLearningPipeline:
    """
    Active learning pipeline with OOD filtering and GO/NO-GO gates.
    
    Pipeline stages:
        1. OOD detection (filter unsafe candidates)
        2. Active learning loop (acquire labels efficiently)
        3. GO/NO-GO gates (deployment decisions)
        4. Performance tracking (RMSE vs budget)
    """
    
    def __init__(
        self,
        base_model: BaseUncertaintyModel,
        acquisition_method: str = "ucb",
        acquisition_kwargs: Optional[dict] = None,
        diversity_method: Optional[str] = None,
        diversity_kwargs: Optional[dict] = None,
        ood_method: str = "mahalanobis",
        ood_kwargs: Optional[dict] = None,
        budget: int = 100,
        batch_size: int = 10,
        random_state: int = 42,
        artifacts_dir: Path = Path("artifacts/al"),
    ):
        """
        Initialize active learning pipeline.
        
        Args:
            base_model: Uncertainty-aware model
            acquisition_method: Acquisition function (ucb, ei, maxvar, eig_proxy, thompson)
            acquisition_kwargs: Kwargs for acquisition function
            diversity_method: Diversity selector (k_medoids, greedy, dpp, None)
            diversity_kwargs: Kwargs for diversity selector
            ood_method: OOD detection method (mahalanobis, kde, conformal)
            ood_kwargs: Kwargs for OOD detector
            budget: Maximum labels to acquire
            batch_size: Batch size per iteration
            random_state: Random seed
            artifacts_dir: Directory to save artifacts
        """
        self.base_model = base_model
        self.acquisition_method = acquisition_method
        self.acquisition_kwargs = acquisition_kwargs or {}
        self.diversity_method = diversity_method
        self.diversity_kwargs = diversity_kwargs or {}
        self.ood_method = ood_method
        self.ood_kwargs = ood_kwargs or {}
        self.budget = budget
        self.batch_size = batch_size
        self.random_state = random_state
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline components
        self.ood_detector = None
        self.acquisition_fn = None
        self.diversity_selector = None
        self.al_loop = None
        
        # Results
        self.results_ = {}
    
    def run(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        y_unlabeled: Optional[np.ndarray] = None,  # For simulation
        go_no_go_threshold_min: Optional[float] = None,
        go_no_go_threshold_max: Optional[float] = None,
    ) -> dict:
        """
        Run active learning pipeline.
        
        Args:
            X_labeled: Initial labeled features (N_labeled, D)
            y_labeled: Initial labeled targets (N_labeled,)
            X_unlabeled: Unlabeled pool features (N_unlabeled, D)
            y_unlabeled: True labels (for simulation only)
            go_no_go_threshold_min: Minimum acceptable value (e.g., T_c > 77K)
            go_no_go_threshold_max: Maximum acceptable value (e.g., T_c < 200K)
            
        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()
        
        print("=" * 80)
        print("ACTIVE LEARNING PIPELINE START")
        print("=" * 80)
        
        # 1. OOD detection
        print("\n[1/4] OOD Detection & Filtering")
        X_pool_filtered, filter_stats = self._filter_ood_candidates(
            X_labeled, X_unlabeled
        )
        print(f"  ✓ Filtered: {filter_stats['n_ood']} OOD, {filter_stats['n_retained']} retained")
        
        # If y_unlabeled provided, filter it too
        if y_unlabeled is not None:
            is_ood = self.ood_detector.predict(X_unlabeled)
            y_pool_filtered = y_unlabeled[~is_ood]
        else:
            y_pool_filtered = None
        
        # 2. Active learning
        print("\n[2/4] Active Learning Loop")
        al_results = self._run_active_learning(
            X_labeled, y_labeled, X_pool_filtered, y_pool_filtered
        )
        print(f"  ✓ Budget used: {al_results['budget_used']}/{self.budget}")
        print(f"  ✓ Iterations: {len(al_results['history'])}")
        
        # 3. GO/NO-GO decisions
        print("\n[3/4] GO/NO-GO Gates")
        if go_no_go_threshold_min is not None or go_no_go_threshold_max is not None:
            go_no_go_results = self._apply_go_no_go_gates(
                al_results["X_train"],
                go_no_go_threshold_min or -np.inf,
                go_no_go_threshold_max or np.inf,
            )
            print(f"  ✓ GO: {go_no_go_results['n_go']}")
            print(f"  ✓ MAYBE: {go_no_go_results['n_maybe']}")
            print(f"  ✓ NO-GO: {go_no_go_results['n_no_go']}")
        else:
            go_no_go_results = None
            print("  ⊘ Skipped (no thresholds provided)")
        
        # 4. Save artifacts
        print("\n[4/4] Artifact Generation")
        self._save_artifacts(al_results, go_no_go_results, filter_stats)
        print(f"  ✓ Artifacts saved to: {self.artifacts_dir}")
        
        # Compile results
        elapsed_time = time.time() - start_time
        
        self.results_ = {
            "ood_filtering": filter_stats,
            "active_learning": {
                "budget_used": al_results["budget_used"],
                "n_iterations": len(al_results["history"]),
                "final_labeled_size": len(al_results["X_train"]),
            },
            "go_no_go": go_no_go_results,
            "artifacts_dir": str(self.artifacts_dir),
            "elapsed_time": elapsed_time,
        }
        
        print("\n" + "=" * 80)
        print(f"ACTIVE LEARNING PIPELINE COMPLETE ({elapsed_time:.2f}s)")
        print("=" * 80)
        
        return self.results_
    
    def _filter_ood_candidates(
        self, X_labeled: np.ndarray, X_unlabeled: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Filter OOD candidates from unlabeled pool."""
        # Create and fit OOD detector
        self.ood_detector = create_ood_detector(self.ood_method, **self.ood_kwargs)
        self.ood_detector.fit(X_labeled)
        
        # Detect OOD samples
        is_ood = self.ood_detector.predict(X_unlabeled)
        
        # Filter
        X_pool_filtered = X_unlabeled[~is_ood]
        
        stats = {
            "n_total": int(len(X_unlabeled)),
            "n_ood": int(is_ood.sum()),
            "n_retained": int(len(X_pool_filtered)),
            "ood_rate": float(is_ood.mean()),
        }
        
        return X_pool_filtered, stats
    
    def _run_active_learning(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_pool: np.ndarray,
        y_pool: Optional[np.ndarray],
    ) -> dict:
        """Run active learning loop."""
        # Create acquisition function
        self.acquisition_fn = create_acquisition_function(
            self.acquisition_method, **self.acquisition_kwargs
        )
        
        # Create diversity selector (optional)
        if self.diversity_method is not None:
            self.diversity_selector = create_diversity_selector(
                self.diversity_method, **self.diversity_kwargs
            )
        else:
            self.diversity_selector = None
        
        # Create AL loop
        self.al_loop = ActiveLearningLoop(
            base_model=self.base_model,
            acquisition_fn=self.acquisition_fn,
            diversity_selector=self.diversity_selector,
            budget=self.budget,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        
        # Run
        if y_pool is None:
            raise ValueError("y_unlabeled must be provided for simulation")
        
        results = self.al_loop.run(X_labeled, y_labeled, X_pool, y_pool)
        
        return results
    
    def _apply_go_no_go_gates(
        self,
        X_final: np.ndarray,
        threshold_min: float,
        threshold_max: float,
    ) -> dict:
        """Apply GO/NO-GO gates to final labeled set."""
        # Get predictions with uncertainty
        y_pred, y_lower, y_upper = self.base_model.predict_with_uncertainty(X_final)
        
        # Dummy y_std (not used in gate, but needed for signature compatibility)
        y_std = (y_upper - y_lower) / 4  # Approximate std from interval
        
        # Apply gate
        decisions = go_no_go_gate(
            y_pred, y_std, y_lower, y_upper,
            threshold_min, threshold_max
        )
        
        # Count decisions
        n_go = (decisions == 1).sum()
        n_maybe = (decisions == 0).sum()
        n_no_go = (decisions == -1).sum()
        
        return {
            "n_go": int(n_go),
            "n_maybe": int(n_maybe),
            "n_no_go": int(n_no_go),
            "go_rate": n_go / len(decisions),
            "maybe_rate": n_maybe / len(decisions),
            "no_go_rate": n_no_go / len(decisions),
            "decisions": decisions.tolist(),
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
        }
    
    def _save_artifacts(
        self,
        al_results: dict,
        go_no_go_results: Optional[dict],
        filter_stats: dict,
    ):
        """Save pipeline artifacts."""
        # Save AL history
        with open(self.artifacts_dir / "al_history.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = []
            for item in al_results["history"]:
                history_serializable.append({
                    "iteration": int(item["iteration"]),
                    "n_labeled": int(item["n_labeled"]),
                    "n_pool": int(item["n_pool"]),
                    "budget_used": int(item["budget_used"]),
                    "acquisition_scores": item["acquisition_scores"].tolist(),
                })
            
            json.dump(history_serializable, f, indent=2)
        
        # Save summary
        summary = {
            "config": {
                "acquisition_method": self.acquisition_method,
                "acquisition_kwargs": self.acquisition_kwargs,
                "diversity_method": self.diversity_method,
                "diversity_kwargs": self.diversity_kwargs,
                "ood_method": self.ood_method,
                "ood_kwargs": self.ood_kwargs,
                "budget": self.budget,
                "batch_size": self.batch_size,
                "random_state": self.random_state,
            },
            "results": {
                "ood_filtering": filter_stats,
                "budget_used": al_results["budget_used"],
                "n_iterations": len(al_results["history"]),
                "final_labeled_size": len(al_results["X_train"]),
                "go_no_go": go_no_go_results,
            },
        }
        
        with open(self.artifacts_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save final model
        self.base_model.save(self.artifacts_dir / "model_final.pkl")

