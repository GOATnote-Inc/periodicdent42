"""
Integrated Experiment Runner with HTC Support

Extends experiment orchestration with high-temperature superconductor
optimization capabilities, integrating:
- Statistical analysis framework
- Uncertainty quantification (ISO GUM + Sobol)
- HTC domain module for superconductor discovery

New HTC experiment types:
- HTC_screening: Screen candidate materials
- HTC_optimization: Multi-objective optimization
- HTC_validation: Validate against known materials

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.src.htc.domain import (
    SuperconductorPrediction,
    SuperconductorPredictor,
    XiConstraintValidator,
    compute_pareto_front,
    load_benchmark_materials,
    predict_tc_with_uncertainty,
    validate_against_known_materials,
)

# Optional git support
try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    warnings.warn("GitPython not available - provenance tracking limited")

logger = logging.getLogger(__name__)


class IntegratedExperimentRunner:
    """
    Unified experiment runner supporting both original experiments and HTC.

    Supported experiment types:
    - E1: Correlation strength analysis (original)
    - HTC_screening: Screen superconductor candidates
    - HTC_optimization: Multi-objective optimization
    - HTC_validation: Validate against known materials
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        evidence_dir: Path = Path("evidence"),
        results_dir: Path = Path("results"),
    ):
        """
        Parameters
        ----------
        config_path : Path, optional
            Path to protocol_preregistration.yaml
        evidence_dir : Path
            Directory for raw results with checksums
        results_dir : Path
            Directory for human-readable summaries
        """
        self.config_path = Path(config_path) if config_path else None
        self.evidence_dir = Path(evidence_dir)
        self.results_dir = Path(results_dir)

        # Create directories
        self.evidence_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Load configuration
        if self.config_path and self.config_path.exists():
            self.config = self._load_config()
        else:
            self.config = self._get_default_config()

        # Git provenance
        if GIT_AVAILABLE:
            try:
                repo = git.Repo(search_parent_directories=True)
                self.git_sha = repo.head.object.hexsha
                self.git_dirty = repo.is_dirty()
            except Exception as e:
                logger.warning(f"Git repository not available: {e}")
                self.git_sha = "unknown"
                self.git_dirty = False
        else:
            self.git_sha = "unknown"
            self.git_dirty = False

        # Initialize HTC components
        self.htc_predictor = SuperconductorPredictor(random_state=42)
        self.xi_validator = XiConstraintValidator(threshold=4.0)

        logger.info(
            f"Initialized IntegratedExperimentRunner (git: {self.git_sha[:8]}, "
            f"dirty: {self.git_dirty})"
        )

    def _load_config(self) -> dict[str, Any]:
        """Load YAML configuration"""
        try:
            import yaml

            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed - using default config")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Default configuration"""
        return {
            "global_settings": {"random_seed": 42, "confidence_level": 0.95},
            "HTC_optimization": {
                "max_pressure_gpa": 1.0,
                "min_tc_kelvin": 77.0,
                "xi_threshold": 4.0,
                "success_criteria": {
                    "pareto_front_size": 5,
                    "best_tc_above_target": 100.0,
                    "constraint_satisfaction_rate": 0.90,
                    "validation_error_threshold": 20.0,
                },
            },
        }

    def run_experiment(
        self, experiment_id: str, data: Optional[Any] = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute experiment with full tracking and provenance.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier (e.g., 'E1', 'HTC_screening')
        data : any, optional
            Experiment-specific data
        **kwargs
            Additional experiment parameters

        Returns
        -------
        results : dict
            Complete results with metadata and provenance
        """
        logger.info(f"Running experiment: {experiment_id}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Git SHA: {self.git_sha[:8]} (dirty: {self.git_dirty})")

        # Route to appropriate experiment
        if experiment_id.startswith("HTC"):
            if "screening" in experiment_id.lower():
                results = self.run_htc_screening(data, **kwargs)
            elif "optimization" in experiment_id.lower():
                results = self.run_htc_optimization(data, **kwargs)
            elif "validation" in experiment_id.lower():
                results = self.run_htc_validation(**kwargs)
            else:
                raise ValueError(f"Unknown HTC experiment type: {experiment_id}")
        else:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        # Standard post-processing
        return self._finalize_results(experiment_id, results)

    # =========================================================================
    # HTC EXPERIMENTS
    # =========================================================================

    def run_htc_screening(
        self, candidate_materials: Optional[list[dict]] = None, **kwargs
    ) -> dict[str, Any]:
        """
        Screen candidate materials for superconductivity.

        Parameters
        ----------
        candidate_materials : list of dict, optional
            Each dict: {'composition': str, 'structure': Structure, 'pressure': float}
        **kwargs
            max_pressure_gpa : float
            min_tc_kelvin : float

        Returns
        -------
        results : dict
            Screening results with predictions and statistics
        """
        logger.info("HTC SCREENING EXPERIMENT")

        # Get parameters
        exp_config = self.config.get("HTC_optimization", {})
        max_pressure = kwargs.get("max_pressure_gpa", exp_config.get("max_pressure_gpa", 1.0))
        min_tc = kwargs.get("min_tc_kelvin", exp_config.get("min_tc_kelvin", 77.0))
        random_state = kwargs.get(
            "random_state", self.config["global_settings"]["random_seed"]
        )

        logger.info(f"Parameters: max_pressure={max_pressure} GPa, min_tc={min_tc} K")

        # Load materials
        if candidate_materials is None:
            logger.info("Using benchmark materials...")
            candidate_materials = load_benchmark_materials(include_ambient=True)

        logger.info(f"Screening {len(candidate_materials)} materials...")

        # Run predictions
        predictions = []
        for i, candidate in enumerate(candidate_materials):
            comp = candidate.get("composition", "Unknown")
            logger.info(f"[{i+1}/{len(candidate_materials)}] Evaluating {comp}...")

            pred = predict_tc_with_uncertainty(
                structure=candidate["structure"],
                pressure_gpa=candidate.get("pressure", 0.0),
                random_state=random_state,
            )
            predictions.append(pred)

            logger.info(
                f"  Tc = {pred.tc_predicted:.1f} K [{pred.tc_lower_95ci:.1f}, "
                f"{pred.tc_upper_95ci:.1f}]"
            )

        # Filter candidates meeting criteria
        passing_candidates = [
            p
            for p in predictions
            if p.satisfies_constraints(max_pressure, min_tc, exp_config.get("xi_threshold", 4.0))
        ]

        success_rate = len(passing_candidates) / len(predictions) if predictions else 0

        logger.info(f"Screening complete: {len(passing_candidates)}/{len(predictions)} passed")
        logger.info(f"Success rate: {success_rate*100:.1f}%")

        # Statistical summary
        statistical_summary = self._compute_htc_statistics(predictions, exp_config)

        return {
            "experiment_type": "HTC_screening",
            "predictions": [pred.to_dict() for pred in predictions],
            "passing_candidates": [pred.to_dict() for pred in passing_candidates],
            "statistical_summary": statistical_summary,
            "metadata": {
                "n_candidates": len(candidate_materials),
                "n_passing": len(passing_candidates),
                "success_rate": success_rate,
                "parameters": {"max_pressure_gpa": max_pressure, "min_tc_kelvin": min_tc},
            },
        }

    def run_htc_optimization(
        self, candidate_materials: Optional[list[dict]] = None, **kwargs
    ) -> dict[str, Any]:
        """
        Multi-objective optimization: maximize Tc, minimize pressure.

        Parameters
        ----------
        candidate_materials : list of dict, optional
            Starting materials for optimization
        **kwargs
            Optimization parameters

        Returns
        -------
        results : dict
            Optimization results with Pareto front
        """
        logger.info("HTC MULTI-OBJECTIVE OPTIMIZATION")

        # Get configuration
        exp_config = self.config.get("HTC_optimization", {})
        random_state = kwargs.get(
            "random_state", self.config["global_settings"]["random_seed"]
        )

        # Run screening first
        screening_results = self.run_htc_screening(candidate_materials, **kwargs)
        predictions = [SuperconductorPrediction(**p) for p in screening_results["predictions"]]

        logger.info("Computing Pareto front...")

        # Compute Pareto front
        pareto_front = compute_pareto_front(
            predictions,
            objectives=["tc_predicted", "pressure_required_gpa"],
            directions=["maximize", "minimize"],
        )

        logger.info(f"Pareto-optimal materials: {len(pareto_front)}")

        # Validation against known materials
        validation_results = validate_against_known_materials(predictions)

        logger.info("Validation against known materials:")
        for material, error in validation_results.items():
            threshold = exp_config["success_criteria"]["validation_error_threshold"]
            status = "✓" if error < threshold else "✗"
            logger.info(f"  {status} {material}: MAE = {error:.1f} K")

        # Pre-registration compliance
        compliance = self._check_htc_compliance(
            predictions, pareto_front, validation_results, exp_config
        )

        return {
            "experiment_type": "HTC_optimization",
            "predictions": [pred.to_dict() for pred in predictions],
            "pareto_front": [pred.to_dict() for pred in pareto_front],
            "validation_results": validation_results,
            "compliance": compliance,
            "statistical_summary": screening_results["statistical_summary"],
            "metadata": {
                "n_evaluated": len(predictions),
                "n_pareto_optimal": len(pareto_front),
                "random_seed": random_state,
            },
        }

    def run_htc_validation(self, **kwargs) -> dict[str, Any]:
        """
        Validate HTC predictor against known superconductors.

        Returns
        -------
        results : dict
            Validation metrics and statistical tests
        """
        logger.info("HTC VALIDATION EXPERIMENT")

        # Load known materials
        materials = load_benchmark_materials(include_ambient=True)

        if not materials:
            raise RuntimeError("No benchmark materials available")

        logger.info(f"Validating against {len(materials)} known superconductors...")

        # Make predictions
        predictions = []
        for mat in materials:
            pred = predict_tc_with_uncertainty(
                structure=mat["structure"], pressure_gpa=mat["pressure"], random_state=42
            )
            predictions.append(pred)

        # Validate
        validation_errors = validate_against_known_materials(predictions)

        logger.info("Validation Results:")
        for material, error in validation_errors.items():
            logger.info(f"  {material}: MAE = {error:.1f} K")

        mean_error = np.mean(list(validation_errors.values())) if validation_errors else 0
        max_error = np.max(list(validation_errors.values())) if validation_errors else 0
        within_20k = sum(e < 20.0 for e in validation_errors.values())

        return {
            "experiment_type": "HTC_validation",
            "validation_errors": validation_errors,
            "predictions": [pred.to_dict() for pred in predictions],
            "summary": {
                "mean_error": float(mean_error),
                "max_error": float(max_error),
                "materials_within_20K": within_20k,
                "total_materials": len(validation_errors),
            },
        }

    # =========================================================================
    # ANALYSIS HELPERS
    # =========================================================================

    def _compute_htc_statistics(
        self, predictions: list[SuperconductorPrediction], exp_config: dict
    ) -> dict[str, Any]:
        """Compute statistical summary for HTC predictions"""
        if not predictions:
            return {}

        tc_values = np.array([p.tc_predicted for p in predictions])
        tc_uncertainties = np.array([p.tc_uncertainty for p in predictions])
        pressures = np.array([p.pressure_required_gpa for p in predictions])
        xi_values = np.array([p.xi_parameter for p in predictions])

        # Stability fractions
        phonon_stable_frac = np.mean([p.phonon_stable for p in predictions])
        thermo_stable_frac = np.mean([p.thermo_stable for p in predictions])

        return {
            "tc_statistics": {
                "mean": float(np.mean(tc_values)),
                "median": float(np.median(tc_values)),
                "std": float(np.std(tc_values)),
                "min": float(np.min(tc_values)),
                "max": float(np.max(tc_values)),
                "mean_uncertainty": float(np.mean(tc_uncertainties)),
            },
            "pressure_statistics": {
                "mean": float(np.mean(pressures)),
                "median": float(np.median(pressures)),
                "below_1_gpa_fraction": float(np.mean(pressures < 1.0)),
                "below_10_gpa_fraction": float(np.mean(pressures < 10.0)),
            },
            "xi_statistics": {
                "mean": float(np.mean(xi_values)),
                "max": float(np.max(xi_values)),
                "violation_rate": float(np.mean(xi_values > 4.0)),
                "safe_margin": float(4.0 - np.max(xi_values)),
            },
            "stability_fractions": {
                "phonon_stable": float(phonon_stable_frac),
                "thermo_stable": float(thermo_stable_frac),
                "both_stable": float(
                    np.mean([p.phonon_stable and p.thermo_stable for p in predictions])
                ),
            },
        }

    def _check_htc_compliance(
        self,
        predictions: list[SuperconductorPrediction],
        pareto_front: list[SuperconductorPrediction],
        validation_results: dict[str, float],
        exp_config: dict,
    ) -> dict[str, Any]:
        """Check pre-registration compliance"""
        criteria = exp_config.get("success_criteria", {})
        checks = []

        # Check 1: ξ bound violation rate
        xi_values = np.array([p.xi_parameter for p in predictions])
        violation_rate = np.mean(xi_values > 4.0)
        expected_max = 1.0 - criteria.get("constraint_satisfaction_rate", 0.90)

        checks.append(
            {
                "test": "xi_constraint_satisfaction",
                "passed": violation_rate <= expected_max,
                "value": float(1.0 - violation_rate),
                "threshold": criteria.get("constraint_satisfaction_rate", 0.90),
                "message": f"ξ ≤ 4.0 satisfaction rate: {(1.0-violation_rate)*100:.1f}%",
            }
        )

        # Check 2: Pareto front size
        pareto_size = len(pareto_front)
        min_pareto_size = criteria.get("pareto_front_size", 5)

        checks.append(
            {
                "test": "pareto_front_size",
                "passed": pareto_size >= min_pareto_size,
                "value": pareto_size,
                "threshold": min_pareto_size,
                "message": f"Pareto front size: {pareto_size}",
            }
        )

        # Check 3: Best Tc above target
        stable_materials = [p for p in predictions if p.phonon_stable and p.thermo_stable]
        if stable_materials:
            best_tc = max(p.tc_predicted for p in stable_materials)
            tc_target = criteria.get("best_tc_above_target", 100.0)

            checks.append(
                {
                    "test": "best_tc_target",
                    "passed": best_tc >= tc_target,
                    "value": float(best_tc),
                    "threshold": tc_target,
                    "message": f"Best stable Tc: {best_tc:.1f} K",
                }
            )

        # Check 4: Validation errors
        if validation_results:
            mean_error = np.mean(list(validation_results.values()))
            error_threshold = criteria.get("validation_error_threshold", 20.0)

            checks.append(
                {
                    "test": "validation_accuracy",
                    "passed": mean_error < error_threshold,
                    "value": float(mean_error),
                    "threshold": error_threshold,
                    "message": f"Mean validation error: {mean_error:.1f} K",
                }
            )

        fully_compliant = all(check["passed"] for check in checks)

        return {
            "fully_compliant": fully_compliant,
            "checks": checks,
            "summary": (
                f"{'✓' if fully_compliant else '✗'} "
                f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed"
            ),
        }

    # =========================================================================
    # FINALIZATION
    # =========================================================================

    def _finalize_results(self, experiment_id: str, results: dict[str, Any]) -> dict[str, Any]:
        """Save results with provenance and checksums"""
        timestamp = datetime.now().isoformat()

        # Add metadata
        results["metadata"] = results.get("metadata", {})
        results["metadata"].update(
            {
                "experiment_id": experiment_id,
                "timestamp": timestamp,
                "git_sha": self.git_sha,
                "git_dirty": self.git_dirty,
            }
        )

        # Convert to JSON
        results_json = json.dumps(results, default=str, indent=2)

        # Calculate checksum
        checksum = hashlib.sha256(results_json.encode()).hexdigest()
        results["metadata"]["checksum"] = checksum

        # Save evidence (raw JSON)
        evidence_path = self.evidence_dir / f"{experiment_id}_{timestamp}.json"
        try:
            with open(evidence_path, "w") as f:
                json.dump(results, f, default=str, indent=2)
            logger.info(f"Evidence saved: {evidence_path}")
        except Exception as e:
            logger.error(f"Failed to save evidence: {e}")

        # Save human-readable summary
        summary_path = self.results_dir / f"{experiment_id}_{timestamp}_summary.txt"
        try:
            with open(summary_path, "w") as f:
                f.write(self._generate_summary_text(experiment_id, results))
            logger.info(f"Summary saved: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

        logger.info(f"Results checksum: {checksum[:16]}...")

        return results

    def _generate_summary_text(self, experiment_id: str, results: dict[str, Any]) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 70,
            f"EXPERIMENT: {experiment_id}",
            f"Timestamp: {results['metadata']['timestamp']}",
            f"Git SHA: {results['metadata']['git_sha'][:8]}",
            "=" * 70,
            "",
        ]

        # Experiment-specific summaries
        if "statistical_summary" in results:
            lines.append("STATISTICAL SUMMARY")
            lines.append("-" * 70)
            stats = results["statistical_summary"]
            if "tc_statistics" in stats:
                tc = stats["tc_statistics"]
                lines.append(
                    f"Tc (K): {tc['mean']:.1f} ± {tc['std']:.1f} "
                    f"[{tc['min']:.1f}, {tc['max']:.1f}]"
                )
            lines.append("")

        if "compliance" in results:
            lines.append("PRE-REGISTRATION COMPLIANCE")
            lines.append("-" * 70)
            comp = results["compliance"]
            lines.append(comp["summary"])
            for check in comp["checks"]:
                status = "✓" if check["passed"] else "✗"
                lines.append(f"  {status} {check['test']}: {check['message']}")
            lines.append("")

        return "\n".join(lines)

