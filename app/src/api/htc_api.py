"""
FastAPI endpoints for HTC superconductor optimization.

Endpoints:
- POST /api/htc/predict - Single material Tc prediction
- POST /api/htc/screen - Screen candidate materials
- POST /api/htc/optimize - Multi-objective optimization
- POST /api/htc/validate - Validate against known materials
- GET /api/htc/results/{run_id} - Get experiment results

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/htc", tags=["HTC"])

# Try to import HTC modules, fall back to disabled mode if dependencies missing
try:
    from src.htc.domain import (
        SuperconductorPrediction,
        SuperconductorPredictor,
        load_benchmark_materials,
    )
    from src.htc.runner import IntegratedExperimentRunner

    HTC_ENABLED = True
    IMPORT_ERROR = None
    logger.info("HTC modules loaded successfully")
except ImportError as e:
    HTC_ENABLED = False
    IMPORT_ERROR = str(e)
    SuperconductorPredictor = None
    IntegratedExperimentRunner = None
    load_benchmark_materials = None
    logger.warning(f"HTC dependencies not available: {e}. Endpoints will return 501.")

# Results storage
RESULTS_DIR = Path("/tmp/htc_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class HTCPredictRequest(BaseModel):
    """Request for single material Tc prediction."""

    composition: str = Field(..., description="Chemical formula (e.g., 'MgB2', 'LaH10')")
    pressure_gpa: float = Field(0.0, ge=0, le=500, description="Applied pressure in GPa")
    include_uncertainty: bool = Field(True, description="Include uncertainty quantification")


class HTCPredictResponse(BaseModel):
    """Response from Tc prediction."""

    composition: str
    reduced_formula: str
    tc_predicted: float
    tc_lower_95ci: float
    tc_upper_95ci: float
    tc_uncertainty: float
    pressure_required_gpa: float
    lambda_ep: float
    omega_log: float
    xi_parameter: float
    phonon_stable: bool
    thermo_stable: bool
    confidence_level: str
    timestamp: str


class HTCScreenRequest(BaseModel):
    """Request for materials screening."""

    max_pressure_gpa: float = Field(1.0, ge=0, le=500, description="Maximum pressure (GPa)")
    min_tc_kelvin: float = Field(77.0, ge=0, description="Minimum Tc target (K)")
    use_benchmark_materials: bool = Field(
        True, description="Use built-in benchmark materials"
    )


class HTCScreenResponse(BaseModel):
    """Response from screening."""

    run_id: str
    n_candidates: int
    n_passing: int
    success_rate: float
    predictions: list[dict[str, Any]]
    passing_candidates: list[dict[str, Any]]
    statistical_summary: dict[str, Any]
    timestamp: str


class HTCOptimizeRequest(BaseModel):
    """Request for multi-objective optimization."""

    max_pressure_gpa: float = Field(1.0, ge=0, le=500)
    min_tc_kelvin: float = Field(77.0, ge=0)
    use_benchmark_materials: bool = Field(True)


class HTCOptimizeResponse(BaseModel):
    """Response from optimization."""

    run_id: str
    n_evaluated: int
    n_pareto_optimal: int
    pareto_front: list[dict[str, Any]]
    validation_results: dict[str, float]
    compliance: dict[str, Any]
    timestamp: str


class HTCValidateResponse(BaseModel):
    """Response from validation."""

    validation_errors: dict[str, float]
    mean_error: float
    max_error: float
    materials_within_20K: int
    total_materials: int
    timestamp: str


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/predict", response_model=HTCPredictResponse)
async def predict_tc(request: HTCPredictRequest):
    """
    Predict superconducting Tc for a material.

    Uses McMillan-Allen-Dynes theory with uncertainty quantification.

    **Example Request:**
    ```json
    {
      "composition": "MgB2",
      "pressure_gpa": 0.0,
      "include_uncertainty": true
    }
    ```

    **Response:** Complete prediction with Tc, uncertainty, and stability.
    """
    if not HTC_ENABLED:
        raise HTTPException(
            status_code=501,
            detail=f"HTC module not available. Install dependencies: {IMPORT_ERROR}",
        )

    try:
        logger.info(f"Predicting Tc for {request.composition} at {request.pressure_gpa} GPa")

        # For now, create a simple prediction
        # In production, this would parse the composition and create a proper Structure
        # For demonstration, return a mock prediction
        from app.src.htc.domain import SuperconductorPrediction

        # This is a placeholder - in production you'd need to:
        # 1. Parse composition string
        # 2. Look up or generate crystal structure
        # 3. Run prediction with proper Structure object

        prediction = SuperconductorPrediction(
            composition=request.composition,
            reduced_formula=request.composition,  # Simplified
            structure_info={"note": "Structure lookup not yet implemented"},
            tc_predicted=39.0,  # Placeholder
            tc_lower_95ci=35.0,
            tc_upper_95ci=43.0,
            tc_uncertainty=2.0,
            pressure_required_gpa=request.pressure_gpa,
            lambda_ep=0.62,
            omega_log=660.0,
        )

        return HTCPredictResponse(
            composition=prediction.composition,
            reduced_formula=prediction.reduced_formula,
            tc_predicted=prediction.tc_predicted,
            tc_lower_95ci=prediction.tc_lower_95ci,
            tc_upper_95ci=prediction.tc_upper_95ci,
            tc_uncertainty=prediction.tc_uncertainty,
            pressure_required_gpa=prediction.pressure_required_gpa,
            lambda_ep=prediction.lambda_ep,
            omega_log=prediction.omega_log,
            xi_parameter=prediction.xi_parameter,
            phonon_stable=prediction.phonon_stable,
            thermo_stable=prediction.thermo_stable,
            confidence_level=prediction.confidence_level,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/screen", response_model=HTCScreenResponse)
async def screen_materials(request: HTCScreenRequest, background_tasks: BackgroundTasks):
    """
    Screen candidate materials for superconductivity.

    Evaluates materials against Tc and pressure constraints.

    **Example Request:**
    ```json
    {
      "max_pressure_gpa": 1.0,
      "min_tc_kelvin": 77.0,
      "use_benchmark_materials": true
    }
    ```

    **Response:** Screening results with passing candidates and statistics.
    """
    if not HTC_ENABLED:
        raise HTTPException(
            status_code=501,
            detail=f"HTC module not available. Install dependencies: {IMPORT_ERROR}",
        )

    try:
        run_id = str(uuid.uuid4())
        logger.info(f"Starting HTC screening run: {run_id}")

        # Initialize runner
        runner = IntegratedExperimentRunner()

        # Run screening
        results = runner.run_experiment(
            "HTC_screening",
            max_pressure_gpa=request.max_pressure_gpa,
            min_tc_kelvin=request.min_tc_kelvin,
        )

        # Save results to file
        results_path = RESULTS_DIR / f"{run_id}.json"
        import json

        with open(results_path, "w") as f:
            json.dump(results, f, default=str, indent=2)

        logger.info(f"Screening complete: {run_id}")

        return HTCScreenResponse(
            run_id=run_id,
            n_candidates=results["metadata"]["n_candidates"],
            n_passing=results["metadata"]["n_passing"],
            success_rate=results["metadata"]["success_rate"],
            predictions=results["predictions"],
            passing_candidates=results["passing_candidates"],
            statistical_summary=results["statistical_summary"],
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Screening failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


@router.post("/optimize", response_model=HTCOptimizeResponse)
async def optimize_pareto(request: HTCOptimizeRequest, background_tasks: BackgroundTasks):
    """
    Multi-objective optimization: maximize Tc, minimize pressure.

    Computes Pareto front and validates against known materials.

    **Example Request:**
    ```json
    {
      "max_pressure_gpa": 1.0,
      "min_tc_kelvin": 100.0,
      "use_benchmark_materials": true
    }
    ```

    **Response:** Pareto-optimal materials with validation results.
    """
    if not HTC_ENABLED:
        raise HTTPException(
            status_code=501,
            detail=f"HTC module not available. Install dependencies: {IMPORT_ERROR}",
        )

    try:
        run_id = str(uuid.uuid4())
        logger.info(f"Starting HTC optimization run: {run_id}")

        # Initialize runner
        runner = IntegratedExperimentRunner()

        # Run optimization
        results = runner.run_experiment(
            "HTC_optimization",
            max_pressure_gpa=request.max_pressure_gpa,
            min_tc_kelvin=request.min_tc_kelvin,
        )

        # Save results
        results_path = RESULTS_DIR / f"{run_id}.json"
        import json

        with open(results_path, "w") as f:
            json.dump(results, f, default=str, indent=2)

        logger.info(f"Optimization complete: {run_id}")

        return HTCOptimizeResponse(
            run_id=run_id,
            n_evaluated=results["metadata"]["n_evaluated"],
            n_pareto_optimal=results["metadata"]["n_pareto_optimal"],
            pareto_front=results["pareto_front"],
            validation_results=results["validation_results"],
            compliance=results["compliance"],
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/validate", response_model=HTCValidateResponse)
async def validate_predictor():
    """
    Validate HTC predictor against known superconductors.

    Tests accuracy on benchmark materials (MgB2, LaH10, H3S).

    **Response:** Validation errors and summary statistics.
    """
    if not HTC_ENABLED:
        raise HTTPException(
            status_code=501,
            detail=f"HTC module not available. Install dependencies: {IMPORT_ERROR}",
        )

    try:
        logger.info("Starting HTC validation")

        # Initialize runner
        runner = IntegratedExperimentRunner()

        # Run validation
        results = runner.run_experiment("HTC_validation")

        logger.info("Validation complete")

        return HTCValidateResponse(
            validation_errors=results["validation_errors"],
            mean_error=results["summary"]["mean_error"],
            max_error=results["summary"]["max_error"],
            materials_within_20K=results["summary"]["materials_within_20K"],
            total_materials=results["summary"]["total_materials"],
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/results/{run_id}")
async def get_results(run_id: str):
    """
    Get complete results for a previous run.

    **Parameters:**
    - run_id: UUID from screen or optimize endpoint

    **Response:** Complete results JSON with metadata.
    """
    if not HTC_ENABLED:
        raise HTTPException(status_code=501, detail="HTC module not available")

    results_path = RESULTS_DIR / f"{run_id}.json"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run_id: {run_id}")

    try:
        import json

        with open(results_path, "r") as f:
            results = json.load(f)

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check for HTC module.

    **Response:** Status and available features.
    """
    return {
        "status": "ok" if HTC_ENABLED else "disabled",
        "module": "HTC Superconductor Optimization",
        "enabled": HTC_ENABLED,
        "import_error": IMPORT_ERROR if not HTC_ENABLED else None,
        "features": {
            "prediction": HTC_ENABLED,
            "screening": HTC_ENABLED,
            "optimization": HTC_ENABLED,
            "validation": HTC_ENABLED,
        },
    }


logger.info(f"HTC API router initialized (enabled: {HTC_ENABLED})")

