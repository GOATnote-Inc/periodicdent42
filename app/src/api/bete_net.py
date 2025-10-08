"""
FastAPI endpoints for BETE-NET superconductor screening.

Endpoints:
- POST /api/bete/predict - Single structure prediction
- POST /api/bete/screen - Batch screening with streaming
- GET /api/bete/report/{run_id} - Download evidence pack

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.bete_net_io.batch import ScreeningConfig, batch_screen
from src.bete_net_io.evidence import create_evidence_pack
from src.bete_net_io.inference import predict_tc

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bete", tags=["BETE-NET"])

# Evidence packs storage
EVIDENCE_DIR = Path("/tmp/bete_evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


class PredictRequest(BaseModel):
    """Request for single structure prediction."""

    cif_content: Optional[str] = Field(None, description="CIF file content")
    mp_id: Optional[str] = Field(None, description="Materials Project ID (e.g., mp-48)")
    mu_star: float = Field(0.10, description="Coulomb pseudopotential (0.10-0.13)")

    def validate_input(self):
        if not self.cif_content and not self.mp_id:
            raise ValueError("Must provide either cif_content or mp_id")
        if self.cif_content and self.mp_id:
            raise ValueError("Provide only one of cif_content or mp_id")


class PredictResponse(BaseModel):
    """Response from single structure prediction."""

    formula: str
    mp_id: Optional[str]
    tc_kelvin: float
    tc_std: float
    lambda_ep: float
    lambda_std: float
    omega_log_K: float
    omega_log_std_K: float
    mu_star: float
    input_hash: str
    evidence_url: str
    timestamp: str


class ScreenRequest(BaseModel):
    """Request for batch screening."""

    mp_ids: Optional[List[str]] = Field(None, description="List of MP-IDs")
    cif_contents: Optional[List[str]] = Field(None, description="List of CIF contents")
    mu_star: float = Field(0.10, description="Coulomb pseudopotential")
    n_workers: int = Field(4, description="Number of parallel workers")


class ScreenResponse(BaseModel):
    """Response from batch screening."""

    run_id: str
    n_materials: int
    status: str
    results_url: Optional[str] = None


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    Predict superconducting Tc for a single crystal structure.

    **Example**:
    ```bash
    curl -X POST http://localhost:8080/api/bete/predict \\
      -H "Content-Type: application/json" \\
      -d '{"mp_id": "mp-48", "mu_star": 0.10}'
    ```

    **Response**:
    ```json
    {
      "formula": "Nb",
      "mp_id": "mp-48",
      "tc_kelvin": 9.2,
      "tc_std": 1.4,
      "lambda_ep": 1.05,
      "evidence_url": "/api/bete/report/abc123..."
    }
    ```
    """
    try:
        request.validate_input()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # Determine input
        input_id = request.mp_id if request.mp_id else "uploaded.cif"

        if request.cif_content:
            # Save temporary CIF
            temp_cif = EVIDENCE_DIR / f"temp_{uuid.uuid4()}.cif"
            temp_cif.write_text(request.cif_content)
            input_id = str(temp_cif)

        # Run prediction
        logger.info(f"Predicting Tc for {input_id} (μ*={request.mu_star:.3f})")
        prediction = predict_tc(input_id, mu_star=request.mu_star)

        # Create evidence pack
        evidence_path = create_evidence_pack(
            prediction, EVIDENCE_DIR, cif_content=request.cif_content
        )

        # Clean up temp CIF
        if request.cif_content:
            temp_cif.unlink(missing_ok=True)

        return PredictResponse(
            formula=prediction.formula,
            mp_id=prediction.mp_id,
            tc_kelvin=prediction.tc_kelvin,
            tc_std=prediction.tc_std,
            lambda_ep=prediction.lambda_ep,
            lambda_std=prediction.lambda_std,
            omega_log_K=prediction.omega_log,
            omega_log_std_K=prediction.omega_log_std,
            mu_star=prediction.mu_star,
            input_hash=prediction.input_hash,
            evidence_url=f"/api/bete/report/{evidence_path.stem}",
            timestamp=prediction.timestamp,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/screen", response_model=ScreenResponse)
async def screen_endpoint(request: ScreenRequest, background_tasks: BackgroundTasks):
    """
    Screen a batch of candidate superconductors.

    **Example**:
    ```bash
    curl -X POST http://localhost:8080/api/bete/screen \\
      -H "Content-Type: application/json" \\
      -d '{
        "mp_ids": ["mp-48", "mp-66", "mp-134"],
        "mu_star": 0.13,
        "n_workers": 8
      }'
    ```

    **Response**:
    ```json
    {
      "run_id": "550e8400-e29b-41d4-a716-446655440000",
      "n_materials": 3,
      "status": "queued",
      "results_url": "/api/bete/report/550e8400..."
    }
    ```
    """
    if not request.mp_ids and not request.cif_contents:
        raise HTTPException(
            status_code=400, detail="Must provide either mp_ids or cif_contents"
        )

    run_id = str(uuid.uuid4())
    inputs = request.mp_ids or request.cif_contents

    logger.info(f"Starting batch screening: run_id={run_id}, n_materials={len(inputs)}")

    # Run screening in background
    def run_screening():
        try:
            output_path = EVIDENCE_DIR / f"{run_id}_results.parquet"
            config = ScreeningConfig(
                inputs=inputs,
                mu_star=request.mu_star,
                output_path=output_path,
                n_workers=request.n_workers,
            )
            batch_screen(config)
            logger.info(f"Screening complete: {run_id}")
        except Exception as e:
            logger.error(f"Screening failed: {run_id} - {e}", exc_info=True)

    background_tasks.add_task(run_screening)

    return ScreenResponse(
        run_id=run_id,
        n_materials=len(inputs),
        status="queued",
        results_url=f"/api/bete/report/{run_id}",
    )


@router.get("/report/{report_id}")
async def get_report(report_id: str):
    """
    Download evidence pack for a completed prediction or screening run.

    **Example**:
    ```bash
    curl http://localhost:8080/api/bete/report/abc123... -o evidence.zip
    ```
    """
    # Try ZIP file first
    zip_path = EVIDENCE_DIR / f"{report_id}.zip"
    if zip_path.exists():
        return FileResponse(
            zip_path, media_type="application/zip", filename=f"evidence_{report_id}.zip"
        )

    # Try Parquet results
    parquet_path = EVIDENCE_DIR / f"{report_id}_results.parquet"
    if parquet_path.exists():
        return FileResponse(
            parquet_path,
            media_type="application/octet-stream",
            filename=f"screening_{report_id}.parquet",
        )

    raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")

