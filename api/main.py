"""
FastAPI REST API for matprov Experiment Tracking

Exposes the prediction registry and experiment tracking as a web service.

Endpoints:
- GET /health - Health check
- GET /api/models - List registered models
- GET /api/predictions - List predictions (with filters)
- POST /api/predictions - Create new prediction
- GET /api/predictions/{id} - Get prediction details
- GET /api/experiments - List experiments
- POST /api/experiments - Create experiment outcome
- GET /api/experiments/{id} - Get experiment details
- GET /api/performance/{model_id} - Get model performance
- GET /api/candidates - Get top candidates for validation
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matprov.registry.database import Database
from matprov.registry.models import Model, Prediction, ExperimentOutcome, PredictionError
from matprov.registry.queries import PredictionQueries
from sqlalchemy import select

# Initialize FastAPI
app = FastAPI(
    title="matprov API",
    description="Materials Provenance Tracking REST API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    from .database import init_db
    init_db()
    print("✅ API startup complete - database initialized")

# Include authentication router
from .routers.auth import router as auth_router
app.include_router(auth_router)

# Database dependency
def get_db():
    """Get database session"""
    db = Database()
    return db


# Pydantic models for API
class ModelCreate(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    version: str = Field(..., description="Model version")
    checkpoint_hash: str = Field(..., description="DVC hash of model checkpoint")
    training_dataset_hash: str = Field(..., description="DVC hash of training dataset")
    architecture: Optional[str] = Field(None, description="Model architecture")
    hyperparameters: Optional[str] = Field(None, description="JSON string of hyperparameters")
    notes: Optional[str] = None


class ModelResponse(BaseModel):
    id: int
    model_id: str
    version: str
    checkpoint_hash: str
    training_dataset_hash: str
    architecture: Optional[str]
    training_date: datetime
    notes: Optional[str]


class PredictionCreate(BaseModel):
    prediction_id: str
    model_id: str
    material_formula: str
    predicted_tc: float
    uncertainty: Optional[float] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None


class PredictionResponse(BaseModel):
    id: int
    prediction_id: str
    material_formula: str
    predicted_tc: float
    uncertainty: Optional[float]
    predicted_class: Optional[str]
    confidence: Optional[float]
    prediction_date: datetime
    model_id: str
    model_version: str


class ExperimentOutcomeCreate(BaseModel):
    prediction_id: str
    experiment_id: str
    actual_tc: float
    validation_status: str = "success"
    phase_purity: Optional[float] = None
    notes: Optional[str] = None


class ExperimentResponse(BaseModel):
    experiment_id: str
    prediction_id: str
    material_formula: str
    predicted_tc: float
    actual_tc: float
    validation_status: str
    phase_purity: Optional[float]
    absolute_error: float
    experiment_date: datetime


class PerformanceResponse(BaseModel):
    model_id: str
    version: str
    total_predictions: int
    validated_predictions: int
    mae: float
    rmse: float
    r2_score: float
    classification_metrics: Optional[Dict[str, float]]


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "matprov-api", "version": "0.1.0"}


@app.get("/api/models", response_model=List[ModelResponse])
async def list_models(
    limit: int = Query(10, ge=1, le=100),
    db: Database = Depends(get_db)
):
    """List all registered models"""
    with db.session() as session:
        models = session.execute(
            select(Model).order_by(Model.training_date.desc()).limit(limit)
        ).scalars().all()
        
        return [
            ModelResponse(
                id=m.id,
                model_id=m.model_id,
                version=m.version,
                checkpoint_hash=m.checkpoint_hash,
                training_dataset_hash=m.training_dataset_hash,
                architecture=m.architecture,
                training_date=m.training_date,
                notes=m.notes
            )
            for m in models
        ]


@app.post("/api/models", response_model=ModelResponse)
async def create_model(
    model: ModelCreate,
    db: Database = Depends(get_db)
):
    """Register a new model"""
    with db.session() as session:
        # Check if model_id already exists
        existing = session.execute(
            select(Model).where(Model.model_id == model.model_id)
        ).scalar_one_or_none()
        
        if existing:
            raise HTTPException(status_code=400, detail=f"Model {model.model_id} already exists")
        
        new_model = Model(
            model_id=model.model_id,
            version=model.version,
            checkpoint_hash=model.checkpoint_hash,
            training_dataset_hash=model.training_dataset_hash,
            architecture=model.architecture,
            hyperparameters=model.hyperparameters,
            notes=model.notes
        )
        session.add(new_model)
        session.flush()
        
        return ModelResponse(
            id=new_model.id,
            model_id=new_model.model_id,
            version=new_model.version,
            checkpoint_hash=new_model.checkpoint_hash,
            training_dataset_hash=new_model.training_dataset_hash,
            architecture=new_model.architecture,
            training_date=new_model.training_date,
            notes=new_model.notes
        )


@app.get("/api/predictions", response_model=List[PredictionResponse])
async def list_predictions(
    model_id: Optional[str] = Query(None),
    validated: Optional[bool] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    db: Database = Depends(get_db)
):
    """List predictions with optional filters"""
    with db.session() as session:
        query = select(Prediction, Model).join(Model, Prediction.model_id == Model.id)
        
        if model_id:
            query = query.where(Model.model_id == model_id)
        
        if validated is not None:
            if validated:
                # Has outcome
                validated_ids = select(ExperimentOutcome.prediction_id)
                query = query.where(Prediction.id.in_(validated_ids))
            else:
                # No outcome
                validated_ids = select(ExperimentOutcome.prediction_id)
                query = query.where(Prediction.id.not_in(validated_ids))
        
        query = query.order_by(Prediction.prediction_date.desc()).limit(limit)
        
        results = session.execute(query).all()
        
        return [
            PredictionResponse(
                id=pred.id,
                prediction_id=pred.prediction_id,
                material_formula=pred.material_formula,
                predicted_tc=pred.predicted_tc,
                uncertainty=pred.uncertainty,
                predicted_class=pred.predicted_class,
                confidence=pred.confidence,
                prediction_date=pred.prediction_date,
                model_id=model.model_id,
                model_version=model.version
            )
            for pred, model in results
        ]


@app.post("/api/predictions", response_model=PredictionResponse)
async def create_prediction(
    prediction: PredictionCreate,
    db: Database = Depends(get_db)
):
    """Create a new prediction"""
    with db.session() as session:
        # Get model
        model = session.execute(
            select(Model).where(Model.model_id == prediction.model_id)
        ).scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {prediction.model_id} not found")
        
        new_pred = Prediction(
            prediction_id=prediction.prediction_id,
            model_id=model.id,
            material_formula=prediction.material_formula,
            predicted_tc=prediction.predicted_tc,
            uncertainty=prediction.uncertainty,
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence
        )
        session.add(new_pred)
        session.flush()
        
        return PredictionResponse(
            id=new_pred.id,
            prediction_id=new_pred.prediction_id,
            material_formula=new_pred.material_formula,
            predicted_tc=new_pred.predicted_tc,
            uncertainty=new_pred.uncertainty,
            predicted_class=new_pred.predicted_class,
            confidence=new_pred.confidence,
            prediction_date=new_pred.prediction_date,
            model_id=model.model_id,
            model_version=model.version
        )


@app.post("/api/experiments", response_model=ExperimentResponse)
async def create_experiment(
    experiment: ExperimentOutcomeCreate,
    db: Database = Depends(get_db)
):
    """Add experimental outcome for a prediction"""
    with db.session() as session:
        # Get prediction
        pred = session.execute(
            select(Prediction).where(Prediction.prediction_id == experiment.prediction_id)
        ).scalar_one_or_none()
        
        if not pred:
            raise HTTPException(status_code=404, detail=f"Prediction {experiment.prediction_id} not found")
        
        # Check if outcome already exists
        existing = session.execute(
            select(ExperimentOutcome).where(ExperimentOutcome.prediction_id == pred.id)
        ).scalar_one_or_none()
        
        if existing:
            raise HTTPException(status_code=400, detail=f"Outcome already exists for prediction {experiment.prediction_id}")
        
        # Create outcome
        outcome = ExperimentOutcome(
            prediction_id=pred.id,
            experiment_id=experiment.experiment_id,
            actual_tc=experiment.actual_tc,
            validation_status=experiment.validation_status,
            phase_purity=experiment.phase_purity,
            is_superconductor=experiment.actual_tc > 0,
            notes=experiment.notes
        )
        session.add(outcome)
        session.flush()
        
        # Compute error
        error = PredictionError(
            prediction_id=pred.id,
            absolute_error=pred.predicted_tc - experiment.actual_tc,
            relative_error=((pred.predicted_tc - experiment.actual_tc) / experiment.actual_tc * 100) if experiment.actual_tc != 0 else None,
            squared_error=(pred.predicted_tc - experiment.actual_tc) ** 2
        )
        session.add(error)
        session.flush()
        
        return ExperimentResponse(
            experiment_id=outcome.experiment_id,
            prediction_id=pred.prediction_id,
            material_formula=pred.material_formula,
            predicted_tc=pred.predicted_tc,
            actual_tc=outcome.actual_tc,
            validation_status=outcome.validation_status,
            phase_purity=outcome.phase_purity,
            absolute_error=error.absolute_error,
            experiment_date=outcome.experiment_date
        )


@app.get("/api/performance/{model_id}", response_model=PerformanceResponse)
async def get_performance(
    model_id: str,
    db: Database = Depends(get_db)
):
    """Get model performance summary"""
    with db.session() as session:
        queries = PredictionQueries(session)
        perf = queries.model_performance_summary(model_id)
        
        if 'error' in perf:
            raise HTTPException(status_code=404, detail=perf['error'])
        
        return PerformanceResponse(**perf)


@app.get("/api/candidates")
async def get_candidates(
    limit: int = Query(10, ge=1, le=100),
    min_tc: float = Query(30.0),
    max_uncertainty: Optional[float] = Query(None),
    db: Database = Depends(get_db)
):
    """Get top candidates for validation"""
    with db.session() as session:
        queries = PredictionQueries(session)
        candidates = queries.top_candidates_for_validation(
            limit=limit,
            min_tc=min_tc,
            max_uncertainty=max_uncertainty
        )
        
        return {"candidates": candidates, "count": len(candidates)}


@app.get("/api/errors")
async def get_large_errors(
    threshold: float = Query(10.0),
    limit: int = Query(10, ge=1, le=100),
    db: Database = Depends(get_db)
):
    """Get predictions with large errors"""
    with db.session() as session:
        queries = PredictionQueries(session)
        errors = queries.predictions_with_large_errors(threshold, limit)
        
        return {"errors": errors, "count": len(errors), "threshold": threshold}


# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

