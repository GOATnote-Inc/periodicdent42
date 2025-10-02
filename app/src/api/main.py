"""FastAPI application for Autonomous R&D Intelligence Layer."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.security import (
    AuthenticationMiddleware,
    RateLimiterMiddleware,
    SecurityHeadersMiddleware,
)
from src.utils.settings import settings
from src.reasoning.dual_agent import DualModelAgent
from src.services.vertex import init_vertex, is_initialized
from src.utils.sse import sse_event, sse_error
from src.services.storage import get_storage
from src.services.db import (
    init_database, close_database, get_experiments, get_optimization_runs,
    ExperimentStatus, OptimizationMethod, get_session, Experiment,
    OptimizationRun, AIQuery
)

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Autonomous R&D Intelligence Layer",
    description="Dual-model AI reasoning with Gemini 2.5 Flash + Pro",
    version="0.1.0"
)

STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def _parse_allowed_origins(raw_origins: str) -> List[str]:
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    if origins:
        return origins

    if settings.ENVIRONMENT.lower() == "development":
        return [
            "http://localhost",
            "http://localhost:3000",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
        ]

    logger.warning("No CORS origins configured; browser requests will be blocked by default.")
    return []


ALLOWED_ORIGINS = _parse_allowed_origins(settings.ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Requested-With",
        "X-API-Key",
    ],
    expose_headers=["X-Request-ID"],
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    RateLimiterMiddleware,
    limit_per_minute=settings.RATE_LIMIT_PER_MINUTE,
)

AUTH_EXEMPT_PATHS = set()
# Note: /health is also exempt since Cloud Run and load balancers need unauthenticated health checks
# Protected endpoints: /api/reasoning/query, /api/storage/*
AUTH_EXEMPT_PATHS.update({"/docs", "/openapi.json", "/", "/static", "/health"})

app.add_middleware(
    AuthenticationMiddleware,
    enabled=settings.ENABLE_AUTH,
    api_key=settings.API_KEY,
    exempt_paths=AUTH_EXEMPT_PATHS,
)

# Global agent instance (initialized on startup)
agent: DualModelAgent = None


class QueryRequest(BaseModel):
    """Request model for reasoning queries."""
    query: str
    context: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vertex_initialized: bool = False
    project_id: str = ""


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global agent
    
    logger.info("Starting Autonomous R&D Intelligence Layer...")
    logger.info(f"Project: {settings.PROJECT_ID}, Location: {settings.LOCATION}")
    
    try:
        # Initialize Cloud SQL database
        init_database()
        
        # Initialize Vertex AI
        init_vertex(settings.PROJECT_ID, settings.LOCATION)
        
        # Initialize agent
        agent = DualModelAgent(
            project_id=settings.PROJECT_ID,
            location=settings.LOCATION
        )
        
        logger.info("✅ Startup complete - ready to serve requests")
    
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't crash - allow health check to report status


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Autonomous R&D Intelligence Layer...")
    
    try:
        # Close database connections
        close_database()
        
        logger.info("✅ Shutdown complete")
    
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Note: Uses /health (not /healthz) as Cloud Run reserves the /healthz path.
    Authentication is enforced by middleware when ENABLE_AUTH=True.
    
    Returns:
        200 OK if service is healthy, including Vertex AI initialization status
    """
    return HealthResponse(
        status="ok",
        vertex_initialized=is_initialized(),
        project_id=settings.PROJECT_ID
    )


@app.post("/api/reasoning/query")
async def query_with_feedback(body: QueryRequest, request: Request):
    """
    Query endpoint with dual-model streaming.
    
    Returns Server-Sent Events (SSE):
    1. `event: preliminary` - Fast Flash response (<2s)
    2. `event: final` - Verified Pro response (10-30s)
    
    Args:
        body: Query request with prompt and context
    
    Returns:
        StreamingResponse with text/event-stream
    """
    if agent is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not initialized"}
        )
    
    prompt = body.query
    context = body.context
    
    logger.info(f"Query received: {prompt[:100]}...")
    
    async def event_stream():
        """Stream SSE events for Flash then Pro."""
        try:
            # Launch both models in parallel
            flash_task = asyncio.create_task(agent._query_flash(prompt, context))
            pro_task = asyncio.create_task(agent._query_pro(prompt, context))
            
            # Stream Flash response immediately
            flash_response = await flash_task
            yield sse_event("preliminary", {
                "response": flash_response,
                "message": "⚡ Quick preview - computing verified response..."
            })
            
            logger.info(f"Flash response sent: {flash_response['latency_ms']}ms")
            
            # Stream Pro response when ready
            pro_response = await pro_task
            yield sse_event("final", {
                "response": pro_response,
                "message": "✅ Verified response ready"
            })
            
            logger.info(f"Pro response sent: {pro_response['latency_ms']}ms")
        
        except Exception as e:
            logger.error(f"Error in event stream: {e}")
            yield sse_error(
                error="Internal server error",
                code="stream_failure"
            )
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/")
async def root():
    """
    Root endpoint - serves the web UI if available, otherwise returns API info.
    """
    # Check if static UI exists
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    # Fallback to JSON response
    return {
        "service": "Autonomous R&D Intelligence Layer",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "reasoning": "/api/reasoning/query",
            "docs": "/docs",
            "ui": "/"
        }
    }


# Storage endpoints
@app.post("/api/storage/experiment")
async def store_experiment(data: Dict[str, Any]):
    """
    Store experiment result in Cloud Storage.
    
    Body:
        {
            "experiment_id": "exp-123",
            "result": {...},
            "metadata": {...}  # optional
        }
    """
    storage = get_storage()
    if not storage:
        return JSONResponse(
            status_code=503,
            content={"error": "Cloud Storage not configured"}
        )
    
    experiment_id = data.get("experiment_id")
    result = data.get("result")
    metadata = data.get("metadata", {})
    
    if not experiment_id or not result:
        return JSONResponse(
            status_code=400,
            content={"error": "experiment_id and result are required"}
        )
    
    try:
        uri = storage.store_experiment_result(experiment_id, result, metadata)
        return {"status": "success", "uri": uri}
    except Exception as e:
        logger.error(f"Storage error storing experiment {experiment_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to store experiment", "code": "storage_error"}
        )


@app.get("/api/storage/experiments")
async def list_experiments_storage():
    """List all stored experiments from Cloud Storage."""
    storage = get_storage()
    if not storage:
        return JSONResponse(
            status_code=503,
            content={"error": "Cloud Storage not configured"}
        )
    
    try:
        experiments = storage.list_experiments()
        return {"experiments": experiments, "count": len(experiments)}
    except Exception as e:
        logger.error(f"Storage error listing experiments: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to list experiments", "code": "storage_error"}
        )


# Database metadata endpoints
@app.get("/api/experiments")
async def list_experiments_db(
    status: str = None,
    optimization_run_id: str = None,
    limit: int = 100,
    created_by: str = None
):
    """
    List experiments from database with optional filtering.
    
    Query Parameters:
        status: Filter by status (pending, running, completed, failed, cancelled)
        optimization_run_id: Filter by optimization run
        limit: Maximum number of results (default: 100, max: 1000)
        created_by: Filter by user
    
    Returns:
        {
            "experiments": [...],
            "count": 10,
            "status": "success"
        }
    """
    session = get_session()
    if session is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Database not available", "code": "db_unavailable"}
        )
    
    try:
        # Validate limit
        limit = min(limit, 1000)
        
        # Validate status
        status_enum = None
        if status:
            try:
                status_enum = ExperimentStatus(status)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Invalid status: {status}",
                        "valid_statuses": [s.value for s in ExperimentStatus]
                    }
                )
        
        # Query experiments
        query = session.query(Experiment)
        
        if optimization_run_id:
            query = query.filter(Experiment.optimization_run_id == optimization_run_id)
        
        if status_enum:
            query = query.filter(Experiment.status == status_enum)
        
        if created_by:
            query = query.filter(Experiment.created_by == created_by)
        
        query = query.order_by(Experiment.created_at.desc()).limit(limit)
        
        experiments = query.all()
        
        # Convert to dict
        experiments_data = [
            {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "status": exp.status.value,
                "parameters": exp.parameters,
                "config": exp.config,
                "result_value": exp.result_value,
                "result_data": exp.result_data,
                "result_uri": exp.result_uri,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
                "started_at": exp.started_at.isoformat() if exp.started_at else None,
                "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
                "created_by": exp.created_by,
                "optimization_run_id": exp.optimization_run_id
            }
            for exp in experiments
        ]
        
        return {
            "experiments": experiments_data,
            "count": len(experiments_data),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Database error listing experiments: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to list experiments", "code": "db_error"}
        )
    
    finally:
        session.close()


@app.get("/api/experiments/{experiment_id}")
async def get_experiment_details(experiment_id: str):
    """
    Get detailed information about a specific experiment.
    
    Returns:
        {
            "experiment": {...},
            "status": "success"
        }
    """
    session = get_session()
    if session is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Database not available", "code": "db_unavailable"}
        )
    
    try:
        experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
        
        if not experiment:
            return JSONResponse(
                status_code=404,
                content={"error": f"Experiment not found: {experiment_id}"}
            )
        
        experiment_data = {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status.value,
            "parameters": experiment.parameters,
            "config": experiment.config,
            "result_value": experiment.result_value,
            "result_data": experiment.result_data,
            "result_uri": experiment.result_uri,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
            "created_by": experiment.created_by,
            "optimization_run_id": experiment.optimization_run_id
        }
        
        # Include optimization run info if available
        if experiment.optimization_run:
            experiment_data["optimization_run"] = {
                "id": experiment.optimization_run.id,
                "name": experiment.optimization_run.name,
                "method": experiment.optimization_run.method.value,
                "objective": experiment.optimization_run.objective
            }
        
        return {
            "experiment": experiment_data,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Database error getting experiment {experiment_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get experiment", "code": "db_error"}
        )
    
    finally:
        session.close()


@app.get("/api/optimization_runs")
async def list_optimization_runs_api(
    status: str = None,
    method: str = None,
    limit: int = 50,
    created_by: str = None
):
    """
    List optimization runs from database with optional filtering.
    
    Query Parameters:
        status: Filter by status (pending, running, completed, failed, cancelled)
        method: Filter by method (bayesian_optimization, reinforcement_learning, etc.)
        limit: Maximum number of results (default: 50, max: 500)
        created_by: Filter by user
    
    Returns:
        {
            "runs": [...],
            "count": 5,
            "status": "success"
        }
    """
    session = get_session()
    if session is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Database not available", "code": "db_unavailable"}
        )
    
    try:
        # Validate limit
        limit = min(limit, 500)
        
        # Validate status
        status_enum = None
        if status:
            try:
                status_enum = ExperimentStatus(status)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Invalid status: {status}",
                        "valid_statuses": [s.value for s in ExperimentStatus]
                    }
                )
        
        # Validate method
        method_enum = None
        if method:
            try:
                method_enum = OptimizationMethod(method)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Invalid method: {method}",
                        "valid_methods": [m.value for m in OptimizationMethod]
                    }
                )
        
        # Query runs
        query = session.query(OptimizationRun)
        
        if status_enum:
            query = query.filter(OptimizationRun.status == status_enum)
        
        if method_enum:
            query = query.filter(OptimizationRun.method == method_enum)
        
        if created_by:
            query = query.filter(OptimizationRun.created_by == created_by)
        
        query = query.order_by(OptimizationRun.created_at.desc()).limit(limit)
        
        runs = query.all()
        
        # Convert to dict
        runs_data = [
            {
                "id": run.id,
                "name": run.name,
                "description": run.description,
                "method": run.method.value,
                "objective": run.objective,
                "search_space": run.search_space,
                "config": run.config,
                "num_experiments": run.num_experiments,
                "best_value": run.best_value,
                "best_experiment_id": run.best_experiment_id,
                "status": run.status.value,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "created_by": run.created_by
            }
            for run in runs
        ]
        
        return {
            "runs": runs_data,
            "count": len(runs_data),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Database error listing optimization runs: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to list optimization runs", "code": "db_error"}
        )
    
    finally:
        session.close()


@app.get("/api/ai_queries")
async def list_ai_queries(
    limit: int = 100,
    selected_model: str = None,
    created_by: str = None,
    include_cost_analysis: bool = True
):
    """
    List AI queries from database with optional cost analysis.
    
    Query Parameters:
        limit: Maximum number of results (default: 100, max: 1000)
        selected_model: Filter by model ("flash" or "pro")
        created_by: Filter by user
        include_cost_analysis: Include cost statistics (default: true)
    
    Returns:
        {
            "queries": [...],
            "count": 50,
            "cost_analysis": {...},  # if include_cost_analysis=true
            "status": "success"
        }
    """
    session = get_session()
    if session is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Database not available", "code": "db_unavailable"}
        )
    
    try:
        # Validate limit
        limit = min(limit, 1000)
        
        # Query AI queries
        query = session.query(AIQuery)
        
        if selected_model:
            if selected_model not in ["flash", "pro"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Invalid model: {selected_model}",
                        "valid_models": ["flash", "pro"]
                    }
                )
            query = query.filter(AIQuery.selected_model == selected_model)
        
        if created_by:
            query = query.filter(AIQuery.created_by == created_by)
        
        query = query.order_by(AIQuery.created_at.desc()).limit(limit)
        
        queries = query.all()
        
        # Convert to dict
        queries_data = [
            {
                "id": q.id,
                "query_text": q.query_text[:200] + "..." if len(q.query_text) > 200 else q.query_text,
                "selected_model": q.selected_model,
                "flash_latency_ms": q.flash_latency_ms,
                "pro_latency_ms": q.pro_latency_ms,
                "flash_tokens": q.flash_tokens,
                "pro_tokens": q.pro_tokens,
                "estimated_cost_usd": q.estimated_cost_usd,
                "created_at": q.created_at.isoformat() if q.created_at else None,
                "created_by": q.created_by,
                "experiment_id": q.experiment_id
            }
            for q in queries
        ]
        
        response = {
            "queries": queries_data,
            "count": len(queries_data),
            "status": "success"
        }
        
        # Add cost analysis if requested
        if include_cost_analysis:
            # Calculate cost statistics
            total_flash_tokens = sum(q.flash_tokens or 0 for q in queries)
            total_pro_tokens = sum(q.pro_tokens or 0 for q in queries)
            total_cost = sum(q.estimated_cost_usd or 0 for q in queries)
            
            flash_queries = len([q for q in queries if q.selected_model == "flash"])
            pro_queries = len([q for q in queries if q.selected_model == "pro"])
            
            avg_flash_latency = sum(q.flash_latency_ms or 0 for q in queries if q.flash_latency_ms) / max(flash_queries, 1)
            avg_pro_latency = sum(q.pro_latency_ms or 0 for q in queries if q.pro_latency_ms) / max(pro_queries, 1)
            
            response["cost_analysis"] = {
                "total_queries": len(queries),
                "flash_queries": flash_queries,
                "pro_queries": pro_queries,
                "total_flash_tokens": total_flash_tokens,
                "total_pro_tokens": total_pro_tokens,
                "estimated_total_cost_usd": round(total_cost, 4),
                "avg_flash_latency_ms": round(avg_flash_latency, 2),
                "avg_pro_latency_ms": round(avg_pro_latency, 2),
                "cost_per_query_usd": round(total_cost / len(queries), 6) if queries else 0
            }
        
        return response
    
    except Exception as e:
        logger.error(f"Database error listing AI queries: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to list AI queries", "code": "db_error"}
        )
    
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.src.api.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )

