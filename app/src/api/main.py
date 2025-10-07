"""FastAPI application for Autonomous R&D Intelligence Layer."""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.security import (
    AuthenticationMiddleware,
    RateLimiterMiddleware,
    SecurityHeadersMiddleware,
)
from src.utils.compliance import generate_request_id
from src.utils.settings import settings
from src.reasoning.dual_agent import DualModelAgent
from src.services.vertex import init_vertex, is_initialized
from src.utils.sse import sse_event, sse_error
from src.services.storage import get_storage
from src.lab.campaign import get_campaign_runner, CampaignReport
from src.services import db

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
print(f"🔍 STATIC_DIR resolved to: {STATIC_DIR}", flush=True)
print(f"🔍 STATIC_DIR exists: {STATIC_DIR.exists()}", flush=True)
if STATIC_DIR.exists():
    print(f"✅ Mounting static files from: {STATIC_DIR}", flush=True)
    files_in_static = list(STATIC_DIR.glob("*"))
    print(f"📁 Files in static: {[f.name for f in files_in_static]}", flush=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"❌ Static directory not found at: {STATIC_DIR}", flush=True)

def _parse_allowed_origins(raw_origins: str) -> List[str]:
    # Always allow Cloud Storage and Cloud Run origins for analytics dashboard
    default_origins = [
        "https://storage.googleapis.com",
        "https://ard-backend-dydzexswua-uc.a.run.app",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    if origins:
        # Combine custom origins with defaults
        all_origins = list(set(default_origins + origins))
        logger.info(f"CORS origins configured: {all_origins}")
        return all_origins
    
    logger.info(f"Using default CORS origins: {default_origins}")
    return default_origins


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
AUTH_EXEMPT_PATHS.update({
    "/docs", 
    "/openapi.json", 
    "/", 
    "/static", 
    "/health",
    "/api/experiments",
    "/api/optimization_runs",
    "/api/ai_queries",
    "/analytics.html",
})

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


class CampaignRequest(BaseModel):
    """Request body for triggering the UV-Vis autonomous campaign."""

    experiments: int = 50
    max_hours: float = 24.0


class CampaignResponse(BaseModel):
    """Response payload summarizing a campaign run."""

    campaign_id: str
    instrument_id: str
    experiments_requested: int
    experiments_completed: int
    started_at: datetime
    completed_at: datetime
    storage_uris: List[str]
    failures: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global agent
    
    logger.info("Starting Autonomous R&D Intelligence Layer...")
    logger.info(f"Project: {settings.PROJECT_ID}, Location: {settings.LOCATION}")
    
    # Debug static directory
    logger.info(f"🔍 STATIC_DIR: {STATIC_DIR}")
    logger.info(f"🔍 STATIC_DIR exists: {STATIC_DIR.exists()}")
    if STATIC_DIR.exists():
        files = list(STATIC_DIR.glob("*"))
        logger.info(f"📁 Files in static directory: {[f.name for f in files]}")
    else:
        logger.error(f"❌ STATIC_DIR does not exist!")
    
    try:
        # Initialize Vertex AI
        init_vertex(settings.PROJECT_ID, settings.LOCATION)

        # Initialize Cloud SQL (or local) database
        db.init_database()

        # Initialize agent
        agent = DualModelAgent(
            project_id=settings.PROJECT_ID,
            location=settings.LOCATION
        )
        
        logger.info("Startup complete - ready to serve requests")
    
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't crash - allow health check to report status


def _verify_health_access(request: Request) -> None:
    if not settings.ENABLE_AUTH:
        return
    api_key = request.headers.get("x-api-key")
    if not api_key or api_key != settings.API_KEY:
        logger.warning("Unauthorized health check attempt from %s", request.client)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """
    Health check endpoint.
    
    Note: Uses /health (not /healthz) as Cloud Run reserves the /healthz path.
    
    Returns:
        200 OK if service is healthy, including Vertex AI initialization status
    """
    _verify_health_access(request)

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

    request_id = generate_request_id()
    logger.info(
        "Query received [request_id=%s, query_length=%d]",
        request_id,
        len(prompt),
    )
    
    async def event_stream():
        """Stream SSE events for Flash then Pro."""
        try:
            # Launch both models in parallel via the agent helper
            flash_task, pro_task = agent.start_query_tasks(prompt, context)
            
            # Stream Flash response immediately
            flash_response = await flash_task
            yield sse_event("preliminary", {
                "response": flash_response,
                "message": "⚡ Quick preview - computing verified response..."
            })
            
            logger.info(
                "Flash response sent [request_id=%s, latency_ms=%s]",
                request_id,
                flash_response.get("latency_ms"),
            )
            
            # Stream Pro response when ready
            pro_response = await pro_task
            yield sse_event("final", {
                "response": pro_response,
                "message": "✅ Verified response ready"
            })
            
            logger.info(
                "Pro response sent [request_id=%s, latency_ms=%s]",
                request_id,
                pro_response.get("latency_ms"),
            )
        
        except Exception as e:
            logger.error(f"Error in event stream: {e}")
            yield sse_error(
                error="Internal server error",
                code="stream_failure"
            )
    
    response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

    response.headers.setdefault("X-Request-ID", request_id)

    return response


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


@app.get("/analytics.html")
async def analytics_page():
    """Serve the analytics dashboard."""
    analytics_path = STATIC_DIR / "analytics.html"
    if analytics_path.exists():
        return FileResponse(analytics_path)
    raise HTTPException(status_code=404, detail="Analytics page not found")


# Campaign orchestration
@app.post("/api/lab/campaign/uvvis", response_model=CampaignResponse)
async def run_uvvis_campaign(request: CampaignRequest):
    """Trigger the UV-Vis autonomous campaign in simulation mode."""

    runner = get_campaign_runner()
    report: CampaignReport = runner.run_campaign(
        min_experiments=request.experiments,
        max_hours=request.max_hours,
    )
    return CampaignResponse(
        campaign_id=report.campaign_id,
        instrument_id=report.instrument_id,
        experiments_requested=report.experiments_requested,
        experiments_completed=report.experiments_completed,
        started_at=report.started_at,
        completed_at=report.completed_at,
        storage_uris=report.storage_uris,
        failures=report.failures,
    )


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
        logger.error(f"Storage error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/api/storage/experiments")
async def list_experiments():
    """List all stored experiments."""
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
        logger.error(f"Storage error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# Database metadata endpoints for analytics dashboard
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
        status: Filter by status (pending, running, completed, failed)
        optimization_run_id: Filter by optimization run ID
        limit: Maximum number of results (default: 100)
        created_by: Filter by creator
    """
    try:
        experiments = db.get_experiments(
            status=status,
            optimization_run_id=optimization_run_id,
            created_by=created_by,
            limit=limit
        )
        
        # Convert to dict for JSON serialization
        exp_dicts = [db.experiment_to_dict(exp) for exp in experiments]
        
        return {
            "experiments": exp_dicts,
            "count": len(exp_dicts)
        }
    
    except Exception as e:
        logger.error(f"Database error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to query experiments"}
        )


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
        status: Filter by status (pending, running, completed, failed)
        method: Filter by optimization method (reinforcement_learning, bayesian_optimization, adaptive_router)
        limit: Maximum number of results (default: 50)
        created_by: Filter by creator
    """
    try:
        runs = db.get_optimization_runs(
            status=status,
            method=method,
            created_by=created_by,
            limit=limit
        )
        
        # Convert to dict for JSON serialization
        run_dicts = [db.optimization_run_to_dict(run) for run in runs]
        
        return {
            "optimization_runs": run_dicts,
            "count": len(run_dicts)
        }
    
    except Exception as e:
        logger.error(f"Database error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to query optimization runs"}
        )


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
        limit: Maximum number of results (default: 100)
        selected_model: Filter by model (flash, pro, adaptive_router)
        created_by: Filter by creator
        include_cost_analysis: Include cost analysis summary (default: true)
    """
    try:
        session = db.get_session()
        if session is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Database not available"}
            )
        
        # Build query
        query = session.query(db.AIQuery)
        
        if selected_model:
            query = query.filter(db.AIQuery.selected_model == selected_model)
        if created_by:
            query = query.filter(db.AIQuery.created_by == created_by)
        
        ai_queries = query.order_by(db.AIQuery.created_at.desc()).limit(limit).all()
        
        # Convert to dict
        query_dicts = [db.ai_query_to_dict(q) for q in ai_queries]
        total_cost = sum((q["cost_usd"] or 0) for q in query_dicts)
        total_queries = len(query_dicts)
        
        response = {
            "ai_queries": query_dicts,
            "count": total_queries
        }
        
        if include_cost_analysis:
            response["cost_analysis"] = {
                "total_cost_usd": round(total_cost, 6),
                "average_cost_per_query": round(total_cost / total_queries, 6) if total_queries > 0 else 0
            }
        
        session.close()
        return response
    
    except Exception as e:
        logger.error(f"Database error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to query AI queries"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.src.api.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )

