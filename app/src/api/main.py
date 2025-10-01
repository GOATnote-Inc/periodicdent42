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
        # Initialize Vertex AI
        init_vertex(settings.PROJECT_ID, settings.LOCATION)
        
        # Initialize agent
        agent = DualModelAgent(
            project_id=settings.PROJECT_ID,
            location=settings.LOCATION
        )
        
        logger.info("Startup complete - ready to serve requests")
    
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't crash - allow health check to report status


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
        logger.error(f"Storage error listing experiments: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to list experiments", "code": "storage_error"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.src.api.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )

