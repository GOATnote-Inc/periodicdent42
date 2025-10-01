"""
FastAPI application for Autonomous R&D Intelligence Layer.

Provides dual-model reasoning via Gemini 2.5 Flash + Pro.
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import logging
import asyncio

from src.utils.settings import settings
from src.reasoning.dual_agent import DualModelAgent
from src.services.vertex import init_vertex, is_initialized
from src.utils.sse import sse_event, sse_error

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

# CORS - permissive for now (TODO: tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
                details=str(e)
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
    """Root endpoint with API info."""
    return {
        "service": "Autonomous R&D Intelligence Layer",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "reasoning": "/api/reasoning/query",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.src.api.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )

