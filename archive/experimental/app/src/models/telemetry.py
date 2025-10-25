"""
Telemetry models for dual-model audit and observability.

These Pydantic models capture the complete execution trace of a dual-model
reasoning request, including input, both model responses, timings, token usage,
and outcome status.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class ModelTrace(BaseModel):
    """
    Execution trace for a single model (Flash or Pro).
    
    Captures all relevant information for scientific audit:
    - Model identity and version
    - Request/response timing
    - Token usage for cost analysis
    - Error information (if applicable)
    - Raw metadata from Vertex AI
    """
    
    model: str = Field(..., description="Model name (e.g., 'gemini-2.5-flash')")
    version: Optional[str] = Field(None, description="Model version string")
    request_id: Optional[str] = Field(None, description="Vertex AI request ID")
    
    prompt_tokens: Optional[int] = Field(None, description="Input tokens consumed")
    completion_tokens: Optional[int] = Field(None, description="Output tokens generated")
    
    latency_ms: Optional[float] = Field(None, description="End-to-end latency in milliseconds")
    
    error: Optional[str] = Field(None, description="Error message if call failed")
    error_class: Optional[str] = Field(None, description="Error class name for categorization")
    
    raw_metadata: Optional[Dict[str, Any]] = Field(None, description="Raw Vertex response metadata")
    
    started_at: str = Field(..., description="ISO 8601 start timestamp")
    ended_at: str = Field(..., description="ISO 8601 end timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "gemini-2.5-flash",
                "version": "flash-001",
                "request_id": "vertex-req-abc123",
                "prompt_tokens": 45,
                "completion_tokens": 128,
                "latency_ms": 450.23,
                "error": None,
                "error_class": None,
                "raw_metadata": {"safety_ratings": []},
                "started_at": "2025-10-07T08:00:00.000Z",
                "ended_at": "2025-10-07T08:00:00.450Z"
            }
        }


class DualRunRecord(BaseModel):
    """
    Complete audit record for a dual-model reasoning request.
    
    Used for:
    - Scientific reproducibility (what inputs produced what outputs)
    - Cost analysis (token usage, model selection impact)
    - Performance debugging (latency breakdown)
    - Error analysis (failure patterns, timeout rates)
    
    Persisted to database with run_id as idempotency key.
    """
    
    run_id: str = Field(..., description="Unique request ID (UUID4) - idempotency key")
    
    project: Optional[str] = Field(None, description="GCP project ID")
    location: Optional[str] = Field(None, description="GCP location (e.g., 'us-central1')")
    
    # Input (sanitized via compliance.sanitize_payload before persistence)
    input: Dict[str, Any] = Field(
        ...,
        description="Request input: {prompt: str, context: dict}"
    )
    
    # Model traces (null if not executed due to early termination)
    flash: Optional[ModelTrace] = Field(None, description="Flash model execution trace")
    pro: Optional[ModelTrace] = Field(None, description="Pro model execution trace")
    
    # Overall request outcome
    status: str = Field(
        ...,
        description="Overall status: ok | timeout | cancelled | error"
    )
    
    # Request timing
    started_at: str = Field(..., description="ISO 8601 request start timestamp")
    ended_at: str = Field(..., description="ISO 8601 request end timestamp")
    
    # Optional user/session context
    user_id: Optional[str] = Field(None, description="User ID (if authenticated)")
    session_id: Optional[str] = Field(None, description="Session ID (if available)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "project": "periodicdent42",
                "location": "us-central1",
                "input": {
                    "prompt": "Suggest an experiment for perovskites",
                    "context": {"domain": "materials"}
                },
                "flash": {
                    "model": "gemini-2.5-flash",
                    "latency_ms": 450.23,
                    "prompt_tokens": 45,
                    "completion_tokens": 128,
                    "started_at": "2025-10-07T08:00:00.000Z",
                    "ended_at": "2025-10-07T08:00:00.450Z"
                },
                "pro": {
                    "model": "gemini-2.5-pro",
                    "latency_ms": 12340.56,
                    "prompt_tokens": 45,
                    "completion_tokens": 512,
                    "started_at": "2025-10-07T08:00:00.000Z",
                    "ended_at": "2025-10-07T08:00:12.340Z"
                },
                "status": "ok",
                "started_at": "2025-10-07T08:00:00.000Z",
                "ended_at": "2025-10-07T08:00:12.340Z"
            }
        }


def create_model_trace(
    model: str,
    started_at: str,
    ended_at: str,
    *,
    response: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None
) -> ModelTrace:
    """
    Factory function to create ModelTrace from response dict or exception.
    
    Args:
        model: Model name
        started_at: ISO 8601 start timestamp
        ended_at: ISO 8601 end timestamp
        response: Response dict from dual_agent (optional if error)
        error: Exception if call failed (optional if response)
    
    Returns:
        Populated ModelTrace instance
    """
    if error:
        return ModelTrace(
            model=model,
            error=str(error),
            error_class=error.__class__.__name__,
            started_at=started_at,
            ended_at=ended_at
        )
    
    if not response:
        # Neither response nor error - shouldn't happen, but handle defensively
        return ModelTrace(
            model=model,
            error="No response and no error",
            error_class="UnknownState",
            started_at=started_at,
            ended_at=ended_at
        )
    
    # Extract usage metadata from response
    usage = response.get("usage", {})
    
    return ModelTrace(
        model=model,
        version=response.get("model"),  # Actual model version if returned
        latency_ms=response.get("latency_ms"),
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        error=response.get("error"),
        error_class=response.get("error_class"),
        raw_metadata=response.get("raw_metadata"),
        started_at=started_at,
        ended_at=ended_at
    )


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.utcnow().isoformat() + "Z"
