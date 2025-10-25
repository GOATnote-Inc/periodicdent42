"""
Server-Sent Events (SSE) utilities for streaming responses.
"""

import json
from typing import Any, Dict


def sse_event(event: str, data: Dict[str, Any]) -> str:
    """
    Format data as Server-Sent Event.
    
    Args:
        event: Event type (e.g., 'preliminary', 'final', 'error')
        data: Data payload (will be JSON serialized)
    
    Returns:
        Formatted SSE string with proper line breaks
    
    Example output:
        event: preliminary
        data: {"response": {...}, "message": "Computing..."}
        
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_message(message: str) -> str:
    """
    Send a simple text message via SSE.
    
    Args:
        message: Text message to send
    
    Returns:
        Formatted SSE string
    """
    return f"data: {json.dumps({'message': message})}\n\n"


def sse_error(
    error: str,
    *,
    code: str = "",
    details: str = "",
    error_type: str = "error",
    retryable: bool = False
) -> str:
    """
    Send error via SSE with structured metadata.
    
    Args:
        error: Error message
        code: Optional error code for client-side handling
        details: Optional error details (internal use only)
        error_type: Error type (error | timeout | cancelled)
        retryable: Whether client should retry
    
    Returns:
        Formatted SSE error event
    """
    payload = {
        "error": error,
        "type": error_type,
        "retryable": retryable
    }

    if code:
        payload["code"] = code

    if details:
        payload["details"] = details

    return sse_event("error", payload)


def sse_done(trace_id: str = "") -> str:
    """
    Send 'done' event to signal stream closure.
    
    Always send this as the final SSE event so clients never hang.
    
    Args:
        trace_id: Optional request trace ID for correlation
    
    Returns:
        Formatted SSE done event
    """
    payload = {}
    if trace_id:
        payload["trace_id"] = trace_id
    
    return sse_event("done", payload)

