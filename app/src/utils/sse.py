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


def sse_error(error: str, details: str = "") -> str:
    """
    Send error via SSE.
    
    Args:
        error: Error message
        details: Optional error details
    
    Returns:
        Formatted SSE error event
    """
    return sse_event("error", {
        "error": error,
        "details": details
    })

