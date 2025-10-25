"""
Rate Limiting Configuration

Implements request rate limiting using slowapi to prevent API abuse.
Supports both in-memory (development) and Redis (production) backends.
"""

import os
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
import redis.asyncio as redis

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
USE_REDIS = os.getenv("USE_REDIS_RATE_LIMIT", "false").lower() == "true"

# Rate limits (configurable via environment)
RATE_LIMIT_ANONYMOUS = os.getenv("RATE_LIMIT_ANONYMOUS", "100/hour")
RATE_LIMIT_AUTHENTICATED = os.getenv("RATE_LIMIT_AUTHENTICATED", "1000/hour")
RATE_LIMIT_PER_MINUTE = os.getenv("RATE_LIMIT_PER_MINUTE", "60/minute")

# Redis client (initialized if USE_REDIS is True)
redis_client = None


def get_redis_client():
    """Get or create Redis client for rate limiting"""
    global redis_client
    
    if not USE_REDIS:
        return None
    
    if redis_client is None:
        try:
            redis_client = redis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            print(f"✅ Connected to Redis for rate limiting: {REDIS_URL}")
        except Exception as e:
            print(f"⚠️  Could not connect to Redis: {e}")
            print("   Falling back to in-memory rate limiting")
            return None
    
    return redis_client


def get_identifier(request: Request) -> str:
    """
    Get identifier for rate limiting.
    
    Uses user ID for authenticated requests, IP address for anonymous.
    
    Args:
        request: FastAPI Request object
    
    Returns:
        Identifier string for rate limiting
    """
    # Check if user is authenticated
    if hasattr(request.state, "user") and request.state.user:
        # Use user ID for authenticated users (more generous limits)
        return f"user:{request.state.user.id}"
    
    # Fall back to IP address for anonymous users
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_identifier,
    default_limits=[RATE_LIMIT_PER_MINUTE],
    storage_uri=REDIS_URL if USE_REDIS else None,
    strategy="fixed-window",  # Can be "fixed-window" or "moving-window"
    headers_enabled=True,  # Add rate limit headers to responses
)


def get_rate_limit_for_user(request: Request) -> str:
    """
    Get appropriate rate limit based on authentication status.
    
    Args:
        request: FastAPI Request object
    
    Returns:
        Rate limit string (e.g., "100/hour")
    """
    if hasattr(request.state, "user") and request.state.user:
        return RATE_LIMIT_AUTHENTICATED
    return RATE_LIMIT_ANONYMOUS


# Custom decorators for different rate limits
def rate_limit_anonymous(func):
    """Decorator for endpoints with anonymous user limits"""
    return limiter.limit(RATE_LIMIT_ANONYMOUS)(func)


def rate_limit_authenticated(func):
    """Decorator for endpoints with authenticated user limits"""
    return limiter.limit(RATE_LIMIT_AUTHENTICATED)(func)


def rate_limit_strict(limit: str):
    """
    Decorator for endpoints with custom strict limits.
    
    Usage:
        @rate_limit_strict("10/minute")
        async def expensive_endpoint():
            ...
    """
    return limiter.limit(limit)

