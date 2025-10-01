"""
Security utilities and middleware for the FastAPI application.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Iterable, Optional, Set

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


def _normalise_paths(exempt_paths: Optional[Iterable[str]]) -> Set[str]:
    if not exempt_paths:
        return set()
    normalised = set()
    for path in exempt_paths:
        if not path:
            continue
        normalised.add(path.rstrip("/"))
    return normalised


def _is_exempt(path: str, exempt_paths: Set[str]) -> bool:
    if not exempt_paths:
        return False
    clean_path = path.rstrip("/") or "/"
    if clean_path in exempt_paths:
        return True
    for exempt in exempt_paths:
        if exempt == "/":
            continue
        if clean_path.startswith(exempt + "/"):
            return True
    return False


def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware."""

    def __init__(
        self,
        app,
        *,
        enabled: bool,
        api_key: Optional[str],
        header_name: str = "x-api-key",
        exempt_paths: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.api_key = api_key
        self.header_name = header_name
        self.exempt_paths = _normalise_paths(exempt_paths)
        if self.enabled and not self.api_key:
            logger.error("Authentication enabled but API key is missing. Incoming requests will be rejected.")

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if not self.enabled:
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if _is_exempt(path, self.exempt_paths):
            return await call_next(request)

        if not self.api_key:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "code": "auth_configuration_error",
                },
            )

        provided_key = request.headers.get(self.header_name)
        if not provided_key or provided_key != self.api_key:
            logger.warning("Unauthorized request rejected for path %s", path)
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "code": "unauthorized"},
            )

        return await call_next(request)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """IP based sliding window rate limiter."""

    def __init__(self, app, *, limit_per_minute: int) -> None:
        super().__init__(app)
        self.limit = max(0, limit_per_minute)
        self.window_seconds = 60
        self._requests = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if self.limit <= 0 or request.method == "OPTIONS":
            return await call_next(request)

        client_ip = _get_client_ip(request)
        now = time.monotonic()

        async with self._lock:
            bucket = self._requests[client_ip]
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.limit:
                logger.warning("Rate limit exceeded for IP %s", client_ip)
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many requests", "code": "rate_limited"},
                )

            bucket.append(now)

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add common security headers to every response."""

    def __init__(self, app) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response: Response = await call_next(request)
        response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "accelerometer=(), autoplay=*, camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()")
        return response

