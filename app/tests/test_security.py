"""
Test security middleware (authentication, rate limiting, security headers).
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.security import (
    AuthenticationMiddleware,
    RateLimiterMiddleware,
    SecurityHeadersMiddleware,
)


def test_authentication_middleware_disabled():
    """When auth is disabled, all requests should pass through."""
    app = FastAPI()
    app.add_middleware(
        AuthenticationMiddleware,
        enabled=False,
        api_key="secret",
        exempt_paths=[],
    )
    
    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    response = client.get("/test")
    
    assert response.status_code == 200


def test_authentication_middleware_rejects_missing_key():
    """When auth is enabled, requests without API key should be rejected."""
    app = FastAPI()
    app.add_middleware(
        AuthenticationMiddleware,
        enabled=True,
        api_key="secret-key",
        exempt_paths=[],
    )
    
    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    response = client.get("/test")
    
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_authentication_middleware_accepts_valid_key():
    """When auth is enabled, requests with valid API key should pass."""
    app = FastAPI()
    app.add_middleware(
        AuthenticationMiddleware,
        enabled=True,
        api_key="secret-key",
        exempt_paths=[],
    )
    
    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    response = client.get("/test", headers={"x-api-key": "secret-key"})
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_authentication_middleware_exempt_paths():
    """Exempt paths should not require authentication."""
    app = FastAPI()
    app.add_middleware(
        AuthenticationMiddleware,
        enabled=True,
        api_key="secret-key",
        exempt_paths=["/public", "/docs"],
    )
    
    @app.get("/public")
    def public_endpoint():
        return {"status": "public"}
    
    @app.get("/protected")
    def protected_endpoint():
        return {"status": "protected"}
    
    client = TestClient(app)
    
    # Public endpoint should work without auth
    response = client.get("/public")
    assert response.status_code == 200
    
    # Protected endpoint should require auth
    response = client.get("/protected")
    assert response.status_code == 401


def test_rate_limiter_allows_under_limit():
    """Requests under the rate limit should be allowed."""
    app = FastAPI()
    app.add_middleware(
        RateLimiterMiddleware,
        limit_per_minute=5,
    )
    
    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    
    # Make 5 requests (under limit)
    for _ in range(5):
        response = client.get("/test")
        assert response.status_code == 200


def test_rate_limiter_rejects_over_limit():
    """Requests over the rate limit should be rejected."""
    app = FastAPI()
    app.add_middleware(
        RateLimiterMiddleware,
        limit_per_minute=3,
    )
    
    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    
    # Make 3 requests (at limit)
    for _ in range(3):
        response = client.get("/test")
        assert response.status_code == 200
    
    # 4th request should be rate limited
    response = client.get("/test")
    assert response.status_code == 429
    assert response.json()["code"] == "rate_limited"


def test_security_headers_middleware():
    """Security headers should be added to all responses."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)
    
    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    response = client.get("/test")
    
    assert response.status_code == 200
    assert "Strict-Transport-Security" in response.headers
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers
    assert "Referrer-Policy" in response.headers
    assert "Permissions-Policy" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_authentication_middleware_custom_header():
    """Custom header name should work for API key."""
    app = FastAPI()
    app.add_middleware(
        AuthenticationMiddleware,
        enabled=True,
        api_key="secret-key",
        header_name="x-custom-key",
        exempt_paths=[],
    )
    
    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    
    # Wrong header should fail
    response = client.get("/test", headers={"x-api-key": "secret-key"})
    assert response.status_code == 401
    
    # Correct custom header should work
    response = client.get("/test", headers={"x-custom-key": "secret-key"})
    assert response.status_code == 200

