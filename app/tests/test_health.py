"""
Test health check endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys

# Mock vertexai modules before any imports that might use them
mock_vertexai = MagicMock()
mock_generative_models = MagicMock()
sys.modules['vertexai'] = mock_vertexai
sys.modules['vertexai.generative_models'] = mock_generative_models
sys.modules['vertexai.preview'] = MagicMock()
sys.modules['vertexai.preview.generative_models'] = mock_generative_models
sys.modules['google.cloud.aiplatform'] = MagicMock()

from src.api.main import app

client = TestClient(app)


def test_health_check_returns_200():
    """Health check should return 200 OK."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_check_includes_project_id():
    """Health check should include project ID."""
    response = client.get("/health")
    
    data = response.json()
    assert "project_id" in data
    assert data["project_id"] == "periodicdent42"


def test_root_endpoint():
    """Root endpoint should return 200."""
    response = client.get("/")
    
    assert response.status_code == 200
    # May return HTML or JSON depending on whether static files exist


def test_health_check_requires_api_key_when_enabled(monkeypatch):
    """Health endpoint should require API key when authentication is enabled."""
    monkeypatch.setattr("src.utils.settings.settings.ENABLE_AUTH", True, raising=False)
    monkeypatch.setattr("src.utils.settings.settings.API_KEY", "secret", raising=False)

    auth_client = TestClient(app)
    response = auth_client.get("/health")

    assert response.status_code == 401


def test_health_check_allows_valid_api_key(monkeypatch):
    """Health endpoint should return 200 when valid API key is provided."""
    monkeypatch.setattr("src.utils.settings.settings.ENABLE_AUTH", True, raising=False)
    monkeypatch.setattr("src.utils.settings.settings.API_KEY", "secret", raising=False)

    auth_client = TestClient(app)
    response = auth_client.get("/health", headers={"x-api-key": "secret"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
