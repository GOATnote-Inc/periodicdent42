"""
Test health check endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys

# Mock vertexai modules before any imports that might use them
sys.modules['vertexai'] = MagicMock()
sys.modules['vertexai.generative_models'] = MagicMock()

from app.src.api.main import app

client = TestClient(app)


def test_health_check_returns_200():
    """Health check should return 200 OK."""
    response = client.get("/healthz")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_check_includes_project_id():
    """Health check should include project ID."""
    response = client.get("/healthz")
    
    data = response.json()
    assert "project_id" in data
    assert data["project_id"] == "periodicdent42"


def test_root_endpoint():
    """Root endpoint should return API info."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data
