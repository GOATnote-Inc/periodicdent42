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


# Note: Authentication behavior is tested in test_security.py
# using middleware directly, since the app-level middleware is
# initialized at import time with static settings.
