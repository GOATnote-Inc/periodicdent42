"""
Smoke test for reasoning endpoint with mocked Vertex AI.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


# Mock response object
class MockResponse:
    def __init__(self, text: str, input_tokens: int = 100, output_tokens: int = 200):
        self.text = text
        self.usage_metadata = MagicMock()
        self.usage_metadata.prompt_token_count = input_tokens
        self.usage_metadata.candidates_token_count = output_tokens


@pytest.fixture
def mock_vertex():
    """Mock Vertex AI models."""
    with patch('app.src.services.vertex.aiplatform') as mock_ai:
        # Mock init
        mock_ai.init = MagicMock()
        
        # Mock Flash model
        mock_flash = MagicMock()
        mock_flash.generate_content = MagicMock(
            return_value=MockResponse("Quick preliminary response from Flash")
        )
        
        # Mock Pro model
        mock_pro = MagicMock()
        mock_pro.generate_content = MagicMock(
            return_value=MockResponse("Detailed verified response from Pro with reasoning steps.")
        )
        
        mock_ai.GenerativeModel = MagicMock(side_effect=[mock_flash, mock_pro])
        
        yield mock_ai


@pytest.mark.asyncio
async def test_reasoning_endpoint_streams_preliminary_and_final(mock_vertex):
    """Test that reasoning endpoint streams both preliminary and final events."""
    from app.src.api.main import app
    
    client = TestClient(app)
    
    # Make request
    response = client.post(
        "/api/reasoning/query",
        json={
            "query": "Suggest an experiment for perovskites",
            "context": {"domain": "materials"}
        },
        stream=True
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Collect events
    events = []
    for line in response.iter_lines():
        if line.startswith(b"event:"):
            event_type = line.decode().split(":", 1)[1].strip()
            events.append(event_type)
    
    # Should have preliminary then final
    assert "preliminary" in events
    assert "final" in events
    
    # Preliminary should come before final
    assert events.index("preliminary") < events.index("final")


@pytest.mark.asyncio
async def test_reasoning_endpoint_requires_query(mock_vertex):
    """Test that query field is required."""
    from app.src.api.main import app
    
    client = TestClient(app)
    
    # Request without query
    response = client.post(
        "/api/reasoning/query",
        json={"context": {}}
    )
    
    # Should fail validation
    assert response.status_code == 422  # Validation error

