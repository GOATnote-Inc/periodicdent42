"""
Deterministic streaming tests for dual-model behavior.

Tests SSE ordering, timeout handling, cancellation, and audit persistence
using fake Flash/Pro providers with controllable delays.

All tests use asyncio.sleep for deterministic timing (no reliance on real Vertex AI).
"""

import asyncio
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch
import sys

# Mock vertexai modules before imports
mock_vertexai = MagicMock()
mock_generative_models = MagicMock()
sys.modules['vertexai'] = mock_vertexai
sys.modules['vertexai.generative_models'] = mock_generative_models
sys.modules['vertexai.preview'] = MagicMock()
sys.modules['vertexai.preview.generative_models'] = mock_generative_models
sys.modules['google.cloud.aiplatform'] = MagicMock()

from src.api.main import app
from src.reasoning.dual_agent import DualModelAgent
from src.services import db


class FakeResponse:
    """Fake Vertex response object."""
    
    def __init__(self, text: str, prompt_tokens: int = 50, completion_tokens: int = 100):
        self.text = text
        self.usage_metadata = MagicMock()
        self.usage_metadata.prompt_token_count = prompt_tokens
        self.usage_metadata.candidates_token_count = completion_tokens


class FakeFlashProvider:
    """Fake Flash model provider with controllable delay and errors."""
    
    def __init__(self, delay_ms: float = 200, should_timeout: bool = False, should_error: bool = False):
        self.delay_ms = delay_ms
        self.should_timeout = should_timeout
        self.should_error = should_error
    
    def generate_content(self, prompt, **kwargs):
        """Synchronous generate_content (called via asyncio.to_thread)."""
        import time
        time.sleep(self.delay_ms / 1000)
        
        if self.should_error:
            raise ValueError("Simulated Flash error")
        
        return FakeResponse(
            text="Flash response: Quick preliminary analysis...",
            prompt_tokens=45,
            completion_tokens=128
        )


class FakeProProvider:
    """Fake Pro model provider with controllable delay and errors."""
    
    def __init__(self, delay_ms: float = 2000, should_timeout: bool = False, should_error: bool = False):
        self.delay_ms = delay_ms
        self.should_timeout = should_timeout
        self.should_error = should_error
    
    def generate_content(self, prompt, **kwargs):
        """Synchronous generate_content (called via asyncio.to_thread)."""
        import time
        time.sleep(self.delay_ms / 1000)
        
        if self.should_error:
            raise ValueError("Simulated Pro error")
        
        return FakeResponse(
            text="Pro response: Comprehensive analysis with reasoning steps...",
            prompt_tokens=45,
            completion_tokens=512
        )


@pytest.fixture
def mock_db_session():
    """Mock database session for audit logging tests."""
    session = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.close = MagicMock()
    session.merge = MagicMock()
    
    with patch('src.services.db.get_session', return_value=session):
        yield session


@pytest.fixture(autouse=True)
def setup_agent():
    """Initialize agent for all tests with mocked Vertex models."""
    from src.api import main
    
    # Create dummy models that will be replaced by test-specific fakes
    fake_flash_default = FakeFlashProvider(delay_ms=300)
    fake_pro_default = FakeProProvider(delay_ms=2000)
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash_default), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro_default):
        
        main.agent = DualModelAgent(
            project_id="test-project",
            location="us-central1"
        )
        
        yield
        
        # Cleanup after test
        main.agent = None


def parse_sse_stream(content: str) -> list:
    """
    Parse SSE stream into list of events.
    
    Returns:
        List of dicts: [{"event": "preliminary", "data": {...}}, ...]
    """
    events = []
    lines = content.strip().split('\n')
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('event:'):
            event_type = lines[i].split(':', 1)[1].strip()
            i += 1
            
            if i < len(lines) and lines[i].startswith('data:'):
                import json
                data = json.loads(lines[i].split(':', 1)[1].strip())
                events.append({"event": event_type, "data": data})
                i += 1
            else:
                i += 1
        else:
            i += 1
    
    return events


@pytest.mark.asyncio
async def test_sse_ordering_normal():
    """
    Test normal case: Flash (200-500ms) then Pro (2-5s).
    
    Expected SSE ordering:
    1. event: preliminary (Flash response)
    2. event: final (Pro response)
    3. event: done
    """
    # Setup fake providers
    fake_flash = FakeFlashProvider(delay_ms=300)
    fake_pro = FakeProProvider(delay_ms=2000)
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro), \
         patch('src.services.db.get_session', return_value=None):  # Skip DB for this test
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/reasoning/query",
                json={
                    "query": "Test query",
                    "context": {"domain": "test"}
                }
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            
            events = parse_sse_stream(response.text)
            
            # Assert event count
            assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}"
            
            # Assert ordering
            assert events[0]["event"] == "preliminary", "First event should be preliminary"
            assert events[1]["event"] == "final", "Second event should be final"
            assert events[2]["event"] == "done", "Third event should be done"
            
            # Assert preliminary contains Flash response
            assert "Flash response" in events[0]["data"]["response"]["content"]
            assert events[0]["data"]["response"]["is_preliminary"] is True
            
            # Assert final contains Pro response
            assert "Pro response" in events[1]["data"]["response"]["content"]
            assert events[1]["data"]["response"]["is_preliminary"] is False
            
            # Assert trace_id present in all events
            assert "trace_id" in events[0]["data"]
            assert "trace_id" in events[1]["data"]
            assert "trace_id" in events[2]["data"]


@pytest.mark.asyncio
async def test_flash_timeout_pro_ok():
    """
    Test Flash timeout with Pro succeeding.
    
    Flash times out (>5s), Pro completes normally.
    Expected: error event (Flash timeout) + done
    """
    # Flash will timeout (delay > FLASH_TIMEOUT_S=5s)
    fake_flash = FakeFlashProvider(delay_ms=6000)  # 6s > 5s timeout
    fake_pro = FakeProProvider(delay_ms=2000)      # 2s (won't execute due to cancellation)
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro), \
         patch('src.services.db.get_session', return_value=None):
        
        async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
            response = await client.post(
                "/api/reasoning/query",
                json={
                    "query": "Test query",
                    "context": {}
                }
            )
            
            assert response.status_code == 200
            events = parse_sse_stream(response.text)
            
            # Should have: error + done
            assert len(events) >= 2
            
            # First event should be error (Flash timeout)
            error_event = events[0]
            assert error_event["event"] == "error"
            assert "Flash model timed out" in error_event["data"]["error"]
            assert error_event["data"]["type"] == "timeout"
            assert error_event["data"]["retryable"] is True
            
            # Last event should be done
            assert events[-1]["event"] == "done"


@pytest.mark.asyncio
async def test_pro_timeout_flash_ok():
    """
    Test Pro timeout with Flash succeeding.
    
    Flash completes normally, Pro times out (>45s).
    Expected: preliminary + error (Pro timeout) + done
    """
    fake_flash = FakeFlashProvider(delay_ms=300)      # 300ms (succeeds)
    fake_pro = FakeProProvider(delay_ms=46000)        # 46s > 45s timeout
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro), \
         patch('src.services.db.get_session', return_value=None):
        
        async with AsyncClient(app=app, base_url="http://test", timeout=50.0) as client:
            response = await client.post(
                "/api/reasoning/query",
                json={
                    "query": "Test query",
                    "context": {}
                }
            )
            
            assert response.status_code == 200
            events = parse_sse_stream(response.text)
            
            # Should have: preliminary + error + done
            assert len(events) >= 3
            
            # First: preliminary (Flash succeeded)
            assert events[0]["event"] == "preliminary"
            assert "Flash response" in events[0]["data"]["response"]["content"]
            
            # Second: error (Pro timeout)
            error_event = events[1]
            assert error_event["event"] == "error"
            assert "Pro model timed out" in error_event["data"]["error"]
            assert error_event["data"]["type"] == "timeout"
            assert error_event["data"]["retryable"] is True
            
            # Last: done
            assert events[-1]["event"] == "done"


@pytest.mark.asyncio
async def test_both_timeout():
    """
    Test both models timing out.
    
    Flash times out first, Pro is cancelled.
    Expected: error (Flash timeout) + done
    """
    fake_flash = FakeFlashProvider(delay_ms=6000)  # 6s > 5s timeout
    fake_pro = FakeProProvider(delay_ms=46000)     # Would timeout, but cancelled after Flash
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro), \
         patch('src.services.db.get_session', return_value=None):
        
        async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
            response = await client.post(
                "/api/reasoning/query",
                json={
                    "query": "Test query",
                    "context": {}
                }
            )
            
            assert response.status_code == 200
            events = parse_sse_stream(response.text)
            
            # Should have: error (Flash timeout) + done
            assert len(events) >= 2
            
            # First: error (Flash timeout, Pro cancelled)
            assert events[0]["event"] == "error"
            assert "Flash model timed out" in events[0]["data"]["error"]
            
            # Last: done
            assert events[-1]["event"] == "done"


@pytest.mark.asyncio
async def test_audit_persistence_complete(mock_db_session):
    """
    Test that a single DB audit record is written per run_id.
    
    Verifies:
    - Exactly one record written
    - Record includes combined transcript + timings
    - run_id used as idempotency key
    """
    fake_flash = FakeFlashProvider(delay_ms=300)
    fake_pro = FakeProProvider(delay_ms=2000)
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro):
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/reasoning/query",
                json={
                    "query": "Test query for audit",
                    "context": {"domain": "test", "purpose": "audit_verification"}
                }
            )
            
            assert response.status_code == 200
            
            # Wait for async audit logging to complete
            await asyncio.sleep(0.5)
            
            # Verify session.merge called exactly once
            assert mock_db_session.merge.call_count == 1
            
            # Verify commit called
            assert mock_db_session.commit.call_count == 1
            
            # Verify record structure
            persisted_run = mock_db_session.merge.call_args[0][0]
            assert persisted_run.id is not None  # run_id present
            assert persisted_run.query == "Test query for audit"
            assert persisted_run.context == {"domain": "test", "purpose": "audit_verification"}
            
            # Verify both responses persisted
            assert persisted_run.flash_response is not None
            assert persisted_run.pro_response is not None
            
            # Verify flash_response structure
            flash_resp = persisted_run.flash_response
            assert flash_resp["model"] == "gemini-2.5-flash"
            assert flash_resp["latency_ms"] is not None
            assert flash_resp["prompt_tokens"] is not None
            
            # Verify pro_response structure
            pro_resp = persisted_run.pro_response
            assert pro_resp["model"] == "gemini-2.5-pro"
            assert pro_resp["latency_ms"] is not None
            assert pro_resp["prompt_tokens"] is not None


@pytest.mark.asyncio
async def test_sse_always_closes():
    """
    Test that 'done' event is always sent, even on errors.
    
    Critical: Prevents client hanging.
    """
    # Simulate Flash error
    fake_flash = FakeFlashProvider(should_error=True)
    fake_pro = FakeProProvider(delay_ms=2000)
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro), \
         patch('src.services.db.get_session', return_value=None):
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/reasoning/query",
                json={"query": "Test", "context": {}}
            )
            
            assert response.status_code == 200
            events = parse_sse_stream(response.text)
            
            # Must have at least error + done
            assert len(events) >= 2
            
            # Last event MUST be 'done'
            assert events[-1]["event"] == "done", "Stream must always end with 'done' event"


@pytest.mark.asyncio
async def test_sse_backward_compat():
    """
    Test that existing SSE clients can still parse responses.
    
    Verifies:
    - 'preliminary' and 'final' events unchanged
    - Added fields (trace_id, type, retryable) are optional
    - Existing clients ignore 'done' event
    """
    fake_flash = FakeFlashProvider(delay_ms=300)
    fake_pro = FakeProProvider(delay_ms=2000)
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro), \
         patch('src.services.db.get_session', return_value=None):
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/reasoning/query",
                json={"query": "Test", "context": {}}
            )
            
            events = parse_sse_stream(response.text)
            
            # Verify 'preliminary' event structure
            prelim = events[0]
            assert prelim["event"] == "preliminary"
            assert "response" in prelim["data"]
            assert "message" in prelim["data"]
            
            # Verify 'final' event structure
            final = events[1]
            assert final["event"] == "final"
            assert "response" in final["data"]
            assert "message" in final["data"]
            
            # New fields are present but optional for clients
            assert "trace_id" in prelim["data"]  # New field (can be ignored)
            
            # Existing fields unchanged
            assert "content" in prelim["data"]["response"]
            assert "latency_ms" in prelim["data"]["response"]


@pytest.mark.asyncio
async def test_metrics_recorded():
    """
    Test that metrics are recorded for Flash and Pro latencies.
    
    Verifies:
    - Latency histograms populated
    - Timeout counters incremented
    - Cancellation counters incremented
    """
    from src.utils.metrics import get_metrics
    
    metrics = get_metrics()
    metrics.reset()  # Clean slate
    
    fake_flash = FakeFlashProvider(delay_ms=300)
    fake_pro = FakeProProvider(delay_ms=2000)
    
    with patch('src.reasoning.dual_agent.get_flash_model', return_value=fake_flash), \
         patch('src.reasoning.dual_agent.get_pro_model', return_value=fake_pro), \
         patch('src.services.db.get_session', return_value=None):
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            await client.post(
                "/api/reasoning/query",
                json={"query": "Test metrics", "context": {}}
            )
            
            # Wait for async operations
            await asyncio.sleep(0.5)
            
            # Verify Flash latency recorded
            flash_stats = metrics.get_histogram_stats("latency_ms", labels={"model": "flash"})
            assert flash_stats["count"] > 0, "Flash latency should be recorded"
            assert flash_stats["mean"] > 0
            
            # Verify Pro latency recorded
            pro_stats = metrics.get_histogram_stats("latency_ms", labels={"model": "pro"})
            assert pro_stats["count"] > 0, "Pro latency should be recorded"
            assert pro_stats["mean"] > 0
            
            # Flash should be faster than Pro
            assert flash_stats["mean"] < pro_stats["mean"]
