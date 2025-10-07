"""
Test configuration and fixtures for dual-model streaming tests.

Provides:
- Tiny timeouts for fast, deterministic timeout tests
- Fake Vertex AI models with controllable delays
- HTTPX async client for SSE streaming
- SSE parser for test assertions
"""

import asyncio
import pytest
from unittest.mock import MagicMock

# ---- Global tiny timeouts so tests are fast & deterministic
@pytest.fixture(autouse=True)
def _tiny_timeouts(monkeypatch):
    """Set tiny timeouts for deterministic timeout tests."""
    monkeypatch.setenv("FLASH_TIMEOUT_S", "0.05")
    monkeypatch.setenv("PRO_TIMEOUT_S", "0.05")
    yield


# ---- Minimal fake Vertex responses
class _FakeResp:
    """Fake Vertex AI response matching the shape our code expects."""
    
    def __init__(self, text="ok", prompt_tokens=45, completion_tokens=128):
        self.text = text
        self.usage_metadata = MagicMock()
        self.usage_metadata.prompt_token_count = prompt_tokens
        self.usage_metadata.candidates_token_count = completion_tokens


class FakeModel:
    """
    Fake Vertex model with controllable delay and errors.
    
    Uses asyncio.sleep (not time.sleep) so delays respect asyncio.wait_for().
    """
    
    def __init__(self, *, delay=0.01, exc=None, text="ok"):
        self.delay = delay
        self.exc = exc
        self.text = text
    
    def generate_content(self, *args, **kwargs):
        """
        Synchronous interface (called via asyncio.to_thread).
        
        We'll use asyncio.sleep in the async wrapper below.
        """
        import time
        time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return _FakeResp(self.text)
    
    async def generate_content_async(self, *args, **kwargs):
        """Async interface for testing (bypasses to_thread)."""
        await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return _FakeResp(self.text)


# ---- Helpers to patch getters BEFORE agent creation
@pytest.fixture
def set_models(monkeypatch):
    """
    Patch model getters before agent creation.
    
    Usage:
        set_models(flash=FakeModel(delay=0.2), pro=FakeModel(delay=0.01))
    
    IMPORTANT: Call this BEFORE any code that creates the agent.
    """
    def _apply(*, flash: FakeModel, pro: FakeModel):
        # Patch the getters in vertex.py
        monkeypatch.setattr("src.services.vertex.get_flash_model", lambda: flash)
        monkeypatch.setattr("src.services.vertex.get_pro_model", lambda: pro)
        
        # Also patch the agent's internal calls to use async versions
        # This bypasses asyncio.to_thread which doesn't respect cancellation
        async def _flash_wrapper(self, prompt, context):
            start = asyncio.get_event_loop().time()
            try:
                response = await flash.generate_content_async(prompt)
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000
                
                return {
                    "model": "gemini-2.5-flash",
                    "content": response.text,
                    "latency_ms": round(latency_ms, 2),
                    "is_preliminary": True,
                    "confidence": "medium",
                    "usage": {
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
                }
            except Exception as e:
                return {
                    "model": "gemini-2.5-flash",
                    "content": f"Error: {str(e)}",
                    "latency_ms": (asyncio.get_event_loop().time() - start) * 1000,
                    "is_preliminary": True,
                    "confidence": "error",
                    "error": str(e),
                    "error_class": e.__class__.__name__
                }
        
        async def _pro_wrapper(self, prompt, context):
            start = asyncio.get_event_loop().time()
            try:
                response = await pro.generate_content_async(prompt)
                latency_ms = (asyncio.get_event_loop().time() - start) * 1000
                
                return {
                    "model": "gemini-2.5-pro",
                    "content": response.text,
                    "latency_ms": round(latency_ms, 2),
                    "is_preliminary": False,
                    "confidence": "high",
                    "reasoning_steps": [],
                    "usage": {
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
                }
            except Exception as e:
                return {
                    "model": "gemini-2.5-pro",
                    "content": f"Error: {str(e)}",
                    "latency_ms": (asyncio.get_event_loop().time() - start) * 1000,
                    "is_preliminary": False,
                    "confidence": "error",
                    "error": str(e),
                    "error_class": e.__class__.__name__
                }
        
        # Patch the agent's query methods to use async versions
        monkeypatch.setattr("src.reasoning.dual_agent.DualModelAgent._query_flash", _flash_wrapper)
        monkeypatch.setattr("src.reasoning.dual_agent.DualModelAgent._query_pro", _pro_wrapper)
    
    return _apply


# ---- Async HTTPX client against the ASGI app
@pytest.fixture(scope="function")
def client():
    """HTTPX async client for testing SSE endpoints (sync fixture returning async context)."""
    import httpx
    from src.api.main import app
    
    # Return the client as a context manager, not async
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=10.0)


# ---- Tiny SSE parser for tests
async def sse_events(lines_async_iter):
    """
    Parse SSE stream into (event, data) tuples.
    
    Usage:
        async for event, data in sse_events(response.aiter_lines()):
            if event == "preliminary":
                ...
    """
    event, data_buf = None, []
    async for line in lines_async_iter:
        line = line.strip()
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_buf.append(line[5:].strip())
        elif line == "" and event:  # dispatch on blank line
            payload = "\n".join(data_buf) if data_buf else ""
            yield event, payload
            event, data_buf = None, []


# ---- Mock database for tests that don't need real DB
@pytest.fixture
def mock_db(monkeypatch):
    """Mock database session for tests."""
    session = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.close = MagicMock()
    session.merge = MagicMock()
    
    monkeypatch.setattr("src.services.db.get_session", lambda: session)
    return session