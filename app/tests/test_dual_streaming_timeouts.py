"""
Deterministic timeout tests for dual-model SSE streaming.

Uses tiny timeouts (0.05s) and fake models with longer delays to trigger
real asyncio.TimeoutError in asyncio.wait_for().

All tests verify:
1. Timeout errors are emitted as SSE events
2. Stream always closes with 'done' event (never hangs)
3. Error events include type and retryable fields
"""

import json
import pytest

from .conftest import FakeModel, sse_events

# Adjust endpoint path to our actual API
STREAM_PATH = "/api/reasoning/query"
PAYLOAD = {"query": "timeout test", "context": {}}


@pytest.mark.asyncio
async def test_pro_timeout_emits_error_then_done(client, set_models, mock_db):
    """
    Test Pro timeout with Flash succeeding.
    
    Flash is fast (0.01s) → preliminary should appear.
    Pro is slow (0.20s > 0.05s timeout) → hits timeout.
    
    Expected SSE ordering:
    1. preliminary (Flash response)
    2. error (Pro timeout)
    3. done
    """
    set_models(
        flash=FakeModel(delay=0.01, text="flash-ok"),
        pro=FakeModel(delay=0.20, text="pro-slow")  # > PRO_TIMEOUT_S=0.05
    )
    
    async with client as c:
        async with c.stream("POST", STREAM_PATH, json=PAYLOAD) as r:
            events = []
            error_data = None
            
            async for ev, data in sse_events(r.aiter_lines()):
                events.append(ev)
                if ev == "error":
                    error_data = json.loads(data)
            
            # Assert ordering
            assert "preliminary" in events, f"Expected preliminary event, got: {events}"
            assert "error" in events, f"Expected error event, got: {events}"
            assert events[-1] == "done", f"Expected done as final event, got: {events}"
            
            # Assert error structure
            assert error_data is not None
            assert "Pro model timed out" in error_data["error"]
            assert error_data["type"] == "timeout"
            assert error_data["retryable"] is True


@pytest.mark.asyncio
async def test_flash_timeout_emits_error_then_done(client, set_models, mock_db):
    """
    Test Flash timeout (Pro may or may not execute depending on cancellation policy).
    
    Flash is slow (0.20s > 0.05s timeout) → triggers timeout before producing preliminary.
    Pro is fast (0.01s) but is cancelled when Flash times out.
    
    Expected SSE ordering:
    1. error (Flash timeout)
    2. done
    
    Note: We do NOT expect 'final' because Pro is cancelled when Flash fails.
    """
    set_models(
        flash=FakeModel(delay=0.20, text="flash-slow"),  # > FLASH_TIMEOUT_S=0.05
        pro=FakeModel(delay=0.01, text="pro-ok")
    )
    
    async with client as c:
        async with c.stream("POST", STREAM_PATH, json=PAYLOAD) as r:
            events = []
            error_data = None
            
            async for ev, data in sse_events(r.aiter_lines()):
                events.append(ev)
                if ev == "error":
                    error_data = json.loads(data)
            
            # Assert ordering
            assert "error" in events, f"Expected error event, got: {events}"
            assert events[-1] == "done", f"Expected done as final event, got: {events}"
            
            # Assert error structure
            assert error_data is not None
            assert "Flash model timed out" in error_data["error"]
            assert error_data["type"] == "timeout"
            assert error_data["retryable"] is True
            
            # Should NOT have preliminary (Flash timed out before producing)
            assert "preliminary" not in events, "Flash timed out, should not produce preliminary"
            
            # Should NOT have final (Pro cancelled when Flash failed)
            assert "final" not in events, "Pro should be cancelled when Flash times out"


@pytest.mark.asyncio
async def test_both_timeout_single_error_then_done(client, set_models, mock_db):
    """
    Test both models exceeding timeout budgets.
    
    Both Flash and Pro are slow (0.20s > 0.05s timeout).
    Flash times out first, Pro is cancelled.
    
    Expected SSE ordering:
    1. error (Flash timeout)
    2. done
    """
    set_models(
        flash=FakeModel(delay=0.20, text="flash-slow"),
        pro=FakeModel(delay=0.20, text="pro-slow")
    )
    
    async with client as c:
        async with c.stream("POST", STREAM_PATH, json=PAYLOAD) as r:
            events = []
            error_data = None
            
            async for ev, data in sse_events(r.aiter_lines()):
                events.append(ev)
                if ev == "error":
                    error_data = json.loads(data)
            
            # Assert ordering
            assert "error" in events, f"Expected error event, got: {events}"
            assert events[-1] == "done", f"Expected done as final event, got: {events}"
            
            # Assert error structure
            assert error_data is not None
            assert "timed out" in error_data["error"].lower()
            assert error_data["type"] == "timeout"
            assert error_data["retryable"] is True
            
            # Should NOT have preliminary or final (both timed out)
            assert "preliminary" not in events
            assert "final" not in events


@pytest.mark.asyncio
async def test_normal_case_no_timeouts(client, set_models, mock_db):
    """
    Sanity check: Normal case with fast models (no timeouts).
    
    Both models complete within timeout budget.
    
    Expected SSE ordering:
    1. preliminary (Flash response)
    2. final (Pro response)
    3. done
    """
    set_models(
        flash=FakeModel(delay=0.01, text="flash-ok"),
        pro=FakeModel(delay=0.02, text="pro-ok")
    )
    
    async with client as c:
        async with c.stream("POST", STREAM_PATH, json=PAYLOAD) as r:
            events = []
            
            async for ev, data in sse_events(r.aiter_lines()):
                events.append(ev)
            
            # Assert ordering
            assert events == ["preliminary", "final", "done"], f"Expected normal ordering, got: {events}"


@pytest.mark.asyncio
async def test_stream_never_hangs_on_exception(client, set_models, mock_db):
    """
    Test that stream always closes with 'done' even on unexpected exceptions.
    
    Flash raises an exception (not timeout).
    
    Expected SSE ordering:
    1. error (Flash exception)
    2. done (always emitted)
    """
    set_models(
        flash=FakeModel(delay=0.01, exc=ValueError("Simulated Flash error")),
        pro=FakeModel(delay=0.01, text="pro-ok")
    )
    
    async with client as c:
        async with c.stream("POST", STREAM_PATH, json=PAYLOAD) as r:
            events = []
            
            async for ev, data in sse_events(r.aiter_lines()):
                events.append(ev)
            
            # Assert 'done' is always last
            assert events[-1] == "done", f"Stream must always end with 'done', got: {events}"
            assert "error" in events, f"Expected error event, got: {events}"