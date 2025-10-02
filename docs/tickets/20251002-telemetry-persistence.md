# Ticket: Persist Telemetry Store

## Context
- Current implementation (`services/telemetry/store.py`) only retains chat runs in memory.
- Product needs reproducible analytics and guardrail QA even after process restarts.

## Problem Statement
Telemetry records disappear after reload and cannot be inspected by tests or dashboards.

## Reproduction Steps
1. Start API (`make run.api`).
2. Call `/api/chat` twice.
3. Restart server – `/services/telemetry/store.TelemetryStore.records` resets to empty.

## Proposed Plan
- Introduce `TelemetryRepository` interface with `InMemoryTelemetryStore` and `FileTelemetryStore` implementations.
- Add `.env` toggle (`TELEMETRY_STORE_PATH`) to persist as newline-delimited JSON.
- Update FastAPI dependency injection to use file-backed store by default when path is set.
- Write pytest covering persistence + schema validation.

## Acceptance Criteria
- Telemetry persists across process restarts when path is configured.
- Unit tests cover serialization/deserialization round trip.
- README updated with storage behaviour.

## Test Plan
- `pytest services/tests/test_telemetry_store.py` (new).
- Manual smoke: trigger chat, restart, confirm file contains records and UI displays them.
