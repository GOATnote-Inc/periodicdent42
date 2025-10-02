# Ticket: Decide Labloop vs Services Canonical Stack

## Context
- `labloop/` directory ships its own FastAPI orchestrator, scheduler, UI, and tests.
- Production endpoints (`apps/api`, `services/*`) diverge from this slice yet share concepts (plans, instrumentation).

## Problem Statement
Maintainers are confused which stack is production. Bug fixes risk landing in one slice only.

## Reproduction Steps
1. Clone repo fresh.
2. Follow top-level README – no mention of `labloop`.
3. Run `uvicorn labloop.orchestrator.main:app --reload` – launches a competing API not referenced elsewhere.

## Proposed Plan
- Run architecture comparison workshop to list required `labloop` features still missing from `services/*`.
- Either (a) migrate critical components into `services/*` and archive `labloop` under `archive/20251002/`, or (b) update README + Make targets to expose `labloop` as supported vertical slice.
- Add CI guard preventing drift once decision made.

## Acceptance Criteria
- Documented canonical stack in README and CONTRIBUTING.
- Redundant code either archived or linked with explicit ownership.
- No orphaned Make targets referencing archived directories.

## Test Plan
- Smoke test chosen canonical API (`make run.api` or `uvicorn labloop...`).
- `make graph` continues to succeed without referencing archived directories.
